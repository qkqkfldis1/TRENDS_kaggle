import os
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold,StratifiedKFold, GroupKFold, KFold
import nilearn as nl
import torch
import random
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
import itertools

import monai
from monai.transforms import \
    LoadNifti, LoadNiftid, AddChanneld, ScaleIntensityRanged, \
    Rand3DElasticd, RandAffined, \
    Spacingd, Orientationd

root = r'/home/youhanlee/project/trends/input/'
#root = r'/home/iclab/projects/Trends/input/'

train = pd.read_csv('{}/train_scores.csv'.format(root)).sort_values(by='Id')
loadings = pd.read_csv('{}/loading.csv'.format(root))
sample = pd.read_csv('{}/sample_submission.csv'.format(root))
reveal = pd.read_csv('{}/reveal_ID_site2.csv'.format(root))
ICN = pd.read_csv('{}/ICN_numbers.csv'.format(root))
fnc = pd.read_csv('{}/fnc.csv'.format(root))



"""
    Load and display a subject's spatial map
"""

def load_subject(filename, mask_niimg):
    """
    Load a subject saved in .mat format with the version 7.3 flag. Return the subject niimg, using a mask niimg as a template for nifti headers.
    Args:
        filename    <str>            the .mat filename for the subject data
        mask_niimg  niimg object     the mask niimg object used for nifti headers
    """
    subject_data = None
    with h5py.File(filename, 'r') as f:
        subject_data = f['SM_feature'][()]
        # print(subject_data.shape)
    # It's necessary to reorient the axes, since h5py flips axis order
    subject_data = np.moveaxis(subject_data, [0, 1, 2, 3], [3, 2, 1, 0])
    # print(subject_data.shape)
    return subject_data
    # subject_niimg = nl.image.new_img_like(mask_niimg, subject_data, affine=mask_niimg.affine, copy_header=True)
    # return subject_niimg

def read_data_sample():
    # Input data files are available in the "../input/" directory.
    # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
    mask_filename = r'{}/fMRI_mask.nii'.format(root)
    subject_filename = '{}/fMRI_train/10004.mat'.format(root)

    mask_niimg = nl.image.load_img(mask_filename)
    print("mask shape is %s" % (str(mask_niimg.shape)))

    subject_niimg = load_subject(subject_filename, mask_niimg)
    print("Image shape is %s" % (str(subject_niimg.shape)))
    num_components = subject_niimg.shape[-1]
    print("Detected {num_components} spatial maps".format(num_components=num_components))

class TReNDsDataset(Dataset):

    def __init__(self, mode='train', fold_index = 0):
        # print("Processing {} datas".format(len(self.img_list)))
        self.mode = mode
        self.fold_index = fold_index
        fnc = pd.read_csv('{}/fnc.csv'.format(root))
        fnc_columns = fnc.columns[1:]
        fnc[fnc_columns] /= 500
        
        degree = pd.read_csv('{}/degree.csv'.format(root))
        degree_columns = degree.columns[:-1]
        degree[degree_columns] /= 10000
        
        if self.mode=='train' or self.mode=='valid' or self.mode=='valid_tta':
            features = ('age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2')
            good_feats = [f'IC_{feat}' for feat in ['15', '22', '06', '21', '04', '11', '10', '16']]
            data = pd.merge(loadings, train, on='Id').dropna()
            comb = itertools.combinations(good_feats, 2)
            for col1, col2 in comb:
                data[f'{col1}_x_{col2}'] = data[col1] * data[col2]
                data[f'{col1}_+_{col2}'] = data[col1] + data[col2]
            
            id_train = list(data.Id)
            fea_train = np.asarray(data.drop(list(features), axis=1).drop('Id', axis=1))
            lbl_train = np.asarray(data[list(features)])
            
            # fnc
            data = pd.merge(fnc, train, on='Id').dropna()
            fnc_train = np.asarray(data.drop(list(features), axis=1).drop('Id', axis=1))
            
            # degree
            data = pd.merge(degree, train, on='Id').dropna()
            deg_train = np.asarray(data.drop(list(features), axis=1).drop('Id', axis=1))

            self.all_samples = []
            for i in tqdm(range(len(id_train))):
                id = id_train[i]
                fea = fea_train[i]
                fnc = fnc_train[i]
                deg = deg_train[i]
                lbl = lbl_train[i]
                filename = os.path.join('{}/fMRI_train_npy/{}.npy'.format(root, id))
                self.all_samples.append([filename, fea, fnc, deg, lbl, str(id)])
                
            fold_df = data[list(features)]
            min_value = fold_df['age'].min()
            max_value = fold_df['age'].max()
            binsplits = np.arange(min_value,max_value,4)
            y_categorized = np.digitize(fold_df['age'].values, bins=binsplits)
            
            fold_df['age_cat'] = y_categorized

            train_indexes = []
            valid_indexes = []
            kf = StratifiedShuffleSplit(n_splits=5,  random_state=42)
            for trn_idx, vld_idx in kf.split(fold_df, fold_df['age_cat']):
                train_indexes.append(trn_idx)
                valid_indexes.append(vld_idx)
                
            self.train_index = train_indexes[fold_index]
            self.valid_index = valid_indexes[fold_index]

            if self.mode=='train':
                self.train_index = [tmp for tmp in self.train_index if os.path.exists(self.all_samples[tmp][0])]
                self.len = len(self.train_index)
                print('fold index:',fold_index)
                print('train num:', self.len)

            elif self.mode=='valid' or self.mode=='valid_tta':
                self.valid_index = [tmp for tmp in self.valid_index if os.path.exists(self.all_samples[tmp][0])]
                self.len = len(self.valid_index)
                print('fold index:',fold_index)
                print('valid num:', self.len)

        elif  self.mode=='test':
            labels_df = pd.read_csv("{}/train_scores.csv".format(root))
            labels_df["is_train"] = True

            features = ('age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2')
            data = pd.merge(loadings, labels_df, on="Id", how="left")
            
            good_feats = [f'IC_{feat}' for feat in ['15', '22', '06', '21', '04', '11', '10', '16']]
            #data = pd.merge(loadings, train, on='Id').dropna()
            comb = itertools.combinations(good_feats, 2)
            for col1, col2 in comb:
                data[f'{col1}_x_{col2}'] = data[col1] * data[col2]
                data[f'{col1}_+_{col2}'] = data[col1] + data[col2]

            id_test = list(data[data["is_train"] != True].Id)
            fea_test = np.asarray(data.drop(list(features), axis=1).drop('Id', axis=1)[data["is_train"] != True].drop("is_train", axis=1))
            lbl_test = np.asarray(data[list(features)][data["is_train"] != True])
            
            data = pd.merge(fnc, labels_df, on="Id", how="left")
            fnc_test = np.asarray(data.drop(list(features), axis=1).drop('Id', axis=1)[data["is_train"] != True].drop("is_train", axis=1))
            
            # degree
            data = pd.merge(degree, labels_df, on='Id', how='left')
            deg_test = np.asarray(data.drop(list(features), axis=1).drop('Id', axis=1)[data["is_train"] != True].drop("is_train", axis=1))
            #print(deg_test.shape)

            self.all_samples = []
            for i in range(len(id_test)):
                id = id_test[i]
                fea = fea_test[i]
                fnc = fnc_test[i]
                deg = deg_test[i]
                lbl = lbl_test[i]

                filename = os.path.join('{}/fMRI_test_npy/{}.npy'.format(root, id))
                if os.path.exists(filename):
                    self.all_samples.append([id, filename, fea, fnc, deg, lbl])
                    #all_samples.append([filename, fea, fnc, lbl, str(id)])

            self.len = len(self.all_samples)
            print(len(id_test))
            print('test num:', self.len)

    def __getitem__(self, idx):

        if self.mode == "train" :
            filename, feat, fnc, deg, lbl, id =  self.all_samples[self.train_index[idx]]
            train_img = np.load(filename).astype(np.float32)
            train_img = train_img.transpose((3,2,1,0))
            # (53, 52, 63, 53)
            train_lbl = lbl

            data_dict = {'image': train_img}
            rand_affine = RandAffined(keys=['image'],
                                      mode=('bilinear', 'nearest'),
                                      prob=0.5,
                                      spatial_size=(52, 63, 53),
                                      translate_range=(5, 5, 5),
                                      rotate_range=(np.pi * 4, np.pi * 4, np.pi * 4),
                                      scale_range=(0.15, 0.15, 0.15),
                                      padding_mode='border')
            affined_data_dict = rand_affine(data_dict)
            train_img = affined_data_dict['image']
            train_feat = feat #affined_data_dict['feat']
            train_fnc = fnc#affined_data_dict['fnc']
            train_deg = deg#affined_data_dict['deg']

            return torch.FloatTensor(train_img), \
                   torch.FloatTensor(train_feat), \
                   torch.FloatTensor(train_fnc), \
                   torch.FloatTensor(train_deg), \
                   torch.FloatTensor(train_lbl)

        elif self.mode == "valid":
            filename, feat, fnc, deg, lbl, id =  self.all_samples[self.valid_index[idx]]
            train_img = np.load(filename).astype(np.float32)
            train_img = train_img.transpose((3, 2, 1, 0))
            # (53, 52, 63, 53)
            train_lbl = lbl

            return torch.FloatTensor(train_img), \
                   torch.FloatTensor(feat), \
                   torch.FloatTensor(fnc), \
                   torch.FloatTensor(deg), \
                   torch.FloatTensor(train_lbl)


        elif self.mode == 'test':
            id, filename, feat, fnc, deg, lbl =  self.all_samples[idx]
            test_img = np.load(filename).astype(np.float32)
            test_img = test_img.transpose((3, 2, 1, 0))
            
            return str(id), \
                   torch.FloatTensor(test_img), \
                   torch.FloatTensor(feat), \
                   torch.FloatTensor(fnc), \
                   torch.FloatTensor(deg)

    def __len__(self):
        return self.len

def run_check_datasets():
    dataset = TReNDsDataset(mode='test')
    for m in range(len(dataset)):
        tmp = dataset[m]
        print(m)

def convert_mat2nii2npy():

    def get_data(filename):
        with h5py.File(filename, 'r') as f:
            subject_data = f['SM_feature'][()]
            # print(subject_data.shape)
        # It's necessary to reorient the axes, since h5py flips axis order
        subject_data = np.moveaxis(subject_data, [0, 1, 2, 3], [3, 2, 1, 0])
        return subject_data

    # train_root = '{}/fMRI_train/'.format(root)
    # train_npy_root = '{}/fMRI_train_npy/'.format(root)
    train_root = '{}/fMRI_test/'.format(root)
    train_npy_root = '{}/fMRI_test_npy/'.format(root)
    os.makedirs(train_npy_root, exist_ok=True)

    mats = os.listdir(train_root)
    mats = [mat for mat in mats if '.mat' in mat]
    random.shuffle(mats)

    for mat in tqdm(mats):
        mat_path = os.path.join(train_root, mat)
        if os.path.exists(mat_path):
            print(mat_path)

        npy_path = os.path.join(train_npy_root, mat.replace('.mat','.npy'))
        if os.path.exists(npy_path):
            print(npy_path, 'exist')
        else:
            data = get_data(mat_path)
            print(npy_path,data.shape)
            np.save(npy_path,data.astype(np.float16))

if __name__ == '__main__':
    run_check_datasets()
    # convert_mat2nii2npy()
