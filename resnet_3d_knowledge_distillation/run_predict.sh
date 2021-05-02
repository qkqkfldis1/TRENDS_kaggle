#rm -rf TReNDs

python test_TReNDs.py --model resnet --model_depth 34 --resnet_shortcut A --batch_size 128 --num_workers 4 --n_epochs 50 --fold_index 0 --feat_index 0\
	--resume_path fold_0_feat_0_epoch_15_batch_81_loss_0.1450279951095581.pth.tar

python test_TReNDs.py --model resnet --model_depth 34 --resnet_shortcut A --batch_size 128 --num_workers 4 --n_epochs 50 --fold_index 1 --feat_index 0 \
	--resume_path fold_1_feat_0_epoch_23_batch_81_loss_0.1413227617740631.pth.tar

python test_TReNDs.py --model resnet --model_depth 34 --resnet_shortcut A --batch_size 128 --num_workers 4 --n_epochs 50 --fold_index 2 --feat_index 0 \
	--resume_path fold_2_feat_0_epoch_23_batch_81_loss_0.13422970473766327.pth.tar

python test_TReNDs.py --model resnet --model_depth 34 --resnet_shortcut A --batch_size 128 --num_workers 4 --n_epochs 50 --fold_index 3 --feat_index 0 \
	--resume_path fold_3_feat_0_epoch_13_batch_81_loss_0.13834697008132935.pth.tar

python test_TReNDs.py --model resnet --model_depth 34 --resnet_shortcut A --batch_size 128 --num_workers 4 --n_epochs 50 --fold_index 4 --feat_index 0 \
	--resume_path fold_4_feat_0_epoch_34_batch_81_loss_0.14502757787704468.pth.tar
    
    
#python train_TReNDs.py --model resnet --model_depth 50 --resnet_shortcut B --batch_size 128 --num_workers 8 --n_epochs 50 --fold_index 1
#python train_TReNDs.py --model resnet --model_depth 50 --resnet_shortcut B --batch_size 128 --num_workers 8 --n_epochs 50 --fold_index 2
#python train_TReNDs.py --model resnet --model_depth 50 --resnet_shortcut B --batch_size 128 --num_workers 8 --n_epochs 50 --fold_index 3
#python train_TReNDs.py --model resnet --model_depth 50 --resnet_shortcut B --batch_size 128 --num_workers 8 --n_epochs 50 --fold_index 4
