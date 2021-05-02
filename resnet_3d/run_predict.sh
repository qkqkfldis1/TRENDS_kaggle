#rm -rf TReNDs

python test_TReNDs.py --model resnet --model_depth 34 --resnet_shortcut A --batch_size 128 --num_workers 4 --n_epochs 50 --fold_index 0 \
	--resume_path epoch_28_batch_75_loss_0.16003692150115967.pth.tar

python test_TReNDs.py --model resnet --model_depth 34 --resnet_shortcut A --batch_size 128 --num_workers 4 --n_epochs 50 --fold_index 1 \
	--resume_path epoch_22_batch_75_loss_0.15952999889850616.pth.tar

python test_TReNDs.py --model resnet --model_depth 34 --resnet_shortcut A --batch_size 128 --num_workers 4 --n_epochs 50 --fold_index 2 \
	--resume_path epoch_26_batch_75_loss_0.1618579626083374.pth.tar

python test_TReNDs.py --model resnet --model_depth 34 --resnet_shortcut A --batch_size 128 --num_workers 4 --n_epochs 50 --fold_index 3 \
	--resume_path epoch_27_batch_75_loss_0.16224882006645203.pth.tar

python test_TReNDs.py --model resnet --model_depth 34 --resnet_shortcut A --batch_size 128 --num_workers 4 --n_epochs 50 --fold_index 4 \
	--resume_path epoch_27_batch_75_loss_0.1576353907585144.pth.tar
    
    
#python train_TReNDs.py --model resnet --model_depth 50 --resnet_shortcut B --batch_size 128 --num_workers 8 --n_epochs 50 --fold_index 1
#python train_TReNDs.py --model resnet --model_depth 50 --resnet_shortcut B --batch_size 128 --num_workers 8 --n_epochs 50 --fold_index 2
#python train_TReNDs.py --model resnet --model_depth 50 --resnet_shortcut B --batch_size 128 --num_workers 8 --n_epochs 50 --fold_index 3
#python train_TReNDs.py --model resnet --model_depth 50 --resnet_shortcut B --batch_size 128 --num_workers 8 --n_epochs 50 --fold_index 4
