rm -rf TReNDs

python train_TReNDs.py --model resnet --model_depth 34 --resnet_shortcut A --batch_size 64 --num_workers 8 --n_epochs 30 --fold_index 0
python train_TReNDs.py --model resnet --model_depth 34 --resnet_shortcut A --batch_size 64 --num_workers 8 --n_epochs 30 --fold_index 1
python train_TReNDs.py --model resnet --model_depth 34 --resnet_shortcut A --batch_size 64 --num_workers 8 --n_epochs 30 --fold_index 2
python train_TReNDs.py --model resnet --model_depth 34 --resnet_shortcut A --batch_size 64 --num_workers 8 --n_epochs 30 --fold_index 3
python train_TReNDs.py --model resnet --model_depth 34 --resnet_shortcut A --batch_size 64 --num_workers 8 --n_epochs 30 --fold_index 4
