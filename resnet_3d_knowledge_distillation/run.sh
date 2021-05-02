rm -rf TReNDs

python train_TReNDs.py --model resnet --model_depth 34 --resnet_shortcut A --batch_size 64 --num_workers 8 --n_epochs 35 --fold_index 0 --feat_index 0
python train_TReNDs.py --model resnet --model_depth 34 --resnet_shortcut A --batch_size 64 --num_workers 8 --n_epochs 35 --fold_index 1 --feat_index 0
python train_TReNDs.py --model resnet --model_depth 34 --resnet_shortcut A --batch_size 64 --num_workers 8 --n_epochs 35 --fold_index 2 --feat_index 0
python train_TReNDs.py --model resnet --model_depth 34 --resnet_shortcut A --batch_size 64 --num_workers 8 --n_epochs 35 --fold_index 3 --feat_index 0
python train_TReNDs.py --model resnet --model_depth 34 --resnet_shortcut A --batch_size 64 --num_workers 8 --n_epochs 35 --fold_index 4 --feat_index 0
