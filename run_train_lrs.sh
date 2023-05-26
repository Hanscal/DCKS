export CUDA_VISIBLE_DEVICES=0
python src/main.py \
        --do_train \
        --evaluate_during_training \
        --output_dir src/checkpoints/ed_ours  \
        --task woD_nb5_dr05_ep10 \
        --data_ratio 0.5 \
        --num_train_epochs 10
