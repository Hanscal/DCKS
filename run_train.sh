export CUDA_VISIBLE_DEVICES=0
python src/main.py \
        --do_train \
        --evaluate_during_training \
        --output_dir src/checkpoints/ed_ours  \
        --task woD_nb1_ep10 \
        --num_train_epochs 10
