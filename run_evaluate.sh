export CUDA_VISIBLE_DEVICES=0
python src/main.py  \
       --do_eval \
       --model_name_or_path checkpoint_path
