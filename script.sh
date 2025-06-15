# train
CUDA_VISIBLE_DEVICES=1 python geobfn_train.py --config_file configs/bfn4molgen_compete_condition.yaml --epochs 3000
CUDA_VISIBLE_DEVICES=1 nohup python geobfn_train.py --config_file configs/bfn4molgen_compete_condition.yaml --epochs 3000 &


CUDA_VISIBLE_DEVICES=5 nohup python geobfn_train.py --config_file configs/bfn4molgen_compete.yaml --epochs 3000 --resume &
CUDA_VISIBLE_DEVICES=5 timeout 72h bash -c 'while true; do nohup python geobfn_train.py --config_file configs/bfn4molgen_compete.yaml --epochs 3000 --resume; sleep 1; done' &

# sample
CUDA_VISIBLE_DEVICES=6 python geobfn_sampling.py --config_file logs/zengchuanlong_geobfn/compete/config.yaml

# evaluate
python evaluate.py --input output/output_20250329_142807.pk
