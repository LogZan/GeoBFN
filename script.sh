# train
CUDA_VISIBLE_DEVICES=5 nohup python geobfn_train.py --config_file configs/bfn4molgen_compete.yaml --epochs 3000 --resume &

# sample
CUDA_VISIBLE_DEVICES=7 python geobfn_sampling.py --config_file logs/zengchuanlong_geobfn/compete/config.yaml

# evaluate
python evaluate.py --input output/output20250327_1534.pkl
