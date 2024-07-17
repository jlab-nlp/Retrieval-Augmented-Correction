export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRANSFORMERS_CACHE=$(pwd)/transformer_cache
python3 pipeline2.py --task factscore
