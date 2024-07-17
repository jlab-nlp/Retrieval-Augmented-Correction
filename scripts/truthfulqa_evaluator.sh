export CUDNN_PATH="/usr/local/lib/python3.8/site-packages/nvidia/cudnn"
export LD_LIBRARY_PATH="$CUDNN_PATH/lib":"/usr/local/cuda/lib64"
export PATH="$PATH":"/usr/local/cuda/bin"
export CUDA_VISIBLE_DEVICES=0
python3 truthfulqa_evaluator.py --model_name $1
