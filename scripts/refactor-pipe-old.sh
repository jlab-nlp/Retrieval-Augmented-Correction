export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=$(pwd)/transformer_cache
python3 refactor-pipe.py \
        --task truthfulqa \
        --generation_model_name meta-llama/Meta-Llama-3-8B-Instruct \
        --breakdown_model_name meta-llama/Meta-Llama-3-8B-Instruct \
        --verification_model_name meta-llama/Meta-Llama-3-8B-Instruct \
        --correction_model_name meta-llama/Meta-Llama-3-8B-Instruct \
        --retrieval_file_path truthfulqa_google_retrieval_top_30_post_post_trim_llama.out \
        --api_key api_key \
        --generation_length 128 \
        --verification_length 512 \
        --correction_length 128 \
        --generation_outputs_path outputs/truthfulqa-rag-google-Meta-Llama-3-8B-Instruct.out \
        --breakdown_outputs_path outputs/truthfulqarag-google-breakdown-para-Meta-Llama-3-8B-Instruct.out \
        --verification_outputs_path outputs/truthfulqa-rag-verify-google-Meta-Llama-3-8B-Instruct.out \
        --use_rag \
	--correction_with_question \
        --correct_false_statement #\
	#--correction_with_question




#        --correction_with_question \
#        --filter_not_mentioned_statement \
#
#        --do_generation \
#        --do_breakdown \
#        --do_verification \
#        --do_correction \
#        --only_generation \
#        --only_breakdown \
#        --only_verification \
#        --only_correction \
#        --debug_only



