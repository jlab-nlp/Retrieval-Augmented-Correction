export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=$(pwd)/transformer_cache
python3 refactor-pipe.py \
        --task factscore \
        --generation_model_name meta-llama/Meta-Llama-3-8B-Instruct \
        --breakdown_model_name meta-llama/Meta-Llama-3-8B-Instruct \
        --verification_model_name meta-llama/Meta-Llama-3-8B-Instruct \
        --correction_model_name meta-llama/Meta-Llama-3-8B-Instruct \
        --retrieval_file_path factscore_google_retrieval_post_trim_llama.out \
        --api_key api_key \
        --generation_length 512 \
        --verification_length 512 \
        --correction_length 1024 \
        --generation_outputs_path outputs/factscore-rag-google-Meta-Llama-3-8B-Instruct.out \
        --breakdown_outputs_path outputs/factscorerag-google-breakdown-para-Meta-Llama-3-8B-Instruct.out \
        --verification_outputs_path outputs/factscore-rag-verify-google-Meta-Llama-3-8B-Instruct.out \
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



