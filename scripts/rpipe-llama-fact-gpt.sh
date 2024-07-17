export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=$(pwd)/transformer_cache
python3 refactor-pipe.py \
        --task factscore \
        --generation_model_name meta-llama/Llama-2-7b-chat-hf \
        --breakdown_model_name gpt-3.5-turbo \
        --verification_model_name gpt-3.5-turbo-16k-0613 \
        --correction_model_name gpt-3.5-turbo-0613 \
        --retrieval_file_path factscore_google_retrieval_post_trim_llama.out \
        --api_key api_key \
        --generation_length 1024 \
	--verification_length 512 \
	--correction_length 1024 \
        --correction_with_question \
        --verification_with_question \
	--correct_false_statement \
	--correct_false_statement_with_question \
        --all_true_not_change \
	--generation_outputs_path outputs/factscore-google-Llama-2-7b-chat-hf.out
        #--only_run_correction \
	#--only_only_correction \
	#--generation_outputs_path outputs/factscore-google-gpt-3.5-turbo.out \
        #--breakdown_outputs_path outputs/factscore-google-breakdown-ggpt-3.5-turbo-gpt-3.5-turbo.out \
	#--verification_outputs_path outputs/factscore-verifywq-google-ggpt-3.5-turbo-gpt-3.5-turbo-16k-0613.out
#mv outputs/factscore-rag-only-correct-wq-gpt-3.5-turbo-1024-gpt-3.5-turbo-0613.out outputs/factscore-rag-only-correct-wq-gpt-3.5-turbo-1024-gpt-3.5-turbo-0613-iter1.out

	#--breakdown_outputs_path outputs/factscore-rag-google-breakdown-ggpt-3.5-turbo-gpt-3.5-turbo.out #\
        #--verification_outputs_path outputs/factscore-verify-google-ggpt-3.5-turbo-gpt-3.5-turbo-16k-0613.out
        #--verification_with_question \
	#--use_rag \
	#--correction_with_question \
	#--correct_false_statement \
        #--generation_outputs_path outputs/factscore-rag-google-gpt-3.5-turbo.out #\
        #--breakdown_outputs_path outputs/factscore-rag-breakdown-para-gpt-3.5-turbo.out \
        #--verification_outputs_path outputs/factscore-rag-verify-google-gpt-3.5-turbo-16k-0613.out \
        #--use_rag \
	#--correction_with_question \
        #--correct_false_statement #\
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



