# Improve-Factuality-with-Retrieval-and-Correction

Implementation for paper: Improving Factuality with Retrieval and Correction

## Environment Set Up

```bash
./setenv.sh
```

If you have python pakage installation issues, just rerun ./setup.sh after ./setenv.sh to debug.

## Running the Inference

```bash
python3 refactor-pipe.py -h
usage: refactor-pipe.py [-h] --task TASK --generation_model_name
                        GENERATION_MODEL_NAME --breakdown_model_name
                        BREAKDOWN_MODEL_NAME --verification_model_name
                        VERIFICATION_MODEL_NAME --correction_model_name
                        CORRECTION_MODEL_NAME --retrieval_file_path
                        RETRIEVAL_FILE_PATH [--use_rag] [--do_generation]
                        [--do_breakdown] [--do_verification] [--do_correction]
                        [--only_generation] [--only_breakdown]
                        [--only_verification]
                        [--generation_outputs_path GENERATION_OUTPUTS_PATH]
                        [--breakdown_outputs_path BREAKDOWN_OUTPUTS_PATH]
                        [--verification_outputs_path VERIFICATION_OUTPUTS_PATH]
                        [--only_run_correction] [--only_only_correction]
                        [--correct_false_statement]
                        [--correct_false_statement_with_question]
                        [--filter_not_mentioned_statement]
                        [--verification_with_question]
                        [--correction_with_question] [--all_true_not_change]
                        [--debug_only] [--api_key API_KEY]
                        [--generation_length GENERATION_LENGTH]
                        [--verification_length VERIFICATION_LENGTH]
                        [--correction_length CORRECTION_LENGTH]
                        [--run_iterative] [--num_iterative NUM_ITERATIVE]

optional arguments:
  -h, --help            show this help message and exit
  --task TASK           The task that will be used to run
  --generation_model_name GENERATION_MODEL_NAME
  --breakdown_model_name BREAKDOWN_MODEL_NAME
  --verification_model_name VERIFICATION_MODEL_NAME
  --correction_model_name CORRECTION_MODEL_NAME
  --retrieval_file_path RETRIEVAL_FILE_PATH
  --use_rag             if use rag
  --do_generation       
  --do_breakdown
  --do_verification
  --do_correction
  --only_generation
  --only_breakdown
  --only_verification
  --generation_outputs_path GENERATION_OUTPUTS_PATH
  --breakdown_outputs_path BREAKDOWN_OUTPUTS_PATH
  --verification_outputs_path VERIFICATION_OUTPUTS_PATH
  --only_run_correction
  --only_only_correction
  --correct_false_statement
  --correct_false_statement_with_question
  --filter_not_mentioned_statement
  --verification_with_question
  --correction_with_question
  --all_true_not_change
  --debug_only
  --api_key API_KEY_PATH
  --generation_length GENERATION_LENGTH
  --verification_length VERIFICATION_LENGTH
  --correction_length CORRECTION_LENGTH
```

See many scripts examples usage of this.

## Evaluation

### Evaluate on truthfulqa

```shell
# Note that you need modify the python path into your python path in set_truthfulqa_eval.sh
./set_truthfulqa_eval.sh
python3 truthfulqa_evaluator.py --output_path "your model output path"
```

### Evaluate on factscore

```shell
./setup_fact.sh
python3 factscore_evaluator.py --output_path "your model output path" --model_name "your model name"
```

