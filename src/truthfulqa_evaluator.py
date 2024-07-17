from task_utils import get_outputs, read_task_examples
import argparse
from dependencies.TruthfulQA.truthfulqa import metrics, evaluate
import pandas as pd
from tqdm import tqdm
import openai
with open("api_key") as f:
    openai.api_key = f.read().strip("\n")

def evaluate_truthfulqa(model_key, questions):
    try:
       questions = metrics.run_BLEURT(model_key, questions, cache_dir="truthfulqa_bleurt_cache")
    except Exception as err:
       print(err)

    try:
       questions = metrics.run_bleu_and_rouge(model_key, questions)
    except Exception as err:
       print(err)
    # , judge_name, info_name
    #try:
    #    questions = metrics.run_end2end_GPT3(model_key, 'GPT-judge', "gpt-3.5-turbo-instruct", questions, info=False)
    #    questions = metrics.run_end2end_GPT3(model_key, 'GPT-info', "gpt-3.5-turbo-instruct", questions, info=True)
    #except Exception as err:
    #    print(err)
    results = evaluate.format_frame(questions)
    results = results.mean(axis=0)
    results = results.reset_index().rename(columns={'level_0': 'Model',
                                                    'level_1': 'Metric',
                                                    0: 'Value'})

    # filter to most informative metrics
    results = results[results['Metric'].isin(['MC1', 'MC2',
                                              'bleu acc',
                                              'rouge1 acc',
                                              'BLEURT acc',
                                              'GPT-judge acc',
                                              'GPT-info acc'])]
    results = pd.pivot_table(results, 'Value', 'Model', 'Metric')
    results.to_csv(f'truthfulqa_results_{model_key}.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    examples = read_task_examples("truthfulqa")
    generations = get_outputs(output_path=args.output_path)
    len_examples = len(examples)
    len_generations = len(generations)
    assert len_examples == len_generations, f"{len_examples}, {len_generations}"
    for example, generation in tqdm(zip(examples, generations), total=len(examples)):
        example.update({args.model_name:generation})
    df = pd.DataFrame(examples)
    evaluate_truthfulqa(args.model_name, df)
