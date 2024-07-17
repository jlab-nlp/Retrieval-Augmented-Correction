import os
from task_utils import get_outputs
from retriever import get_retrieved_outputs
from rouge_score import rouge_scorer
import argparse
from tqdm import tqdm
from sacrebleu.metrics import BLEU

def compute_rouge1(model_outs, gold_contents):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    total_score = 0
    for model_out, gold_content in tqdm(zip(model_outs, gold_contents), total=len(model_outs), desc="run_rougeL_on_factscore"):
        score = scorer.score(gold_content, model_out)
        total_score += score["rougeL"].fmeasure
    return total_score/len(model_outs)

def compute_bleu(model_outs, gold_contents):
    bleu = BLEU(effective_order=True)
    total_score = 0
    for model_out, gold_content in tqdm(zip(model_outs, gold_contents), total=len(model_outs),
                                        desc="run_BLEU_on_factscore"):
        score = bleu.sentence_score(model_out,[gold_content]).score
        total_score += score
    print(total_score)
    return total_score/len(model_outs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output", type=str, required=True)
    args = parser.parse_args()
    gold_texts = get_retrieved_outputs("factscore_gold_retrieved.txt", splitter="-----")
    model_outputs = get_outputs("outputs/"+args.model_output)
    rouge1_score = compute_rouge1(model_outputs, gold_texts)
    bleu_score = compute_bleu(model_outputs, gold_texts)
    if not os.path.exists("factscore_results"):
        os.mkdir("factscore_results")
    print(args.model_output)
    print("rouge1_score:", rouge1_score)
    print("bleu:", bleu_score)
    # with open(os.path.join("factscore_results", "factscore_results_"+str(args.model_output)+".txt")) as f:
    #     f.write("rouge1_score:" + str(rouge1_score)+"\n")
    #     f.write("bleu_score:" + str(bleu_score)+"\n")

