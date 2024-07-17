from factscore.factscorer import FactScorer
from task_utils import read_task_examples, get_outputs
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    args = parser.parse_args()
    # with open("api_key", "r") as f:
    #     api_key = f.read().strip("\n")
    fs = FactScorer(openai_key="api_key", cache_dir=f".cache/factscore/{args.model_name}")
    output_path = f"{args.output_dir}/{args.model_name}.out"
    topics = read_task_examples("factscore")
    generations = get_outputs(output_path=output_path)
    print(len(topics))
    assert len(topics) == len(generations)
    out = fs.get_score(topics, generations, gamma=10)
    print("FActScore:", out["score"])
    print("FActScore w/o length penalty:", out["init_score"])
    print("% of responding (not abstaining from answering):", out["respond_ratio"])
    print("average number of atomic facts per response:", out["num_facts_per_response"])
