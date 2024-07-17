
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from tqdm import tqdm
from task_utils import supported_tasks, generate_prompt_input_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, default="factscore", help="The task that will be used to run ")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--use_8bit", type=bool, action='store_true')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument('--max-tokens', type=int, default=1024, help="maximum # of tokens to be generated")
    parser.add_argument("--debug", type=bool, action='store_true')
    parser.add_argument("")
    args = parser.parse_args()
    if args.task not in supported_tasks:
        print(f"Task {args.task} is not supported!")
        exit(0)
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.use_8bit:
        model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, torch_dtype=torch.bfloat16,
                                                     device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    input_examples = generate_prompt_input_examples(args.task)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    model_name_no_head = args.model_name.split("/")[1]
    with open(os.path.join(args.output_dir, args.task + "-" + model_name_no_head + ".out"), "w") as f:
        for id, input_example in tqdm(enumerate(input_examples), total=len(input_examples),
                                      desc=f"running {args.model_type} on {args.task}..."):
            inputs = tokenizer(input_example, return_tensors="pt")
            out = model.generate(**inputs, max_new_tokens=args.max_tokens)
            model_output = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
            model_output = model_output[len(input_example):]
            if args.debug:
                print("input_example:", input_example)
                print("model_output:", model_output)
            f.write(model_output + "\n---\n")


if __name__ == '__main__':
    main()
