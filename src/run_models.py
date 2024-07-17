import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from tqdm import tqdm
import openai
import logging
import time
import numpy as np

def run_llm(task, input_examples, model_name, save_to_file=True, output_dir="outputs", max_tokens=1024, use_8bit=False,
            loaded_model=None, loaded_tokenizer=None):
    model_outputs = []
    if loaded_tokenizer:
        tokenizer = loaded_tokenizer
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token="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")
    #tokenizer.eos_token ='<|eotid|>'
    if loaded_model:
        model = loaded_model
    else:
        if use_8bit:
            model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, torch_dtype=torch.bfloat16,
                                                         device_map="auto",
                                                         token="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")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token="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")
    #model.generation_config = GenerationConfig(**{"bos_token_id": 128000, "eos_token_id": [128001, 128009], "max_new_tokens":max_tokens , "do_sample": False}) #,   "temperature": 0.6, #   "top_p": 0.9})
    
    #model.to("cuda")
    #terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    for eid, input_example in tqdm(enumerate(input_examples), total=len(input_examples),
                                  desc=f"running {model_name} on {task}..."):
        messages = [{"role": "user", "content": input_example}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
        out = model.generate(**inputs, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, max_new_tokens=max_tokens)#generation_config=model.generation_config)
        model_output = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        #model_output = model_output[len(input_example):]
        #print(model_output)
        #exit(0)
        if "assistant\n\n" in  model_output:
            model_output = model_output.split("assistant\n\n")[1]
        elif "[/INST]  " in model_output:
            model_output = model_output.split("[/INST]  ")[1]
        elif "Answer:\n" in model_output:
            model_output = model_output.split("Answer:\n")[1] 
        elif "model\n" in model_output:
            model_output = model_output.split("model\n")[1]
        else:
            model_output = model_output[len(input_example)+2:] 
        #print(model_output)
        #exit(0)
        model_outputs.append(model_output)
    assert len(input_examples) == len(model_outputs)
    if save_to_file:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        model_name_no_head = model_name.split("/")[1]
        with open(os.path.join(output_dir, task + "-" + model_name_no_head + ".out"), "w") as f:
            for model_out in model_outputs:
                f.write(model_out + "\n---\n")
    return model_outputs


def generate_chat_completions(input_texts, max_tokens, top_p=0.3, model_type="gpt-3.5-turbo"):
    received = False
    num_rate_errors = 0
    result = None
    while not received:
        try:
            result = openai.ChatCompletion.create(
                model=model_type,
                messages=[
                    {"role": "user", "content": input_texts},
                ],
                max_tokens=max_tokens,
                top_p=top_p,
                # stop=["\n\n", "\n#"]
            )
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError:
                # something is wrong: e.g. prompt too long
                logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{input_texts}\n\n")
                assert False
            logging.error("API error: %s (%d). Waiting %dsec" % (error, num_rate_errors, np.power(2, num_rate_errors)))
            time.sleep(np.power(2, num_rate_errors))
    if result:
        return result['choices'][0]["message"]["content"]
    else:
        return result

def run_openai(task, input_examples, model_name, save_to_file=True, output_dir="outputs", max_tokens=1024):
    model_outputs = []
    with open("api_key") as f:
        openai.api_key = f.read().strip("\n")
    for eid, input_example in tqdm(enumerate(input_examples), total=len(input_examples),
                                  desc=f"running {model_name} on {task}..."):
        model_output = generate_chat_completions(input_example, max_tokens=max_tokens,
                                                 model_type=model_name)
        model_outputs.append(model_output)
    assert len(input_examples) == len(model_outputs)
    if save_to_file:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        with open(os.path.join(output_dir, task + "-" + model_name + ".out"), "w") as f:
            for model_out in model_outputs:
                f.write(model_out + "\n---\n")
    return model_outputs


if __name__ == '__main__':
    from task_utils import generate_prompt_input_examples
    task = "truthfulqa"
    input_examples =generate_prompt_input_examples(task)
    run_openai(task, input_examples, "gpt-3.5-turbo")

