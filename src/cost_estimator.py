from task_utils import generate_prompt_input_examples, get_outputs
from retriever import get_retrieved_outputs
from components import generate_rag_inputs, generate_rag_find_inputs
import tiktoken
from tqdm import tqdm
from components import generate_break_down_inputs, generate_correction_inputs, generate_verification_inputs

model_inputs_rates = {"gpt-4-turbo": 0.01, "gpt-4": 0.03, "gpt-4-32k": 0.06,
                      "gpt-3.5-turbo": 0.0005, "gpt-3.5-turbo-instruct": 0.0015}
model_outputs_rates = {"gpt-4-turbo": 0.03, "gpt-4": 0.06, "gpt-4-32k": 0.12,
                       "gpt-3.5-turbo": 0.0015, "gpt-3.5-turbo-instruct": 0.0020}


def estimate_generation_cost_by_task(task, model="gpt-3.5-turbo"):
    retrieved_content = get_retrieved_outputs("truthfulqa_google_retrieval_top_30_post_post_trim.out")
    tokenizer = tiktoken.encoding_for_model(model)
    input_examples = generate_prompt_input_examples(task)
    assert len(input_examples) == len(retrieved_content)
    input_examples = generate_rag_find_inputs(input_examples, retrieved_content, top=3)
    input_counter = 0
    output_counter = 0
    for input_example in tqdm(input_examples, total=len(input_examples)):
        input_tokens = tokenizer.encode(input_example)
        input_counter += len(input_tokens)
        output_counter += 128
    return input_counter * model_inputs_rates[model] / 1000 + output_counter * model_outputs_rates[model] / 1000


def estimate_by_inputs(input_examples, output_length, model):
    tokenizer = tiktoken.encoding_for_model(model)
    input_counter = 0
    output_counter = 0
    for input_example in input_examples:
        input_counter += len(tokenizer.encode(input_example))
        output_counter += output_length
    return input_counter * model_inputs_rates[model] / 1000 + output_counter * model_outputs_rates[model] / 1000


def estimated_all_pipe(task, model="gpt-3.5-turbo"):
    tokenizer = tiktoken.encoding_for_model(model)
    input_examples = generate_prompt_input_examples(task)
    input_counter = 0
    output_counter = 0
    for input_example in input_examples:
        input_counter += len(tokenizer.encode(input_example))
        output_counter += 1024
    generation_cost = input_counter * model_inputs_rates[model] / 1000 + output_counter * model_outputs_rates[model] / 1000
    print(f"{task} generation cost is ${generation_cost}.")

def estimated_break_down(model_outputs):
    pass

if __name__ == '__main__':
    # enc = tiktoken.encoding_for_model("gpt-4")
    # print(len(enc.encode("hello world")))
    # generations = get_outputs("outputs/factscore-rag-gpt-3.5-turbo.out")
    # print(len(generations))
    # exit(0)
    print(estimate_generation_cost_by_task("truthfulqa"))
