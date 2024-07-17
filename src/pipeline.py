from components import (generate_break_down_inputs,
                        generate_correction_inputs,
                        generate_verification_inputs,
                        generate_rag_inputs,
                        generate_statement_correction_single,
                        generate_rag_find_inputs)
from task_utils import generate_prompt_input_examples, get_outputs
from run_models import run_llm, run_openai
from retriever import retrieve, get_retrieved_outputs
from cost_estimator import estimate_by_inputs
import logging
import argparse
import openai
from tqdm import tqdm


class Pipeline(object):
    def __init__(self,
                 task,
                 generation_model_name,
                 breakdown_model_name,
                 filter_model_name,
                 verification_model_name,
                 correction_model_name,
                 use_openai_in_generation=True,
                 use_openai_in_breakdown=True,
                 use_openai_in_filter=True,
                 use_openai_in_verification=True,
                 use_openai_in_correction=True,
                 api_key="api_key",
                 breakdown_strategy="paragraph",
                 retriever_type="BM25",
                 retriever_top=5,
                 skip_generation=True,
                 generation_length=1024):
        self.task = task
        self.data = generate_prompt_input_examples(task)
        if breakdown_strategy != "paragraph" and breakdown_strategy != "sentence":
            raise RuntimeError("Unknown break down strategy.")
        self.breakdown_strategy = breakdown_strategy
        self.generation_model_name = generation_model_name
        self.breakdown_model_name = breakdown_model_name
        self.filter_model_name = filter_model_name
        self.verification_model_name = verification_model_name
        self.correction_model_name = correction_model_name
        self.use_openai_in_generation = use_openai_in_generation
        self.use_openai_in_breakdown = use_openai_in_breakdown
        self.use_openai_in_filter = use_openai_in_filter
        self.use_openai_in_correction = use_openai_in_correction
        self.use_openai_in_verification = use_openai_in_verification
        self.retriever_type = retriever_type
        self.retriever_top = retriever_top
        self.api_key = api_key
        self.skip_generation = skip_generation
        self.generation_length = generation_length

    def run(self):
        if (self.use_openai_in_generation or self.use_openai_in_breakdown
                or self.use_openai_in_verification or self.use_openai_in_correction):
            if not self.api_key:
                raise RuntimeError("Please define openai api key.")
            else:
                with open(self.api_key) as f:
                    self.api_key = f.read().strip("\n")
                    openai.api_key = self.api_key
        logging.info(f"running the {self.task} generation using {self.generation_model_name}...")
        if not self.skip_generation:
            if self.use_openai_in_generation:
                # print("Generation estimated cost (US dollars):",
                #       estimate_by_inputs(self.data, self.generation_length, self.generation_model_name))
                retrieved_contents = get_retrieved_outputs(f"truthfulqa_google_retrieval_top_30_post_post_trim.out")
                assert len(self.data) == len(retrieved_contents)
                rag_inputs = generate_rag_find_inputs(self.data, retrieved_contents)
                model_outputs = run_openai(self.task+"-rag-find-google", rag_inputs, self.generation_model_name)
            else:
                model_outputs = run_llm(self.task, self.data, self.generation_model_name)
        else:
            #path = f"outputs/{self.task}-rag-find-google-{self.generation_model_name}.out"
            path = "outputs/factscore-rag-correct-google-gpt-3.5-turbo-0613.out"
            logging.info(f"loading outputs from {path}")
            model_outputs = get_outputs(path)
        model_outputs = model_outputs
        logging.info(f"running the {self.task} model_outputs breakdown using {self.breakdown_model_name}...")
        break_down_inputs = generate_break_down_inputs(model_outputs, self.breakdown_strategy)
        facts_outputs = []
        if self.breakdown_strategy == "paragraph":
            if self.use_openai_in_breakdown:
                print("break down estimated cost (US dollars):",
                      estimate_by_inputs(break_down_inputs, self.generation_length, self.breakdown_model_name))
                facts_outputs = run_openai(self.task + "-rag-google-correct-1-breakdown-para", break_down_inputs, self.breakdown_model_name)
            else:
                facts_outputs = run_llm(self.task + "rag-google-correct-1-breakdown-para", break_down_inputs, self.breakdown_model_name)
        elif self.breakdown_strategy == "sentence":
            for break_down_input in break_down_inputs:
                facts_output = ""
                for breakdown_sentence in break_down_input:
                    if self.use_openai_in_breakdown:
                        facts_output_sent = run_openai(self.task + "-breakdown-sent", breakdown_sentence,
                                                       self.breakdown_model_name)
                    else:
                        facts_output_sent = run_llm(self.task + "-breakdown-sent", breakdown_sentence,
                                                    self.breakdown_model_name)
                    facts_output += facts_output_sent + "\n"
                facts_outputs.append(facts_output)
        else:
            raise RuntimeError("Unknown break down strategy.")
        #facts_outputs = get_outputs(f"outputs/factscore-rag-google-correct-1-breakdown-para-gpt-3.5-turbo.out")
        print(len(facts_outputs))
        # retrieved_contents = retrieve(self.task+"-retrieve", model_outputs, 10, model_name=self.generation_model_name)
        retrieved_contents = get_retrieved_outputs(f"factscore_google_retrieval_post_reduce.out")
        # print(len(retrieved_contents))
        # exit(0)
        # filered_retrieved_contents = []
        # for retrieved_content in tqdm(retrieved_contents, total=len(retrieved_contents)):
        #     contents = retrieved_content.split("\n---\n")
        #     count = 0
        #     new_content = ""
        #     for content in contents:
        #         input_text = (f"\"{content}\"\n\nIs the above content factual (Yes or No)?\n")
        #         if self.use_openai_in_filter:
        #             is_fact = run_openai(self.task+"-facts_filter", [input_text], self.filter_model_name, max_tokens=5, save_to_file=False)
        #         else:
        #             is_fact = run_llm(self.task + "-facts_filter", [input_text], self.filter_model_name, max_tokens=5, save_to_file=False)
        #         assert len(is_fact) == 1
        #         if "yes" in is_fact[0].lower():
        #             new_content += content + "\n---\n"
        #             count += 1
        #             if count >= 5:
        #                 break
        #     filered_retrieved_contents.append(new_content[:-2])
        # for retrieved_content in tqdm(retrieved_contents, total=len(retrieved_contents)):
        #     contents = retrieved_content.split("\n---\n")
        #     count = 0
        #     new_content = ""
        #     for content in contents:
        #         new_content += content + "\n---\n"
        #         count += 1
        #         if count >= 5:
        #             break
        #     filered_retrieved_contents.append(new_content[:-2])
        verification_inputs = generate_verification_inputs(facts_outputs, retrieved_contents=retrieved_contents)
        if self.use_openai_in_verification:
            verified_outputs = run_openai(self.task + "-rag-verify-google-correct-1", verification_inputs, self.verification_model_name, max_tokens=256)
        else:
            verified_outputs = run_llm(self.task + "-rag-verify-google-correct-1", verification_inputs, self.verification_model_name)
        exit(0)
        #verified_outputs = get_outputs(f"outputs/factscore-rag-verify-google-correct-statement-1-gpt-3.5-turbo-16k-0613.out")
        assert len(model_outputs) == len(verified_outputs) == len(retrieved_contents)
        correct_statement = False
        filtered_inputs = []
        eid = 0
        for facts_output, verified_output in zip(facts_outputs, verified_outputs):
            facts = facts_output.split("\n")
            verified_results = verified_output.split("\n")
            if len(facts) == len(verified_results):
                statements = ""
                for fact, verified_result in zip(facts, verified_results):
                    if correct_statement:
                        if "false" not in verified_result.lower():
                            statements += fact + "\n"
                        elif "false" in verified_result.lower():
                            statement_correct_input = generate_statement_correction_single(fact, retrieved_contents[eid])
                            corrected_statement = run_openai(self.task + f"-statement_correct_{eid}", statement_correct_input,
                                       self.generation_model_name, max_tokens=128, save_to_file=False)
                            statements += corrected_statement[0] + "\n"
                    else:
                        if "false" not in verified_result.lower():
                            statements += fact + "\n"

                filtered_inputs.append(statements)
            else:
                print(f"verified number example {eid} error, use original")
                statements = ""
                for fact in facts:
                    statements += fact + "\n"
                filtered_inputs.append(statements)
            eid += 1
        correction_inputs = generate_correction_inputs(model_outputs, facts_outputs)
        if self.use_openai_in_correction:
            corrected_outputs = run_openai(self.task + "-rag-correct-google-correct", correction_inputs, self.correction_model_name)
        else:
            corrected_outputs = run_llm(self.task + "-rag-correct-google-correct", correction_inputs, self.correction_model_name)
        assert len(model_outputs) == len(corrected_outputs)
        return corrected_outputs

# 2 problems
# 1. detected not mentioned content in retrival part
# 2. batch prompting not work.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, default="truthfulqa",
                        help="The task that will be used to run ")
    parser.add_argument("--generation_model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--filter_model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--breakdown_model_name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--verification_model_name", type=str, default="gpt-3.5-turbo-16k-0613")
    parser.add_argument("--correction_model_name", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--use_openai_in_generation", action="store_true")
    parser.add_argument("--use_openai_in_breakdown", action="store_true")
    parser.add_argument("--use_openai_in_verification", action="store_true")
    parser.add_argument("--use_openai_in_correction", action="store_true")
    parser.add_argument('--api_key', type=str,  default=None)
    parser.add_argument("--breakdown_strategy", type=str, default="paragraph")
    parser.add_argument("--retriever_type", type=str, default="bm25")
    parser.add_argument("--retriever_top", type=int, default=5)
    args = parser.parse_args()
    pipe = Pipeline(args.task,
                    args.generation_model_name,
                    args.breakdown_model_name,
                    args.filter_model_name,
                    args.verification_model_name,
                    args.correction_model_name,
                    use_openai_in_breakdown=True,
                    use_openai_in_filter=True,
                    use_openai_in_verification=True,
                    use_openai_in_correction=True,
                    api_key="api_key",
                    breakdown_strategy="paragraph",
                    retriever_type="BM25",
                    retriever_top=5,
                    skip_generation=True,
                    generation_length=512)
    pipe.run()
