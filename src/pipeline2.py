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
from transformers import AutoTokenizer, AutoModelForCausalLM
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
                retrieved_contents = get_retrieved_outputs(f"factscore_google_retrieval_post_trim_llama.out")
                assert len(self.data) == len(retrieved_contents)
                rag_inputs = generate_rag_inputs(self.data, retrieved_contents)
                model_outputs = run_openai(self.task+"-rag-google", rag_inputs, self.generation_model_name)
            else:
                retrieved_contents = get_retrieved_outputs(f"factscore_google_retrieval_post_trim_llama.out") 
                assert len(self.data) == len(retrieved_contents)
                rag_inputs = generate_rag_inputs(self.data, retrieved_contents)
                model_outputs = run_llm(self.task+"-rag-google", rag_inputs, self.generation_model_name, max_tokens=512)
        else:
            #path = f"outputs/{self.task}-rag-find-google-{self.generation_model_name}.out"
            path = "outputs/truthfulqa-rag-google-Meta-Llama-3-8B-Instruct.out"
            logging.info(f"loading outputs from {path}")
            model_outputs = get_outputs(path)
        exit(0)
        model_outputs = model_outputs
        # logging.info(f"running the {self.task} model_outputs breakdown using {self.breakdown_model_name}...")
        break_down_inputs = generate_break_down_inputs(model_outputs, self.breakdown_strategy)
        facts_outputs = []
        #if self.breakdown_strategy == "paragraph":
            #self.use_openai_in_breakdown = True
        #    if self.use_openai_in_breakdown:
        #        print("break down estimated cost (US dollars):",
        #              estimate_by_inputs(break_down_inputs, self.generation_length, self.breakdown_model_name))
        #        facts_outputs = run_openai(self.task + "-rag-google-breakdown-para-llama3", break_down_inputs, self.breakdown_model_name)
        #    else:
        #        facts_outputs = run_llm(self.task + "rag-google-breakdown-para", break_down_inputs, self.breakdown_model_name)
        #elif self.breakdown_strategy == "sentence":
        #    for break_down_input in break_down_inputs:
        #        facts_output = ""
        #        for breakdown_sentence in break_down_input:
        #            if self.use_openai_in_breakdown:
        #                facts_output_sent = run_openai(self.task + "-breakdown-sent", breakdown_sentence,
        #                                               self.breakdown_model_name)
        #else:
        #                facts_output_sent = run_llm(self.task + "-breakdown-sent", breakdown_sentence,
        #                                            self.breakdown_model_name)
        #            facts_output += facts_output_sent + "\n"
        #        facts_outputs.append(facts_output)
        #else:
        #    raise RuntimeError("Unknown break down strategy.")
        
        facts_outputs = get_outputs(f"outputs/factscore-rag-google-breakdown-para-llama3-gpt-3.5-turbo.out")
        print(len(facts_outputs))
        exit(0)
        #retrieved_contents = retrieve(self.task+"-retrieve", model_outputs, 10, model_name=self.generation_model_name)
        retrieved_contents = get_retrieved_outputs(f"factscore_google_retrieval_post_trim_llama.out")
        print("ddd:", len(retrieved_contents))
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
        #new_facts_outputs = []
        #for fact_output in facts_outputs:
        #    facts = fact_output.split("Here are the independent facts without pronouns:\n")[1]
        #    facts = facts.replace("\n\n", "\n")
        #    new_facts_outputs.append(facts)
        #verification_inputs = generate_verification_inputs(facts_outputs, retrieved_contents=retrieved_contents)
        #for verification_input in verification_inputs:
        #    print(verification_input, flush=True)
        #    print("----", flush=True)
        #exit(0)
        # print([verification_inputs[0]])
        # exit(0)
        #self.use_openai_in_verification = True
        #if self.use_openai_in_verification:
        #   verified_outputs = run_openai(self.task + "-rag-verify-google-llama3", verification_inputs, self.verification_model_name, max_tokens=512)
        #else:
        #   verified_outputs = run_llm(self.task + "-rag-verify-google", verification_inputs, self.verification_model_name, max_tokens=512)
        #exit(0)
        verified_outputs = get_outputs(f"outputs/factscore-rag-verify-google-llama3-gpt-3.5-turbo-16k-0613.out")
        print(len(model_outputs))
        print(len(verified_outputs))
        print(len(retrieved_contents))
        assert len(model_outputs) == len(verified_outputs) == len(retrieved_contents)
        correct_statement = True
        filtered_inputs = []
        eid = 0
        count =0
        llama_tokenizer = None#AutoTokenizer.from_pretrained(args.generation_model_name,
        #                                                token="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")
        llama_model = None#AutoModelForCausalLM.from_pretrained(args.generation_model_name, device_map="auto",
        #                                                   token="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")
        #facts_outputs = new_facts_outputs
        for facts_output, verified_output in zip(facts_outputs, verified_outputs):
            facts = facts_output.strip("\n").split("\n")
            #verified_output = verified_output.split(":assistant\n\n")[1]#.split("\n\n")[1].strip("\n")
            #verified_output = verified_output.replace("\n\n", "\n").strip("\n")
            if not verified_output:
                verified_results = []
            elif verified_output[0] == "\n":
                verified_results = verified_output[1:].split("\n")
            elif verified_output[0] == "H":
                verified_results = verified_output.split("\n\n")[1].strip("\n").split("\n")
            else:
                verified_results = verified_output.strip("\n").split("\n")
            n_facts = len(facts) 
            if len(verified_results) >= n_facts:
                verified_results = verified_results[:n_facts]
            else:
                verified_results = []
            # if len(facts) != len(verified_results):
            #     print("---------")
            #     print(eid)
            #     print(facts)
            #     print(verified_results)
            #     print("---------")
            #     eid += 1
            #     count+=1
            #     continue
            # else:
            #     eid+=1
            #     continue
            if len(facts) == len(verified_results):
                statements = ""
                for fact, verified_result in zip(facts, verified_results):
                    if correct_statement:
                        if "false" not in verified_result.lower():
                            statements += fact + "\n"
                        elif "false" in verified_result.lower():
                            statement_correct_input = generate_statement_correction_single(fact, retrieved_contents[eid])
                            corrected_statement = run_openai(self.task + f"-statement_correct_{eid}", statement_correct_input,
                                       self.breakdown_model_name, max_tokens=128, save_to_file=False)#,
                                                          #loaded_model=llama_model, loaded_tokenizer=llama_tokenizer)
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
        # print(count)
        # exit(0)
        with_question = True
        new_model_outputs = []
        if with_question:
            for question, answer in zip(self.data, model_outputs):
                new_model_outputs.append(question + "\nAnswer:\"" + answer +"\"")
            correction_inputs = generate_correction_inputs(new_model_outputs, facts_outputs)
        else:
            correction_inputs = generate_correction_inputs(model_outputs, facts_outputs)
        #for correction_input in correction_inputs:
        #    print(correction_input, flush=True)
        #    print("----", flush=True)
        #exit(0)
        self.use_openai_in_correction = True
        if self.use_openai_in_correction:
            corrected_outputs = run_openai(self.task + "-rag-correct-statement-google-llama3-512-512", correction_inputs, self.correction_model_name, max_tokens=512)
        else:
            corrected_outputs = run_llm(self.task + "-rag-correct-wq-google-512-512", correction_inputs, self.correction_model_name, max_tokens=512, loaded_model=llama_model, loaded_tokenizer=llama_tokenizer)
        assert len(model_outputs) == len(corrected_outputs)
        return corrected_outputs

# 2 problems
# 1. detected not mentioned content in retrival part
# 2. batch prompting not work.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, default="truthfulqa",
                        help="The task that will be used to run ")
    parser.add_argument("--generation_model_name", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    parser.add_argument("--filter_model_name", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    parser.add_argument("--breakdown_model_name", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    parser.add_argument("--verification_model_name", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    parser.add_argument("--correction_model_name", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
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
                    use_openai_in_generation=args.use_openai_in_generation,
                    use_openai_in_breakdown=args.use_openai_in_generation,
                    use_openai_in_filter=args.use_openai_in_generation,
                    use_openai_in_verification=args.use_openai_in_generation,
                    use_openai_in_correction=args.use_openai_in_generation,
                    api_key="api_key",
                    breakdown_strategy="paragraph",
                    retriever_type="BM25",
                    retriever_top=5,
                    skip_generation=False,
                    generation_length=512)
    pipe.run()
