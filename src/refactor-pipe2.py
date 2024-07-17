import os
from components1 import (generate_break_down_inputs,
                        generate_correction_inputs,
                        generate_verification_inputs,
                        generate_rag_inputs,
                        generate_statement_correction_single,
                        generate_rag_find_inputs,
                        generate_statement_correction_all)
from task_utils import generate_prompt_input_examples, get_outputs
from run_models1 import run_llm, run_openai
from retriever import get_retrieved_outputs
from cost_estimator import estimate_by_inputs
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import argparse
import openai
from tqdm import tqdm
from collections import defaultdict

gpt3_5_models = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613",
                 "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-instruct",
                 "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613"]
gpt4_models = ["gpt-4-turbo", "gpt-4-turbo-2024-04-09", "gpt-4-turbo-preview",
               "gpt-4-0125-preview", "gpt-4-1106-preview", "gpt-4", "gpt-4-0613",
               "gpt-4-32k", "gpt-4-32k-0613"]
openai_models = gpt3_5_models + gpt4_models
openai_models_dicts = defaultdict(bool)
openai_models_dicts.setdefault(False)
for model_name in openai_models:
    openai_models_dicts[model_name] = True


class Pipeline(object):
    def __init__(self,
                 task,
                 generation_model_name,
                 breakdown_model_name,
                 verification_model_name,
                 correction_model_name,
                 retrieved_file_path,
                 use_rag=True,
                 do_generation=False,
                 do_breakdown=False,
                 do_verification=False,
                 do_correction=False,
                 only_generation=False,
                 only_breakdown=False,
                 only_verification=False,
                 generation_outputs_path=None,
                 breakdown_outputs_path=None,
                 verification_outputs_path=None,
                 correct_false_statement=False,
                 filter_not_mentioned_statement=False,
                 correct_false_statement_with_question=False,
                 verification_with_question=False,
                 correction_with_question=False,
                 all_true_not_change=False,
                 debug_only=False,
                 api_key="api_key",
                 generation_length=1024,
                 verification_length=512,
                 correction_length=512):
        self.retrieved_file_path = retrieved_file_path
        self.use_rag = use_rag
        self.generation_outputs_path = generation_outputs_path
        self.breakdown_outputs_path = breakdown_outputs_path
        self.verification_outputs_path = verification_outputs_path
        self.task = task
        self.do_generation = do_generation
        self.do_breakdown = do_breakdown
        self.do_verification = do_verification
        self.do_correction = do_correction
        self.data = generate_prompt_input_examples(task)
        self.generation_model_name = generation_model_name
        self.breakdown_model_name = breakdown_model_name
        self.verification_model_name = verification_model_name
        self.correction_model_name = correction_model_name
        self.debug_only = debug_only
        # self.retriever_type = retriever_type
        # self.retriever_top = retriever_top
        self.api_key = api_key
        # self.skip_generation = skip_generation
        self.generation_length = generation_length
        self.verification_length = verification_length
        self.only_generation = only_generation
        self.only_breakdown = only_breakdown
        self.only_verification = only_verification
        self.correct_false_statement = correct_false_statement
        self.filter_not_mentioned_statement = filter_not_mentioned_statement
        self.correct_false_statement_with_question = correct_false_statement_with_question
        self.correction_with_question = correction_with_question
        self.verification_with_question = verification_with_question
        self.all_true_not_change = all_true_not_change
        self.correction_length = correction_length


    def run(self):
        if (openai_models_dicts[self.generation_model_name] or openai_models_dicts[self.breakdown_model_name] or
                openai_models_dicts[self.verification_model_name] or openai_models_dicts[self.correction_model_name]):
            if not self.api_key:
                raise RuntimeError("Please define openai api key.")
            else:
                with open(self.api_key) as f:
                    self.api_key = f.read().strip("\n")
                    openai.api_key = self.api_key
        logging.info(f"running the {self.task} generation using {self.generation_model_name}...")
        if self.only_generation:
            assert self.do_generation or self.generation_outputs_path is None
        if self.do_generation or self.generation_outputs_path is None:
            if openai_models_dicts[self.generation_model_name]:
                if self.use_rag:
                    retrieved_contents = get_retrieved_outputs(self.retrieved_file_path)
                    assert len(self.data) == len(retrieved_contents)
                    if self.task == "truthfulqa":
                        print("task is truthfulqa...run rag find...")
                        rag_inputs = generate_rag_find_inputs(self.data, retrieved_contents)
                        #print(rag_inputs[-3])
                        #exit(0)
                    else:
                        rag_inputs = generate_rag_inputs(self.data, retrieved_contents)
                    model_outputs = run_openai(self.task + "-rag-gold-google", rag_inputs, self.generation_model_name,
                                               max_tokens=self.generation_length)
                else:
                    model_outputs = run_openai(self.task + "-gold-google", self.data, self.generation_model_name,
                                               max_tokens=self.generation_length)
            else:
                if self.use_rag:
                    retrieved_contents = get_retrieved_outputs(self.retrieved_file_path)
                    assert len(self.data) == len(retrieved_contents)
                    if self.task == "truthfulqa":
                        print("task is truthfulqa...run rag find...")
                        rag_inputs = generate_rag_find_inputs(self.data, retrieved_contents)
                        #print([rag_inputs[-3]])
                        #exit(0)
                    else:
                        rag_inputs = generate_rag_inputs(self.data, retrieved_contents)
                    model_outputs = run_llm(self.task + "-rag-gold-google", rag_inputs, self.generation_model_name,
                                            max_tokens=self.generation_length)
                else:
                    model_outputs = run_llm(self.task + "-gold-google", self.data, self.generation_model_name,
                                            max_tokens=self.generation_length)
        else:
            logging.info(f"loading outputs from {self.generation_outputs_path}")
            model_outputs = get_outputs(self.generation_outputs_path)
        print(len(model_outputs))
            #exit(0)
        if self.only_generation:
            exit(0)
        model_outputs = model_outputs
        if self.use_rag:
            rag_name = "-rag-gold"
        else:
            rag_name = "-gold"
        gen_name = self.generation_model_name.replace("/", "-")
        # logging.info(f"running the {self.task} model_outputs breakdown using {self.breakdown_model_name}...")
        if self.only_breakdown:
            assert self.do_breakdown or self.breakdown_outputs_path is None
        if self.do_breakdown or self.breakdown_outputs_path is None:
            break_down_inputs = generate_break_down_inputs(model_outputs)
            # facts_outputs = []
            # if self.breakdown_strategy == "paragraph":
            #if self.use_rag:
            #    rag_name = "-rag"
            #else:
            #    rag_name = ""
            if openai_models_dicts[self.breakdown_model_name]:
                print("break down estimated cost (US dollars):",
                      estimate_by_inputs(break_down_inputs, self.generation_length, self.breakdown_model_name))
                #if self.use_rag:
                #    rag_name = "-rag"
                #else:
                #    rag_name = ""
                facts_outputs = run_openai(self.task + f"{rag_name}-google-breakdown-g{gen_name}", break_down_inputs,
                                           self.breakdown_model_name)
            else:
                facts_outputs = run_llm(self.task + f"{rag_name}-google-breakdown-g{gen_name}", break_down_inputs,
                                        self.breakdown_model_name)
            # elif self.breakdown_strategy == "sentence":
            #     for break_down_input in break_down_inputs:
            #         facts_output = ""
            #         for breakdown_sentence in break_down_input:
            #             if self.breakdown_model_name in openai_models:
            #                 facts_output_sent = run_openai(self.task + "-breakdown-sent", breakdown_sentence,
            #                                                self.breakdown_model_name)
            #             else:
            #                 facts_output_sent = run_llm(self.task + "-breakdown-sent", breakdown_sentence,
            #                                             self.breakdown_model_name)
            #             facts_output += facts_output_sent + "\n"
            #         facts_outputs.append(facts_output)
            # else:
            #     raise RuntimeError("Unknown break down strategy.")
        else:
            facts_outputs = get_outputs(self.breakdown_outputs_path)
        print(len(facts_outputs))
        if self.only_breakdown:
            exit(0)
        # exit(0)
        # retrieved_contents = retrieve(self.task+"-retrieve", model_outputs, 10, model_name=self.generation_model_name)
        retrieved_contents = get_retrieved_outputs(self.retrieved_file_path)
        print("ddd:", len(retrieved_contents))
        # exit(0)
        # filered_retrieved_contents = []
        # for retrieved_content in tqdm(retrieved_contents, total=len(retrieved_contents)):
        #     contents = retrieved_content.splita("\n---\n")
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
        if not openai_models_dicts[self.breakdown_model_name]:
            new_facts_outputs = []
            for fact_output in facts_outputs:
                if "assistant\n\n" in fact_output:
                    fact_output = fact_output.split("assistant\n\n")[1]
                # facts = fact_output.split("\n\n")[1].strip("\n")
                if fact_output[0].lower().startswith("here "):
                    xxx = fact_output.split(":\n\n")
                    if len(xxx) >= 2:
                        fact_output = xxx[1]
                    else:
                        raise RuntimeError(f"{fact_output}")
                facts = fact_output.replace("\n\n", "\n").strip("\n")
                #print([facts])
                new_facts_outputs.append(facts)
            facts_outputs = new_facts_outputs
        if self.only_verification:
            assert self.do_verification or self.verification_outputs_path is None
        if self.do_verification or self.verification_outputs_path is None:
            if self.verification_with_question:
                verification_inputs = generate_verification_inputs(facts_outputs, retrieved_contents=retrieved_contents,
                                                                   questions=self.data)
                verify_name = "wq"
            else:
                verification_inputs = generate_verification_inputs(facts_outputs, retrieved_contents=retrieved_contents)
                verify_name = ""

            if self.debug_only:
                for verification_input in verification_inputs:
                    print(verification_input, flush=True)
                    print("----", flush=True)
                    exit(0)
            if openai_models_dicts[self.verification_model_name]:
                verified_outputs = run_openai(self.task + f"{rag_name}-verify{verify_name}-google-g{gen_name}", verification_inputs,
                                              self.verification_model_name, max_tokens=self.verification_length)
            else:
                verified_outputs = run_llm(self.task + f"{rag_name}-verify{verify_name}-google-g{gen_name}", verification_inputs,
                                           self.verification_model_name, max_tokens=self.verification_length)
        else:
            verified_outputs = get_outputs(self.verification_outputs_path)
        if self.only_verification:
            exit(0)

        #
        # print([verification_inputs[0]])
        # exit(0)
        # self.use_openai_in_verification = True
        # if self.use_openai_in_verification:
        #   verified_outputs = run_openai(self.task + "-rag-verify-google-llama3", verification_inputs, self.verification_model_name, max_tokens=512)
        # else:
        #   verified_outputs = run_llm(self.task + "-rag-verify-google", verification_inputs, self.verification_model_name, max_tokens=512)
        # exit(0)
        # verified_outputs = get_outputs(f"outputs/truthfulqa-rag-verify-google-llama3-gpt-3.5-turbo-16k-0613.out")
        print(len(model_outputs))
        print(len(verified_outputs))
        print(len(retrieved_contents))
        assert len(model_outputs) == len(verified_outputs) == len(retrieved_contents)
        correct_statement = True
        filtered_inputs = []
        eid = 0
        llama_tokenizer = None
        llama_model = None
        if not openai_models_dicts[self.correction_model_name]:
            llama_tokenizer = AutoTokenizer.from_pretrained(args.generation_model_name,
                                                            token="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")
            llama_model = AutoModelForCausalLM.from_pretrained(args.generation_model_name, device_map="auto",
                                                               token="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")
        do_not_change_list = []
        for facts_output, verified_output in zip(facts_outputs, verified_outputs):
            #print(facts_output)
            #exit(0)
            facts = facts_output.strip("\n").split("\n")
            #print(verified_output)
            #exit(0)
            if not verified_output:
                verified_results = []
            elif ":assistant\n\n" in verified_output:
                verified_output_t = verified_output.split(":assistant\n\n")[1]
                #print("ddd:",verified_output_t)
                #exit(0)
                if verified_output_t[0] == "H":
                    verified_results = verified_output_t.split("\n\n")[1].strip("\n").split("\n")
                else:
                    verified_results = verified_output_t.strip("\n").split("\n")
            elif verified_output[0] == "\n":
                verified_results = verified_output[1:].split("\n")
            elif verified_output[0] == "H":
                verified_results = verified_output.split("\n\n")[1].strip("\n").split("\n")
            else:
                verified_results = verified_output.strip("\n").split("\n")
            #print(facts)
            #print(verified_results)
            #exit(0)
            do_not_change = True
            n_facts = len(facts)
            if len(verified_results) >= n_facts:
                verified_results = verified_results[:n_facts]
            else:
                verified_results = []
            if len(facts) == len(verified_results):
                statements = ""
                for fact, verified_result in zip(facts, verified_results):
                    if "not mentioned" in verified_result.lower():
                        if not self.filter_not_mentioned_statement:
                            statements += fact + "\n"
                    elif "false" in verified_result.lower():
                        do_not_change = False
                        if self.correct_false_statement:
                            if self.correct_false_statement_with_question:
                                statement_correct_input = generate_statement_correction_single(fact,
                                                                                               retrieved_contents[eid],
                                                                                               self.data[eid])
                            else:
                                statement_correct_input = generate_statement_correction_single(fact,
                                                                                               retrieved_contents[eid])
                            if openai_models_dicts[self.correction_model_name]:
                                corrected_statement = run_openai(self.task + f"-statement_correct_{eid}",
                                                                 statement_correct_input, self.correction_model_name,
                                                                 max_tokens=128,
                                                                 save_to_file=False)  # , loaded_model=llama_model, loaded_tokenizer=llama_tokenizer)=
                            else:
                                corrected_statement = run_llm(self.task + f"-statement_correct_{eid}",
                                                              statement_correct_input, self.correction_model_name,
                                                              max_tokens=128,
                                                              save_to_file=False,
                                                              loaded_model=llama_model,
                                                              loaded_tokenizer=llama_tokenizer)
                            statements += corrected_statement[0] + "\n"
                    else:
                        statements += fact + "\n"
                filtered_inputs.append(statements)
            else:
                print(f"verified number example {eid} error, use original")
                statements = ""
                for fact in facts:
                    statements += fact + "\n"
                filtered_inputs.append(statements)
            eid += 1
            do_not_change_list.append(do_not_change)
        # print(count)
        # exit(0)
        #print("ddd:", len(filtered_inputs))
        #print("ddd0:", len(self.data))
        #print("ddd00:", len(model_outputs))
        assert len(model_outputs) == len(do_not_change_list)
        new_model_outputs = []
        if self.correction_with_question:
            for question, answer in zip(self.data, model_outputs):
                new_model_outputs.append(question + "\nAnswer:\"" + answer + "\"")
            #print("ddd2:",len(new_model_outputs))
            #exit(0)
            correction_inputs = generate_correction_inputs(new_model_outputs, filtered_inputs)
        else:
            #print("ddd2:", len(filtered_inputs))
            #print(filtered_inputs[0])
            correction_inputs = generate_correction_inputs(model_outputs, filtered_inputs)
        #print(len(model_outputs))
        #print(len(filtered_inputs))
        #print(len(correction_inputs))
        #print(correction_inputs[0])
        #exit(0)
        if self.debug_only:
            for correction_input in correction_inputs:
                print(correction_input, flush=True)
                print("----", flush=True)
            exit(0)
        correct_file_name = self.task
        if self.use_rag:
            correct_file_name += "-rag-gold"
        else:
            correct_file_name += "-gold"
        if self.correct_false_statement:
            correct_file_name += "-correct-statement"
        else:
            correct_file_name += "-correct"
        if self.filter_not_mentioned_statement:
            correct_file_name += "-fn"
        if self.correct_false_statement_with_question:
            correct_file_name += "-cwq"
        if self.verification_with_question:
            correct_file_name += "-vwq"
        if self.correction_with_question:
            correct_file_name += "-wq"
        correct_file_name += "-" + gen_name
        correct_file_name += f"-{self.verification_length}" + f"-{self.correction_length}"
        if openai_models_dicts[self.correction_model_name]:
            corrected_outputs = run_openai(correct_file_name,
                                           correction_inputs, self.correction_model_name,
                                           max_tokens=self.correction_length)
        else:
            corrected_outputs = run_llm(correct_file_name, correction_inputs,
                                        self.correction_model_name, max_tokens=self.correction_length,
                                        loaded_model=llama_model, loaded_tokenizer=llama_tokenizer)
        assert len(model_outputs) == len(corrected_outputs)
        if self.all_true_not_change:
            print("all true do not change mode!")
            new_corrected_outputs = []
            for model_output, corrected_output, do_not_change in zip(model_outputs, corrected_outputs, do_not_change_list):
                if do_not_change:
                    new_corrected_outputs.append(model_output)
                else:
                    new_corrected_outputs.append(corrected_output)
            model_name_no_head = self.correction_model_name.replace("/", "-")
            out_file_path = os.path.join("outputs", correct_file_name + "-" + model_name_no_head + "-atnc" + ".out")
            with open(out_file_path, "w") as f:
                for model_out in new_corrected_outputs:
                    f.write(model_out + "\n---\n")
            return new_corrected_outputs
        return corrected_outputs


    def only_run_correction(self, only_only_correction):
        only_correct_name = "only_correction"
        if (openai_models_dicts[self.generation_model_name] or openai_models_dicts[self.breakdown_model_name] or
                openai_models_dicts[self.verification_model_name] or openai_models_dicts[self.correction_model_name]):
            if not self.api_key:
                raise RuntimeError("Please define openai api key.")
            else:
                with open(self.api_key) as f:
                    self.api_key = f.read().strip("\n")
                    openai.api_key = self.api_key
        logging.info(f"running the {self.task} generation using {self.generation_model_name}...")
        if self.only_generation:
            assert self.do_generation or self.generation_outputs_path is None
        if self.do_generation or self.generation_outputs_path is None:
            if openai_models_dicts[self.generation_model_name]:
                if self.use_rag:
                    retrieved_contents = get_retrieved_outputs(self.retrieved_file_path)
                    assert len(self.data) == len(retrieved_contents)
                    if self.task == "truthfulqa":
                        print("task is truthfulqa...run rag find...")
                        rag_inputs = generate_rag_find_inputs(self.data, retrieved_contents)
                    else:
                        rag_inputs = generate_rag_inputs(self.data, retrieved_contents)
                    model_outputs = run_openai(self.task + "-rag-gold-google", rag_inputs, self.generation_model_name,
                                               max_tokens=self.generation_length)
                else:
                    model_outputs = run_openai(self.task + "-gold-google", self.data, self.generation_model_name,
                                               max_tokens=self.generation_length)
            else:
                if self.use_rag:
                    retrieved_contents = get_retrieved_outputs(self.retrieved_file_path)
                    assert len(self.data) == len(retrieved_contents)
                    rag_inputs = generate_rag_inputs(self.data, retrieved_contents)
                    model_outputs = run_llm(self.task + "-rag-gold-google", rag_inputs, self.generation_model_name,
                                            max_tokens=self.generation_length)
                else:
                    model_outputs = run_llm(self.task + "-gold-google", self.data, self.generation_model_name,
                                            max_tokens=self.generation_length)
        else:
            logging.info(f"loading outputs from {self.generation_outputs_path}")
            model_outputs = get_outputs(self.generation_outputs_path)
        if self.only_generation:
            exit(0)
        model_outputs = model_outputs
        if self.use_rag:
            rag_name = "-rag-gold"
        else:
            rag_name = "-gold"
        # logging.info(f"running the {self.task} model_outputs breakdown using {self.breakdown_model_name}...")
        if self.only_breakdown:
            assert self.do_breakdown or self.breakdown_outputs_path is None
        if self.do_breakdown or self.breakdown_outputs_path is None:
            break_down_inputs = generate_break_down_inputs(model_outputs)
            # facts_outputs = []
            # if self.breakdown_strategy == "paragraph":
            #if self.use_rag:
            #    rag_name = "-rag"
            #else:
            #    rag_name = ""
            if openai_models_dicts[self.breakdown_model_name]:
                print("break down estimated cost (US dollars):",
                      estimate_by_inputs(break_down_inputs, self.generation_length, self.breakdown_model_name))
                #if self.use_rag:
                #    rag_name = "-rag"
                #else:
                #    rag_name = ""
                facts_outputs = run_openai(self.task + f"{rag_name}-{only_correct_name}-google-breakdown-g{self.generation_model_name}", break_down_inputs,
                                           self.breakdown_model_name)
            else:
                facts_outputs = run_llm(self.task + f"{rag_name}-{only_correct_name}-google-breakdown-g{self.generation_model_name}", break_down_inputs,
                                        self.breakdown_model_name)
            # elif self.breakdown_strategy == "sentence":
            #     for break_down_input in break_down_inputs:
            #         facts_output = ""
            #         for breakdown_sentence in break_down_input:
            #             if self.breakdown_model_name in openai_models:
            #                 facts_output_sent = run_openai(self.task + "-breakdown-sent", breakdown_sentence,
            #                                                self.breakdown_model_name)
            #             else:
            #                 facts_output_sent = run_llm(self.task + "-breakdown-sent", breakdown_sentence,
            #                                             self.breakdown_model_name)
            #             facts_output += facts_output_sent + "\n"
            #         facts_outputs.append(facts_output)
            # else:
            #     raise RuntimeError("Unknown break down strategy.")
        else:
            facts_outputs = get_outputs(self.breakdown_outputs_path)
        print(len(facts_outputs))
        if self.only_breakdown:
            exit(0)
        # exit(0)
        # retrieved_contents = retrieve(self.task+"-retrieve", model_outputs, 10, model_name=self.generation_model_name)
        retrieved_contents = get_retrieved_outputs(self.retrieved_file_path)
        print("ddd:", len(retrieved_contents))
        # exit(0)
        # filered_retrieved_contents = []
        # for retrieved_content in tqdm(retrieved_contents, total=len(retrieved_contents)):
        #     contents = retrieved_content.splita("\n---\n")
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
        if not openai_models_dicts[self.breakdown_model_name]:
            new_facts_outputs = []
            for fact_output in facts_outputs:
                if "assistant\n\n" in fact_output:
                    fact_output = fact_output.split("assistant\n\n")[1]
                # facts = fact_output.split("\n\n")[1].strip("\n")
                if fact_output[0].lower().startswith("here "):
                    xxx = fact_output.split(":\n\n")
                    if len(xxx) >= 2:
                        fact_output = xxx[1]
                    else:
                        raise RuntimeError(f"{fact_output}")
                facts = fact_output.replace("\n\n", "\n").strip("\n")
                #print([facts])
                new_facts_outputs.append(facts)
        #
        # print([verification_inputs[0]])
        # exit(0)
        # self.use_openai_in_verification = True
        # if self.use_openai_in_verification:
        #   verified_outputs = run_openai(self.task + "-rag-verify-google-llama3", verification_inputs, self.verification_model_name, max_tokens=512)
        # else:
        #   verified_outputs = run_llm(self.task + "-rag-verify-google", verification_inputs, self.verification_model_name, max_tokens=512)
        # exit(0)
        # verified_outputs = get_outputs(f"outputs/truthfulqa-rag-verify-google-llama3-gpt-3.5-turbo-16k-0613.out")
        print(len(model_outputs))
        print(len(retrieved_contents))
        assert len(model_outputs) == len(retrieved_contents)
        correct_statement = True

        do_not_change_list = []
        # eid = 0
        llama_tokenizer = None
        llama_model = None
        if not openai_models_dicts[self.correction_model_name]:
            llama_tokenizer = AutoTokenizer.from_pretrained(args.generation_model_name,
                                                            token="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")
            llama_model = AutoModelForCausalLM.from_pretrained(args.generation_model_name, device_map="auto",
                                                               token="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")
        if not only_only_correction:
            filtered_inputs = []
            for eid, facts_output in enumerate(facts_outputs):
                facts = facts_output.strip("\n").split("\n")
                statements = ""
                for fact in facts:
                    statement_correct_input = generate_statement_correction_all(fact,
                                                                                retrieved_contents[eid],
                                                                                self.data[eid])
                    if openai_models_dicts[self.correction_model_name]:
                        corrected_statement = run_openai(self.task + f"-statement_correct_{eid}",
                                                         statement_correct_input,
                                                         self.correction_model_name,
                                                         max_tokens=128,
                                                         save_to_file=False)  # , loaded_model=llama_model, loaded_tokenizer=llama_tokenizer)=
                    else:
                        corrected_statement = run_llm(self.task + f"-statement_correct_{eid}",
                                                      statement_correct_input, self.correction_model_name,
                                                      max_tokens=128,
                                                      save_to_file=False,
                                                      loaded_model=llama_model,
                                                      loaded_tokenizer=llama_tokenizer)
                    statements += corrected_statement[0] + "\n"
                filtered_inputs.append(statements)

        # print(count)
        # exit(0)
        #print("ddd:", len(filtered_inputs))
        #print("ddd0:", len(self.data))
        #print("ddd00:", len(model_outputs))
        new_model_outputs = []
        if self.correction_with_question:
            for question, answer in zip(self.data, model_outputs):
                new_model_outputs.append(question + "\nAnswer:\"" + answer + "\"")
            #print("ddd2:",len(new_model_outputs))
            #exit(0)
            if not only_only_correction:
                correction_inputs = generate_correction_inputs(new_model_outputs, filtered_inputs)
            else:
                correction_inputs = generate_correction_inputs(new_model_outputs, facts_outputs)
        else:
            #print("ddd2:", len(filtered_inputs))
            #print(filtered_inputs[0])
            if not only_only_correction:
                correction_inputs = generate_correction_inputs(model_outputs, filtered_inputs)
            else:
                correction_inputs = generate_correction_inputs(model_outputs, facts_outputs)
        #print(len(model_outputs))
        #print(len(filtered_inputs))
        #print(len(correction_inputs))
        #print(correction_inputs[0])
        #exit(0)
        if self.debug_only:
            for correction_input in correction_inputs:
                print(correction_input, flush=True)
                print("----", flush=True)
            exit(0)
        correct_file_name = self.task
        if self.use_rag:
            correct_file_name += "-rag-gold"
        else:
            correct_file_name += "-gold"
        if not only_only_correction:
            correct_file_name += "-only-correct-wq"
        else:
            correct_file_name += "-only-only-correct-wq"
        correct_file_name += "-"+self.generation_model_name.replace("/", "-")
        correct_file_name += f"-{self.correction_length}"
        if openai_models_dicts[self.correction_model_name]:
            corrected_outputs = run_openai(correct_file_name,
                                           correction_inputs, self.correction_model_name,
                                           max_tokens=self.correction_length)
        else:
            corrected_outputs = run_llm(correct_file_name, correction_inputs,
                                        self.correction_model_name, max_tokens=self.correction_length,
                                        loaded_model=llama_model, loaded_tokenizer=llama_tokenizer)
        assert len(model_outputs) == len(corrected_outputs)
        return corrected_outputs

    def iterative_run(self, num_iterative):
        last_iteration_generation_path = None
        for i in tqdm(range(num_iterative), total=num_iterative, desc="run_verification_correction"):
            if i == 0:
                model_outputs = get_outputs(self.generation_outputs_path)
            else:
                model_outputs = get_outputs(last_iteration_generation_path)
            break_down_inputs = generate_break_down_inputs(model_outputs)
            if openai_models_dicts[self.breakdown_model_name]:
                facts_outputs = run_openai(self.task + "{rag_name}-google-breakdown-para-llama3", break_down_inputs,
                                           self.breakdown_model_name)
            else:
                facts_outputs = run_llm(self.task + "{rag_name}-google-breakdown-para", break_down_inputs,
                                        self.breakdown_model_name)
            retrieved_contents = get_retrieved_outputs(self.retrieved_file_path)
            verification_inputs = generate_verification_inputs(facts_outputs, retrieved_contents=retrieved_contents)
            if openai_models_dicts[self.verification_model_name]:
                verified_outputs = run_openai(self.task + "{rag_name}-verify-google-llama3", verification_inputs,
                                              self.verification_model_name, max_tokens=self.verification_length)
            else:
                verified_outputs = run_llm(self.task + "{rag_name}-verify-google", verification_inputs,
                                           self.verification_model_name, max_tokens=self.verification_length)
            assert len(model_outputs) == len(verified_outputs) == len(retrieved_contents)
            filtered_inputs = []
            eid = 0
            llama_tokenizer = None
            llama_model = None
            if openai_models_dicts[self.correction_model_name]:
                llama_tokenizer = AutoTokenizer.from_pretrained(args.generation_model_name,
                                                                token="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")
                llama_model = AutoModelForCausalLM.from_pretrained(args.generation_model_name, device_map="auto",
                                                                   token="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")

            for facts_output, verified_output in zip(facts_outputs, verified_outputs):
                facts = facts_output.strip("\n").split("\n")
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
                if len(facts) == len(verified_results):
                    statements = ""
                    for fact, verified_result in zip(facts, verified_results):
                        if "not mentioned" in verified_result.lower():
                            if not self.filter_not_mentioned_statement:
                                statements += fact + "\n"
                        elif "false" in verified_result.lower():
                            if self.correct_false_statement:
                                statement_correct_input = generate_statement_correction_single(fact,
                                                                                               retrieved_contents[eid],
                                                                                               self.data[eid])
                                if openai_models_dicts[self.correction_model_name]:
                                    corrected_statement = run_openai(self.task + f"-statement_correct_{eid}",
                                                                     statement_correct_input,
                                                                     self.correction_model_name,
                                                                     max_tokens=128,
                                                                     save_to_file=False)  # , loaded_model=llama_model, loaded_tokenizer=llama_tokenizer)=
                                else:
                                    corrected_statement = run_llm(self.task + f"-statement_correct_{eid}",
                                                                  statement_correct_input, self.correction_model_name,
                                                                  max_tokens=128,
                                                                  save_to_file=False,
                                                                  loaded_model=llama_model,
                                                                  loaded_tokenizer=llama_tokenizer)
                                statements += corrected_statement[0] + "\n"
                        else:
                            statements += fact + "\n"
                    filtered_inputs.append(statements)
                else:
                    print(f"verified number example {eid} error, use original")
                    statements = ""
                    for fact in facts:
                        statements += fact + "\n"
                    filtered_inputs.append(statements)
                eid += 1
            new_model_outputs = []
            if self.correction_with_question:
                for question, answer in zip(self.data, model_outputs):
                    new_model_outputs.append(question + "\nAnswer:\"" + answer + "\"")
                correction_inputs = generate_correction_inputs(new_model_outputs, facts_outputs)
            else:
                correction_inputs = generate_correction_inputs(model_outputs, facts_outputs)
            if openai_models_dicts[self.correction_model_name]:
                corrected_outputs, outputs_path = run_openai(
                    self.task + f"-rag-correct-statement-google-llama3-512-128-iter{i + 1}",
                    correction_inputs, self.correction_model_name,
                    max_tokens=self.correction_length, output_file_path=True)
            else:
                corrected_outputs, outputs_path = run_llm(
                    self.task + f"-rag-correct-statement-wq-google-512-128-iter{i + 1}", correction_inputs,
                    self.correction_model_name, max_tokens=self.correction_length,
                    loaded_model=llama_model, loaded_tokenizer=llama_tokenizer,
                    output_file_path=True)
            assert len(model_outputs) == len(corrected_outputs)
            last_iteration_generation_path = outputs_path
        return last_iteration_generation_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, default="truthfulqa",
                        help="The task that will be used to run ")
    parser.add_argument("--generation_model_name", type=str, required=True, default="gpt-3.5-turbo")
    parser.add_argument("--breakdown_model_name", type=str, required=True, default="gpt-3.5-turbo")
    parser.add_argument("--verification_model_name", type=str, required=True, default="gpt-3.5-turbo-16k-0613")
    parser.add_argument("--correction_model_name", type=str, required=True, default="gpt-3.5-turbo-0613")
    parser.add_argument("--retrieval_file_path", type=str, required=True)
    parser.add_argument("--use_rag", action="store_true")
    parser.add_argument("--do_generation", action="store_true")
    parser.add_argument("--do_breakdown", action="store_true")
    parser.add_argument("--do_verification", action="store_true")
    parser.add_argument("--do_correction", action="store_true")
    parser.add_argument("--only_generation", action="store_true")
    parser.add_argument("--only_breakdown", action="store_true")
    parser.add_argument("--only_verification", action="store_true")
    parser.add_argument("--generation_outputs_path", type=str, default=None)
    parser.add_argument("--breakdown_outputs_path", type=str, default=None)
    parser.add_argument("--verification_outputs_path", type=str, default=None)
    parser.add_argument("--only_run_correction", action="store_true")
    parser.add_argument("--only_only_correction", action="store_true")
    parser.add_argument("--correct_false_statement", action="store_true")
    parser.add_argument("--correct_false_statement_with_question", action="store_true")
    parser.add_argument("--filter_not_mentioned_statement", action="store_true")
    parser.add_argument("--verification_with_question", action="store_true")
    parser.add_argument("--correction_with_question", action="store_true")
    parser.add_argument("--all_true_not_change", action="store_true")
    parser.add_argument("--debug_only", action="store_true")
    parser.add_argument('--api_key', type=str, default="api_key")
    parser.add_argument("--generation_length", type=int, default=1024)
    parser.add_argument("--verification_length", type=int, default=512)
    parser.add_argument("--correction_length", type=int, default=1024)
    parser.add_argument("--run_iterative", action="store_true")
    parser.add_argument("--num_iterative", type=int, default=3)
    args = parser.parse_args()
    pipe = Pipeline(args.task,
                    args.generation_model_name,
                    args.breakdown_model_name,
                    args.verification_model_name,
                    args.correction_model_name,
                    args.retrieval_file_path,
                    use_rag=args.use_rag,
                    do_generation=args.do_generation,
                    do_breakdown=args.do_breakdown,
                    do_verification=args.do_verification,
                    do_correction=args.do_correction,
                    only_generation=args.only_generation,
                    only_breakdown=args.only_breakdown,
                    only_verification=args.only_verification,
                    generation_outputs_path=args.generation_outputs_path,
                    breakdown_outputs_path=args.breakdown_outputs_path,
                    verification_outputs_path=args.verification_outputs_path,
                    correct_false_statement=args.correct_false_statement,
                    filter_not_mentioned_statement=args.filter_not_mentioned_statement,
                    correct_false_statement_with_question=args.correct_false_statement_with_question,
                    verification_with_question=args.verification_with_question,
                    correction_with_question=args.correction_with_question,
                    debug_only=args.debug_only,
                    api_key=args.api_key,
                    generation_length=args.generation_length,
                    verification_length=args.verification_length,
                    correction_length=args.correction_length,
                    all_true_not_change=args.all_true_not_change)
    if args.only_run_correction:
        pipe.only_run_correction(args.only_only_correction)
    elif args.run_iterative:
        pipe.iterative_run(args.num_iterative)
    else:
        pipe.run()
