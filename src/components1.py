
from nltk import sent_tokenize
supported_components = ["rag","rag-find", "facts_filter", "breakdown-para", "breakdown-sent", "verification",
                        "correction", "statement_correction"]


def generate_components_prompts_with_inputs(component, input_text, retrieved=None, question=None):
    if component == "rag":
        return f"{input_text}\nAnswer based on the following text and keep the answer authentic to the texts:\n\"{retrieved}\"\n\nAnswer:\n"
    elif component == "rag-find":
        retrieved = retrieved.strip("\n")
        return (f"Passages:\"{retrieved}\"\n{input_text}\nPlease find the answer to the question from the above passages and generate the answer text. "
                f"If there is an answer in the documents, please keep the answer authentic to the passage, "
                f"if the question is to ask for opinion or if there is no answer found in the documents, "
                f"please output \"I have no comment\".\nAnswer:\n")
    elif component == "filter":
        return (f"\"{input_text}\"\n\nIs the above content factual (Yes or No)?\n")
    elif component == "breakdown-para":
        # The input text should be a paragraph (the model output)
        input_text = input_text.strip("\n")
        return (f"Please breakdown the following content into independent facts without pronouns(Do not use He, She, It...)"
                f"(each fact should be a full sentence, each fact per line):\"{input_text}\"\nFacts:\n")
    elif component == "breakdown-sent":
        # the input text should be a sentence
        return (f"Please breakdown the following sentence into independent facts each facts without "
                f"pronouns (each fact per line without line number):\"{input_text}\"\n")
    elif component == "verification":
        # the input text should be retrieved context and break-down facts.
        if retrieved:
            if question:
                input_text = input_text.strip("\n")
                return (
                    f"{question}\npassage:\"{retrieved}\"\nPlease verify the below statements to the above question into true or false or not "
                    f"mentioned based on the above passages (one answer per line with label true or false or not "
                    f"mentioned.)\nTrue means the similar statement can be found in the above passage and have the "
                    f"same meaning.\nFalse means the similar statement can be found in the above passage  but have "
                    f"the different meaning.\nNot Mentioned means the similar statement cannot be found in the above "
                    f"passage.\n\nStatements:\"{input_text}\"\n\nOutput Format:\nStatement 1: True\nStatement 2: False \n ... \nStatement N: Not Mentioned\n\nAnswer(start with the output directly without additional comments):\n")
            else:
                input_text = input_text.strip("\n")
                return (
                    f"passage:\"{retrieved}\"\nPlease verify the below statements into true or false or not "
                    f"mentioned based on the above passages (one answer per line with label true or false or not "
                    f"mentioned.)\nTrue means the similar statement can be found in the above passage and have the "
                    f"same meaning.\nFalse means the similar statement can be found in the above passage  but have "
                    f"the different meaning.\nNot Mentioned means the similar statement cannot be found in the above "
                    f"passage.\n\nStatements:\"{input_text}\"\n\nOutput Format:\nStatement 1: True\nStatement 2: False \n ... \nStatement N: Not Mentioned\n\nAnswer(start with the output directly without additional comments):\n")

        else:
            raise RuntimeError("retrieved content cannot be empty.")
    elif component == "correction":
        if retrieved:
            # the input_text should be the model output
            input_text = input_text.strip("\n")
            retrieved = retrieved.strip("\n")
            return (f"{input_text}\n\nPlease correct the above answer into a corrected one "
                    f"based on the following verified facts. In your answer, start with the corrected answer directly "
                    f"without repeating the question or the original answer. "
                    f"\n\nVerified facts:\"{retrieved}\"\n\nCorrected answer:\n")
                    #f"if the answer is \"I have no comment\", output \"I have no comment\"."
                    #f"\n\nVerified facts:\"{retrieved}\"\n\nCorrected answer:\n")
                    #f"if the answer is \"I have no comment\", output \"I have no comment\"."
                    #f"\n\nVerified facts:\"{retrieved}\"\n\nCorrected answer:\n")
        else:
            raise RuntimeError("verified statements cannot be empty.")
    elif component == "statement_correction":
        if retrieved:
            if question:
                return (f"{question}\npassage:\"{retrieved}\"\nCorrect the following statement and output the corrected version "
                        f"based on the above passage. In your answer, start with the corrected answer directly without repeating the question or the original statement. \n\nStatement:\"{input_text}\"\n\nAnswer:\n")
            else:
                return (f"passage:\"{retrieved}\"\nCorrect the following statement and output the corrected version "
                        f"based on the above passage. In your answer, start with the corrected answer directly without repeating the question or the original statement. \n\nStatement:\"{input_text}\"\n\nAnswer:\n")
        else:
            raise RuntimeError("retrieved content cannot be empty.")
    elif component == "statement_correction_all":
        if retrieved:
            if question:
                return (f"{question}\npassage:\"{retrieved}\"\nCorrect the following statement and output the corrected version "
                        f"based on the above passage. If the statement is correct, directly output the original statement. "
                        f"In your answer, start with the corrected answer or original correct statement directly without repeating the question. "
                        f"The answer should be a single sentence and should be concise and to the point of the question. "
                        f"\n\nStatement:\"{input_text}\"\n\nAnswer:\n")
            else:
                RuntimeError("question cannot be empty.")

        else:
            raise RuntimeError("retrieved content cannot be empty.")


def generate_break_down_inputs(model_outputs, break_strategy="paragraph"):
    input_examples = []
    if break_strategy == "paragraph":
        for model_output in model_outputs:
            input_examples.append(generate_components_prompts_with_inputs("breakdown-para", model_output))
    elif break_strategy == "sentence":
        for model_output in model_outputs:
            sentences = sent_tokenize(model_output)
            input_examples.append([generate_components_prompts_with_inputs("breakdown-sent", sentence)
                                   for sentence in sentences])
    else:
        raise RuntimeError("unknown paragraph strategy, valid: paragraph or sentence")
    return input_examples


def generate_verification_inputs(breakdown_facts, retrieved_contents, questions=None):
    numbered_facts = []
    for breakdown_fact in breakdown_facts:
        facts = breakdown_fact.split("\n")
        new_fact = ""
        n = 0
        for fact in facts:
            if fact:
                fact_number = n + 1
                if fact[0].isdigit():
                    new_fact += f"Statement {fact}" + "\n"
                else:
                    new_fact += f"Statement {fact_number}: {fact}" + "\n"
                n += 1
        numbered_facts.append(new_fact)
    #docs = retrieved_contents.split("---")[:-1]
    input_examples = []
    if questions:
        for breakdown_fact, retrieved_content, question in zip(numbered_facts, retrieved_contents, questions):
            input_examples.append(generate_components_prompts_with_inputs("verification",
                                                                          breakdown_fact,
                                                                          retrieved_content, question))
    else:
        for breakdown_fact, retrieved_content in zip(numbered_facts, retrieved_contents):
            # docs = retrieved_content.split("---")[:-1]
            # retrieved_content = "\n".join(docs)
            input_examples.append(generate_components_prompts_with_inputs("verification",
                                                                          breakdown_fact,
                                                                          retrieved_content))

    return input_examples


# def generate_statement_correction_inputs(false_statements, retrieved_contents):
#     input_examples = []
#     for false_statement, retrieved_content in zip(false_statements, retrieved_contents):
#         input_examples.append(generate_components_prompts_with_inputs("statement_correction",
#                                                                       false_statement,
#                                                                       retrieved_content))

def generate_correction_inputs(model_outputs, verified_facts):
    input_examples = []
    count = 0
    for model_output, verified_fact in zip(model_outputs, verified_facts):
        #print(verified_fact)
        #print(count)
        count+=1
        input_examples.append(generate_components_prompts_with_inputs("correction",
                                                                      model_output,
                                                                      verified_fact))
    return input_examples


def generate_rag_inputs(model_inputs, retrieved_inputs):
    input_examples = []
    for model_input, retrieved_input in zip(model_inputs, retrieved_inputs):
        input_examples.append(generate_components_prompts_with_inputs("rag", model_input, retrieved_input))
    return input_examples

def generate_rag_find_inputs(model_inputs, retrieved_inputs, top=3):
    input_examples = []
    for model_input, retrieved_input in zip(model_inputs, retrieved_inputs):
        docs = retrieved_input.split("---")[:-1]
        #if len(docs) <= top:
        #    input_examples.append(generate_components_prompts_with_inputs("rag-find", model_input, retrieved_input[:-6]))
        #else:
        new_retrieved_input = "\n".join(docs)
        input_examples.append(generate_components_prompts_with_inputs("rag-find", model_input, new_retrieved_input[:-1]))
    return input_examples

def generate_statement_correction_single(model_input, retrieved_input, question=None):
    return [generate_components_prompts_with_inputs("statement_correction", model_input, retrieved_input, question)]

def generate_statement_correction_all(model_input, retrieved_input, question):
    return [generate_components_prompts_with_inputs("statement_correction_all", model_input, retrieved_input, question)]



