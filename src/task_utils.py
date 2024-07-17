import csv

supported_tasks = {"factscore":"data/factscore/prompt_entities.txt",
                   "truthfulqa":"data/truthfulqa/TruthfulQA.csv"}

def read_task_examples(task):
    examples = []
    if task == "factscore":
        data_path = supported_tasks [task]
        with open(data_path) as f:
            lines = f.readlines()
            for line in lines:
                examples.append(line.strip())
    elif task == "truthfulqa":
        data_path = supported_tasks[task]
        with open(data_path) as f:
            reader = csv.DictReader(f)
            for example in reader:
                examples.append(example)
    return examples


def generate_prompt_input_examples(task):
    task_examples = read_task_examples(task)
    input_texts = []
    if task == "factscore":
        for example in task_examples:
            input_text = f"Question: Tell me a bio of {example}."
            input_texts.append(input_text)
    elif task == "truthfulqa":
        for example in task_examples:
            input_text = f"Question: {example['Question']}"
            input_texts.append(input_text)
    return input_texts


def get_outputs(output_path):
    predicted_labels = []
    temp_str = ""
    with open(output_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.strip("\n") == "---":
                predicted_labels.append(temp_str)
                temp_str = ""
            else:
                temp_str += line
    return predicted_labels


if __name__ == '__main__':
    test_examples = generate_prompt_input_examples("truthfulqa")