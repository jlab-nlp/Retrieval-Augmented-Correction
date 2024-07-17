import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
import json
from retriever import get_retrieved_outputs
from collections import Counter
from transformers import AutoTokenizer
import csv
import tiktoken
cse_id = "f6281fd1b8ecd4256"
api_key = "f7503b390362b46d65f0aa5eb1c536d4800ff2f5"

def google_search(search_term, top=10, filter_link_keywords=[], is_clean=True):
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": search_term
    })
    headers = {
        'X-API-KEY': 'e5916dbedb24df7a1bb51bbe7c7b60f564bbec6a',
        'Content-Type': 'application/json'
    }
    reconnect = 0
    while reconnect < 3:
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            break
        except (requests.exceptions.RequestException, ValueError):
            reconnect += 1
            print('url: {} failed * {}'.format(url, reconnect))
    json.loads(response.text)
    result = json.loads(response.text)
    results = result["organic"][:top]
    documents = []
    for result in results:
        link = result["link"]
        flag = False
        for keyword in filter_link_keywords:
            if keyword.lower() in link.lower():
                flag = True
                break
        if flag:
            continue
        try:
            link_result = requests.get(link, timeout=3)
        except:
            continue
        if is_clean:
            cleantext = BeautifulSoup(link_result.text, "lxml").text
            cleantext = re.sub(r'\n+', '\n', cleantext)
            documents.append(cleantext)
        else:
            documents.append(link_result.text)
    return documents


def generate_retrieved_content_for_factscore(input_file="data/factscore/prompt_entities.txt",
                                             output_file="factscore_google_retrieval.out"):
    search_entities = []
    with open(input_file) as f:
        lines = f.readlines()
        for line in lines:
            if line:
                search_entities.append(line.strip())
    with open(output_file, "w") as f:
        for entity in tqdm(search_entities, total=len(search_entities)):
            keyword = f"\"{entity}\" Wikipedia"
            documents = google_search(keyword, top=10)
            for document in documents:
                if entity in document:
                    f.write(document.split("From Wikipedia, the free encyclopedia\n")[1].split("\nReferences[edit]")[0]+"\n-----\n")
                    break

def post_processing_factscore_google(input_file="factscore_google_retrieval.out",
                                     output_file="factscore_google_retrieval_post.out"):
    retrieved_documents = get_retrieved_outputs(input_file)
    new_documents = []
    for retrieved_document in retrieved_documents:
        lines = retrieved_document.split("\n")
        new_line = ""
        for line in lines:
            if "[edit]" not in line and len(line) <= 82:
                continue
            else:
                new_line += line + "\n"
        new_documents.append(new_line)
    with open(output_file, "w") as f:
        for new_document in new_documents:
            new_document = re.sub(r"\[edit]", "", new_document[:-1])
            new_document = new_document.split("\nFilmography\n")[0]
            new_document = new_document.split("\nReferences\n")[0]
            new_document = new_document.split("\nProduction\n")[0]
            new_document = new_document.split("\nDiscography\n")[0]
            new_document = new_document.split("\nCareer statistics\n")[0]
            new_document = new_document.split("\nFootnotes\n")[0]
            new_document = new_document.split("\nNotes\n")[0]
            new_document = new_document.split("\nNotes and references\n")[0]
            if new_document[-1] == "\n":
                new_document = new_document[:-1]
            f.write(new_document+"\n-----\n")


def generate_retrieved_content_for_truthfulqa(input_file="data/truthfulqa/TruthfulQA.csv",
                                             output_file="truthfulqa_google_retrieval_top_30.out"):
    search_questions = []
    filter_keywords = ["huggingface", "paperswithcode", "kaggle", "openreview", "github", "arxiv"]
    with open(input_file) as f:
        reader = csv.DictReader(f)
        for example in reader:
            search_questions.append(example['Question'])
    with open(output_file, "w") as f:
        for question in tqdm(search_questions, total=len(search_questions)):
            if question[0] == "\"":
                documents = google_search(question[0:][:-1], top=30, filter_link_keywords=filter_keywords, is_clean=False)
            else:
                documents = google_search(question, top=30, filter_link_keywords=filter_keywords, is_clean=False)
            for document in documents:
                f.write(document + "\n---\n")
            f.write("\n------\n")


def post_processing_truthfulqa_google(input_file="truthfulqa_google_retrieval_top_30_post.out",
                                      output_file="truthfulqa_google_retrieval_top_30_post_post.out"):
    retrieved_documents = get_retrieved_outputs(input_file, splitter="-----")
    with open(output_file, "w") as f:
        for retrieved_document in tqdm(retrieved_documents, total=len(retrieved_documents)):
            sub_documents = retrieved_document.split("\n---\n")
            for document in sub_documents:
                if document:
                    soup = BeautifulSoup(document)
                    text = soup.get_text("\n")
                    text = re.sub(r'\n+\s*', '\n', text)
                    lines = text.split("\n")
                    new_line = ""
                    line_number = 0
                    for line in lines:
                        if len(line) > 82 and "�" not in line and "/font" not in line.lower():
                            new_line += line.replace("\xa0", " ").encode('ascii', errors='ignore').decode('ascii') + "\n"
                            line_number += 1
                    if new_line != "" and line_number > 1:
                        f.write(new_line[:-1]+"\n---\n")
            f.write("\n-----\n")

def check_truthfulqa_google_content(input_file="truthfulqa_google_retrieval_top_30_post_post_trim.out",
                                    question_input="data/truthfulqa/TruthfulQA.csv"):
    search_questions = []
    with open(question_input) as f:
        reader = csv.DictReader(f)
        for example in reader:
            search_questions.append(example['Question'])
    retrieved_documents = get_retrieved_outputs(input_file, splitter="-----")
    assert len(retrieved_documents) == len(search_questions)
    counter = Counter()
    for question, documents in zip(search_questions, retrieved_documents):
        docs = documents.split("---")
        counter[len(docs[:-1])] += 1
        if len(docs[:-1]) == 1:
            print("1:"+question)
            print("1:"+documents)
        elif len(docs[:-1]) == 2:
            print("2:"+question)
            print("2:" + documents)
    print(sum(list(counter.values())))
    print(sorted(counter.items(), key=lambda i: i[0], reverse=True))

    exit(0)


def trim_retrieved__by_length(retrieve_file="factscore_google_retrieval_post.out", model="meta-llama/Llama-2-7b-chat-hf",
                              max_len=3000):
    if model == "gpt-3.5-turbo":
        tokenizer = tiktoken.encoding_for_model(model)
    elif model == "meta-llama/Llama-2-7b-chat-hf":
        tokenizer = AutoTokenizer.from_pretrained(model,token="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")
    retrieved_content = get_retrieved_outputs(retrieve_file)
    new_retrieved_content = []
    for docs in tqdm(retrieved_content, total=len(retrieved_content)):
        doc_tokens = tokenizer.encode(docs)
        if len(doc_tokens) > max_len:
            new_retrieved_content.append(tokenizer.decode(doc_tokens[:max_len]))
        else:
            new_retrieved_content.append(docs)
    with open(retrieve_file.split(".")[0]+"_trim_llama.out", "w") as f:
        for retrieved_document in tqdm(new_retrieved_content, total=len(new_retrieved_content)):
            if "\n---\n" in retrieved_document:
                sub_docs = retrieved_document.split("\n---\n")
                if sub_docs[-1].strip() == "":
                    sub_docs = sub_docs[:-1]
                for sub_doc in sub_docs:
                    if "" not in sub_doc:
                        f.write(sub_doc + "\n---\n")
            else:
                f.write(retrieved_document[:-1])
            f.write("\n-----\n")


if __name__ == '__main__':
    trim_retrieved__by_length()
    exit(0)
    # exit(0)
    check_truthfulqa_google_content()
    #generate_retrieved_content_for_truthfulqa(input_file="data/truthfulqa/truthfulqa_p.csv", output_file="truthfulqa_p_google_retrieval_top_30.out")
    # exit(0)
    # generate_retrieved_content_for_factscore()
    # post_processing_factscore_google()
    #post_processing_truthfulqa_google()