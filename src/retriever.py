from opensearchpy import OpenSearch
import os
from tqdm import tqdm


def knn_bert_search(client, query_example, top, index='pile-nlp-index'):
    query = {
        "size": top,
        "query": {
            "neural": {
                "text_embedding": {
                    "query_text": query_example,
                    "model_id": "aNyqao0BVPT9SGgabK44",
                    "k": 100
                }
            }
        },
        "fields": ["text", "source", "tid"],
        "_source": False
    }
    response = client.search(
        body=query,
        index=index
    )
    text = ""
    for i, result in enumerate(response["hits"]["hits"]):
        text += result["fields"]["text"][0] + "\n---\n"
    return text[:-2]


def knn_search(client, query_example, top, index='pile-nlp-index-1'):
    query = {
        "size": top,
        "query": {
            "neural": {
                "text_embedding": {
                    "query_text": "query: " + query_example,
                    "model_id": "htzibY0BVPT9SGgav652",
                    "k": 100
                }
            }
        },
        "fields": ["text", "source", "tid"],
        "_source": False
    }
    response = client.search(
        body=query,
        index=index
    )
    text = ""
    for i, result in enumerate(response["hits"]["hits"]):
        text += result["fields"]["text"][0].replace("passage: ", "") + "\n---\n"
    return text[:-2]


def bm25_search(client, query_example, index='pile-index-nlp-1', top=5):
    query = {
        "size": top,
        "query": {
            "match": {
                "text": query_example
            }
        },
        "fields": ["text", "source", "tid"],
        "_source": False
    }

    response = client.search(
        body=query,
        index=index
    )
    text = ""
    for i, result in enumerate(response["hits"]["hits"]):
        text += result["fields"]["text"][0].replace("passage: ", "") + "\n---\n"
    return text[:-2]


def retrieve(task, examples, top=5, index='pile-index-nlp-1', retrieve_type="bm25", save_to_file=True, output_dir="outputs",
             model_name="gpt-3.5-turbo"):
    retrieve_contents = []
    host = 'localhost'
    port = 9200

    # Create the client with SSL/TLS and hostname verification disabled.
    client = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_compress=True,  # enables gzip compression for request bodies
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False
    )
    for example in tqdm(examples, total=len(examples), desc="retrieving"):
        if retrieve_type == "bm25":
            response = bm25_search(client, example, index, top)
        elif retrieve_type == "knn-bert":
            response = knn_bert_search(client, example, top, index)
        elif retrieve_type == "knn-e5":
            response = knn_bert_search(client, example, top, index)
        else:
            raise RuntimeError("Unknown retrieve type!")
        retrieve_contents.append(response)
    if save_to_file:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        with open(os.path.join(output_dir, task + "-" + model_name + ".out"), "w") as f:
            for model_out in retrieve_contents:
                f.write(model_out + "\n-----\n")
    return retrieve_contents


def get_retrieved_outputs(output_path, splitter="-----"):
    predicted_labels = []
    temp_str = ""
    with open(output_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.strip("\n") == splitter:
                predicted_labels.append(temp_str)
                temp_str = ""
            else:
                temp_str += line
    return predicted_labels


if __name__ == '__main__':
    host = 'localhost'
    port = 9200

    # Create the client with SSL/TLS and hostname verification disabled.
    client = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_compress=True,  # enables gzip compression for request bodies
        use_ssl=False,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False
    )
    print(bm25_search(client, "This is a map", index="pile-index-nlp-1", top=5))
