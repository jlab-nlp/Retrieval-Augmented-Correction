import sqlite3
if __name__ == '__main__':
    con = sqlite3.connect(".cache/factscore/enwiki-20230401.db")
    cur = con.cursor()
    retrieved_texts = []
    entitles = []
    with open("data/factscore/prompt_entities.txt") as f:
        lines = f.readlines()
        for line in lines:
            if line:
                entity = line.strip()
                entitles.append(entity)
                res = cur.execute(f"SELECT text FROM documents WHERE title = \"{entity}\"")
                results = res.fetchall()
                if len(results) > 0:
                    sentences = results[0][0].split("</s><s>")
                    # text = ""
                    # for sentence in sentences[:2]:
                    #     text = text + "\n" + sentence.replace("<s>", "")
                    # retrieved_texts.append(text)
                    retrieved_texts.append(results[0][0].replace("</s><s>", "\n")
                                           .replace("<s>", "")
                                           .replace("</s>", ""))
                else:
                    print(entity)
    assert len(entitles) == len(retrieved_texts)
    with open("factscore_gold_retrieved.txt", "w") as f:
        for retrieved_text in retrieved_texts:
            f.write(retrieved_text + "\n-----\n")