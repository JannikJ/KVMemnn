import json
import re


kb_vocab_file_name = "../../data/kvret_entities_cleaning.json"
loaded_vocabulary = False
kb_vocabulary = {}
sorted_kb_vocabulary = []


def load_kb_vocabulary():
    global kb_vocabulary
    global loaded_vocabulary
    global sorted_kb_vocabulary
    try:
        with open(kb_vocab_file_name, 'r', encoding='utf-8') as f:
            loaded_file = json.load(f)
    except FileNotFoundError:
        with open("../" + kb_vocab_file_name, 'r', encoding='utf-8') as f:
            loaded_file = json.load(f)
    for tag in loaded_file:
        print(tag)
        if tag == "poi":
            for poi in loaded_file[tag]:
                kb_vocabulary[poi["address"]] = "poi_address"
                kb_vocabulary[poi["poi"]] = "poi_poi"
                kb_vocabulary[poi["type"]] = "poi_type"
        else:
            for real_tag in loaded_file[tag]:
                kb_vocabulary[real_tag] = tag
    sorted_kb_vocabulary = list(kb_vocabulary.items())
    sorted_kb_vocabulary.sort(key=lambda tup: str(tup[0]), reverse=True)
    loaded_vocabulary = True


def replace_with_tokens(text):
    global kb_vocabulary
    if not loaded_vocabulary:
        load_kb_vocabulary()
    for token in sorted_kb_vocabulary:
        text = re.sub(r"\b" + str(token[0]) + r"s?\b", " <" + token[1].strip() + "> ", text, flags=re.IGNORECASE)
    return text


if __name__ == "__main__":
    print(replace_with_tokens("What is the temperature over the next two days in Mountain View and where is the next pizza restaurant. The temperature is 20f."))
