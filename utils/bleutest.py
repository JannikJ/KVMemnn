import heapq, sys, os
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import pandas as pd
from nltk.translate import bleu_score

dialog_type = " - original"
smt = bleu_score.SmoothingFunction()
df = pd.read_csv("../output_kb" + dialog_type + ".csv", encoding="ISO-8859-1", sep=";")
df.dropna(inplace=True)

scores = []
for index, row in df.iterrows():
    groundtruth = row['outputs']
    output = row['predictions'].replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip()
    try:
        scores.append(sentence_bleu([output.split(" ")], groundtruth.split(" "), smoothing_function=smt.method7))
    except:
        scores.append(0)

c = 0
for s in scores:
    c = c + s
print(c / len(scores), len(scores))
