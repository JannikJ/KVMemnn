import pandas as pd

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate import bleu_score
import spacy

from examples import main_examples

use_spacy = False
if use_spacy:
    nlp = spacy.load('en')
smt = bleu_score.SmoothingFunction()
file_name_suffix = "- original"
sentence_level = True
generate_examples = True


def load_output():
    if generate_examples:
        return main_examples(" " + file_name_suffix, "_", " - kb")
    else:
        return pd.read_csv("../output_kb" + file_name_suffix + ".csv", encoding="ISO-8859-1", sep=';')


def save_scores(output_file, scores):
    output_file['bleu_score'] = pd.Series(scores)
    output_file.to_csv("output_kb" + file_name_suffix + "-bleu-eval.csv", index=False, encoding="ISO-8859-1", sep=';', decimal=',')


def main():
    output_file = load_output()
    predicted = output_file['predictions']
    expected = output_file['outputs']
    # predicted = ["What city do you want the weather for?"]
    # expected = ["What city do you want the weather for?"]
    scores = []
    # for out,pred in zip(expected["output"], predicted["u1"]):
    if use_spacy:
        predicted = [pred.replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip() for pred in predicted]
        expected = [out.replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip() for out in expected]
        predicted_tokens = [nlp(pred) for pred in predicted]
        expected_tokens = [nlp(out) for out in expected]
        for index, _ in enumerate(predicted_tokens):
            score = expected_tokens[index].similarity(predicted_tokens[index])
            scores.append(score)
            print(expected_tokens[index], "&", predicted_tokens[index])
            print("SCORE: " + str(score))
        c = 0
        for s in scores:
            c = c + s
        print(c / len(scores), len(scores))
        save_scores(output_file, scores)
    else:
        if sentence_level:
            for out, pred in zip(expected, predicted):
                out = out.replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip()
                pred = pred.replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip()
                # print(type(eval(pred)))
                print(out, "&", pred)
                try:
                    score = sentence_bleu([pred.split(" ")], out.split(" "), smoothing_function=smt.method7)
                    print("SCORE: " + str(score))
                    scores.append(score)
                except:
                    scores.append(0)
            c = 0
            for s in scores:
                c = c + s
            print(c / len(scores), len(scores))
            save_scores(output_file, scores)
        else:
            out = [out.replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip() for out in expected]
            pred = [pred.replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip() for pred in predicted]
            score = corpus_bleu(out, pred, smoothing_function=smt.method7)
            print(str(score))


if __name__ == "__main__":
    main()
