import pandas as pd

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import bleu_score

smt = bleu_score.SmoothingFunction()


def load_output():
    return pd.read_csv("output_kb.csv", encoding="ISO-8859-1", sep=',')


def save_scores(output_file, scores):
    output_file['bleu_score'] = pd.Series(scores)
    output_file.to_csv('output_kb-bleu-eval.csv', index=False, encoding="ISO-8859-1", sep=';', decimal=',')


def main():
    output_file = load_output()
    predicted = output_file['u1']
    expected = output_file['outputs']
    # predicted = ["What city do you want the weather for?"]
    # expected = ["What city do you want the weather for?"]
    scores = []
    # for out,pred in zip(expected["output"], predicted["u1"]):
    for out, pred in zip(expected, predicted):
        out = out.replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip()
        pred = pred.replace("<unk>", "").replace("<eos>", "").replace("<pad>", "").replace("_", " ").strip()
        # print(type(eval(pred)))
        print(out, "&", pred)
        try:
            score = sentence_bleu([out.split(" ")], pred.split(" "), smoothing_function=smt.method7)
            print("SCORE: " + str(score))
            scores.append(score)
        except:
            scores.append(0)
    c = 0
    for s in scores:
        c = c + s
    print(c / len(scores), len(scores))
    save_scores(output_file, scores)


if __name__ == "__main__":
    main()
