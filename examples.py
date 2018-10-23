import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model.memnn import KVMMModel
from reader import Vocabulary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# beam search
def run_example(model, kbs, vocabulary, text, groundtruth):
    encoded = vocabulary.string_to_int(text)
    input_tensors = torch.from_numpy(np.expand_dims(encoded, axis=0))
    kbs = torch.from_numpy(np.expand_dims(kbs, axis=0))
    prediction = model(input_tensors, kbs[0])
    prediction = F.softmax(prediction, dim=2)
    # print(prediction[0])
    # sys.exit(1)
    print('input:', text)
    print('groundtruth:', groundtruth)
    print('symbol prediction:', ' '.join(vocabulary.int_to_string(prediction[0].max(1)[1].detach().cpu().numpy())))
    output = ' '.join(vocabulary.int_to_string(prediction[0].max(1)[1].detach().cpu().numpy()))
    return output


def run_examples(model, kbs, vocabulary, examples, groundtruths):
    predicted = []
    input = []
    for example, groundtruth in zip(examples, groundtruths):
        print('~~~~~')
        input.append(example)
        predicted.append(run_example(model, kbs, vocabulary, example, groundtruth))
    return predicted


def main_examples(dialog_type, underscore, kb, iteration=500000):
    df = pd.read_csv("data/test_data" + underscore + dialog_type + kb + ".csv", encoding="ISO-8859-1", delimiter=',')
    inputs = list(df["input"])
    outputs = list(df["output"])
    vocab = Vocabulary('data/vocabulary' + dialog_type + '.json', padding=pad_length)

    kb_vocabulary = Vocabulary('data/vocabulary' + dialog_type + '.json', padding=4)

    model = KVMMModel(pad_length=20,
                      embedding_size=200,
                      batch_size=1,
                      vocab_size=vocab.size(),
                      n_chars=vocab.size(),
                      n_labels=vocab.size(),
                      encoder_units=200,
                      decoder_units=200).to(device)
    weights_file = "model_weights_" + dialog_type[3:] + "_iter_" + iteration + ".pytorch"
    model.load_state_dict(torch.load(weights_file))

    kbfile = "data/normalised_kbtuples.csv"
    df = pd.read_csv(kbfile)
    kbs = list(df["subject"] + " " + df["relation"])
    # print(kbs[:3])
    kbs = np.array(list(map(kb_vocabulary.string_to_int, kbs)))
    kbs = np.repeat(kbs[np.newaxis, :, :], 1, axis=0)
    data = run_examples(model, kbs, vocab, inputs, outputs)
    df = pd.DataFrame(columns=["inputs", "outputs", "prediction"])
    d = {'outputs': [], 'inputs': [], 'predictions': []}
    for i, o, p in zip(inputs, outputs, data):
        d["outputs"].append(str(o))
        d["inputs"].append(str(i))
        d["predictions"].append(str(p))
    df = pd.DataFrame(d)
    return df
    # print(outputs)


if __name__ == "__main__":
    pad_length = 20
    dialog_type = " - navigate"
    underscore = "_"
    kb = " - kb"
    result_df = main_examples(dialog_type, underscore, kb)
    result_df.to_csv("output_kb" + dialog_type + ".csv", encoding="ISO-8859-1", sep=";")
