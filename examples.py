import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model.memnn import KVMMModel
from reader import Vocabulary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pad_length = 20

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


def main_examples(dialog_type, underscore, kb, iteration=500000, delimiter=",", final_folder="", test_mode=""):
    if final_folder == "":
        final_folder = dialog_type[3:]
    try:
        df = pd.read_csv("data/test_data" + underscore + test_mode + dialog_type + kb + ".csv", encoding="ISO-8859-1", delimiter=delimiter)
    except FileNotFoundError:
        df = pd.read_csv("../data/test_data" + underscore + test_mode + dialog_type + kb + ".csv", encoding="ISO-8859-1", delimiter=delimiter)
    inputs = list(df["input"])
    outputs = list(df["output"])
    actual_clusters = list(df["actual_cluster"])
    try:
        vocab = Vocabulary('data/vocabulary-train' + dialog_type + ' - perfect decomposition - preprocessedFinal.json', padding=pad_length)
        kb_vocabulary = Vocabulary('data/vocabulary-train' + dialog_type + ' - perfect decomposition - preprocessedFinal.json', padding=4)
    except FileNotFoundError:
        vocab = Vocabulary('../data/vocabulary-train' + dialog_type + ' - perfect decomposition - preprocessedFinal.json', padding=pad_length)
        kb_vocabulary = Vocabulary('../data/vocabulary-train' + dialog_type + ' - perfect decomposition - preprocessedFinal.json', padding=4)

    model = KVMMModel(pad_length=20,
                      embedding_size=200,
                      batch_size=1,
                      vocab_size=vocab.size(),
                      n_chars=vocab.size(),
                      n_labels=vocab.size(),
                      encoder_units=200,
                      decoder_units=200).to(device)
    weights_file = "final-" + final_folder + "/model_weights_" + dialog_type[3:] + "_iter_" + str(iteration) + ".pytorch"
    try:
        try:
            model.load_state_dict(torch.load(weights_file))
        except RuntimeError:
            model.load_state_dict(torch.load(weights_file, map_location='cpu'))
    except FileNotFoundError:
        try:
            model.load_state_dict(torch.load("../" + weights_file))
        except RuntimeError:
            model.load_state_dict(torch.load("../" + weights_file, map_location='cpu'))

    kbfile = "data/normalised_kbtuples.csv"
    try:
        df = pd.read_csv(kbfile)
    except FileNotFoundError:
        df = pd.read_csv("../" + kbfile)
    kbs = list(df["subject"] + " " + df["relation"])
    # print(kbs[:3])
    kbs = np.array(list(map(kb_vocabulary.string_to_int, kbs)))
    kbs = np.repeat(kbs[np.newaxis, :, :], 1, axis=0)
    data = run_examples(model, kbs, vocab, inputs, outputs)
    df = pd.DataFrame(columns=["inputs", "outputs", "prediction", "actual_cluster"])
    d = {'outputs': [], 'inputs': [], 'predictions': [], "actual_cluster": []}
    for index, (i, o, p) in enumerate(zip(inputs, outputs, data)):
        d["outputs"].append(str(o))
        d["inputs"].append(str(i))
        d["predictions"].append(str(p))
        d["actual_cluster"].append(actual_clusters[index])
    df = pd.DataFrame(d)
    return df
    # print(outputs)


if __name__ == "__main__":
    dialog_type = " - full"
    underscore = "_"
    kb = " - kb"
    result_df = main_examples(dialog_type, underscore, kb)
    result_df.to_csv("final-full/output_kb" + dialog_type + ".csv", encoding="ISO-8859-1", sep=";")
