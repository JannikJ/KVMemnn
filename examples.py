import re

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
    if encoded.__contains__(1518):
        print("UNK")
    for enc in encoded:
        if enc > 6085:
            print("BIG")
    input_tensors = torch.from_numpy(np.expand_dims(encoded, axis=0))
    kbs = torch.from_numpy(np.expand_dims(kbs, axis=0))
    prediction = model(input_tensors, kbs[0])
    prediction = F.softmax(prediction, dim=2)
    # print(prediction[0])
    # sys.exit(1)
    print('input:', text)
    print("encoded input:", vocabulary.int_to_string(encoded))
    print('groundtruth:', groundtruth)
    print('symbol prediction:', ' '.join(vocabulary.int_to_string(prediction[0].max(1)[1].detach().cpu().numpy())))
    for out in prediction[0].max(1)[1].detach().cpu().numpy():
        if out > 6085:
            print("BIG")
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


if __name__ == "__main__":
    use_original = True
    pad_length = 20
    dialog_type = " - original"
    underscore = "_"
    kb = " - kb"
    if use_original:
        old_dialog_type = dialog_type
        dialog_type = ""
        underscore = ""
        kb = ""
    df = pd.read_csv("data/val_data" + underscore + dialog_type + kb + ".csv", encoding="ISO-8859-1", delimiter=',')
    if use_original:
        dialog_type = old_dialog_type
        inputs = list(df["inputs"])
        outputs = list(df["outputs"])
    else:
        inputs = list(df["input"])
        outputs = list(df["output"])
    vocab = Vocabulary('data/vocabulary-train' + dialog_type + '.json', padding=pad_length)

    kb_vocabulary = Vocabulary('data/vocabulary-train' + dialog_type + '.json', padding=4)

    model = KVMMModel(pad_length=20,
                      embedding_size=200,
                      batch_size=1,
                      vocab_size=vocab.size(),
                      n_chars=vocab.size(),
                      n_labels=vocab.size(),
                      encoder_units=200,
                      decoder_units=200).to(device)
    weights_file = "weights/model_weights_original_iter_500000.pytorch"
    model.load_state_dict(torch.load(weights_file, map_location='cpu'))

    kbfile = "data/normalised_kbtuples.csv"
    df = pd.read_csv(kbfile)
    kbs = list(df["subject"] + "_" + df["relation"])
    kbs = [re.sub(" ", "_", kb) for kb in kbs]
    # print(kbs[:3])
    kbs = np.array(list(map(kb_vocabulary.string_to_int, kbs)))
    # TODO: Fix kb here
    kbs = np.repeat(kbs[np.newaxis, :, :], 1, axis=0)
    data = run_examples(model, kbs, vocab, inputs, outputs)
    df = pd.DataFrame(columns=["inputs", "outputs", "prediction"])
    d = {'outputs': [], 'inputs': [], 'predictions': []}
    for i, o, p in zip(inputs, outputs, data):
        d["outputs"].append(str(o))
        d["inputs"].append(str(i))
        d["predictions"].append(str(p))
    df = pd.DataFrame(d)
    df.to_csv("output_kb" + dialog_type + ".csv", encoding="ISO-8859-1", sep=";")
    # print(outputs)
