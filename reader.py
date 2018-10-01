import json
import csv
import random
import operator
import io
import re

import numpy as np
import pandas as pd


random.seed(1984)

INPUT_PADDING = 50
OUTPUT_PADDING = 100


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

class Vocabulary(object):

    def __init__(self, vocabulary_file, padding=None):
        """
            Creates a vocabulary from a file
            :param vocabulary_file: the path to the vocabulary
        """
        self.vocabulary_file = vocabulary_file
        with io.open(vocabulary_file, 'r',encoding='utf-8') as f:
            self.vocabulary = json.load(f)

        self.padding = padding
        self.reverse_vocabulary = {v: k for k, v in self.vocabulary.items()}

    def size(self):
        """
            Gets the size of the vocabulary
        """
        return len(self.vocabulary.keys())

    def string_to_int(self, text):
        """
            Converts a string into it's character integer 
            representation
            :param text: text to convert
        """
        text = re.sub("\t", " ", text)
        text = re.sub(",", " ", text)
        text = re.sub("addressnue", "address", text)
        text = re.sub("addressive", "address", text)
        tokens = text.split(" ")
        tokens = [x for x in tokens if x.strip() != ""]
        #print(tokens)
        integers = []

        for index, token in enumerate(tokens):
            if token == "p.f.changs" or token == "p.f.changs?"\
                    or token == "p.f.changs!"\
                    or token == "p.f.changs."\
                    or token == "p.f.changs,"\
                    or token == "p.f.changs:"\
                    or token == "p.f.changs;":
                tokens[index] = "p"
                tokens.insert(index + 1, "f")
                tokens.insert(index + 2, "changs")
            elif token == "p.f.":
                tokens[index] = "p"
                tokens.insert(index + 1, "f")
            elif (token[len(token) - 1] == "?" and token != "?")\
                    or (token[len(token) - 1] == "!" and token != "!")\
                    or (token[len(token) - 1] == "." and token != ".")\
                    or (token[len(token) - 1] == "," and token != ",")\
                    or (token[len(token) - 1] == ":" and token != ":")\
                    or (token[len(token) - 1] == ";" and token != ";"):
                tokens[index] = token[0:len(token) - 1]
                # tokens.insert(index + 1, token[len(token) - 1])
        tokens = [t for t in tokens if t != "?" and t != "!" and t != "." and t != "," and t != ":" and t != ";"]

        if self.padding and len(tokens) >= self.padding:
            # truncate if too long
            tokens = tokens[-(self.padding - 1):]

        tokens.append('<eos>')

        for c in tokens:
            if c.strip(",").strip(".").strip(":").strip("!").strip("?").strip(",").strip("/").strip('"').strip('(').strip(')').strip('[').strip(']') in self.vocabulary:
                integers.append(self.vocabulary[c.strip(",").strip(".").strip(":").strip("!").strip("?").strip(",").strip("/").strip('"').strip('(').strip(')').strip('[').strip(']')])
            else:
                integers.append(self.vocabulary['<unk>'])


        # pad:
        if self.padding and len(integers) < self.padding:
            integers.reverse()
            integers.extend([self.vocabulary['<pad>']]
                            * (self.padding - len(integers)))
            integers.reverse()

        if len(integers) != self.padding:
            print(text)
            raise AttributeError('Length of text was not padding.')
        return integers

    def int_to_string(self, integers):
        """
            Decodes a list of integers
            into it's string representation
        """
        tokens = []
        for i in integers:
            tokens.append(self.reverse_vocabulary[i])

        return tokens


class Data(object):

    def __init__(self, file_name, vocabulary,kb_vocabulary):
        """
            Creates an object that gets data from a file
            :param file_name: name of the file to read from
            :param vocabulary: the Vocabulary object to use
            :param batch_size: the number of datapoints to return
            :param padding: the amount of padding to apply to 
                            a short string
        """

        self.input_vocabulary = vocabulary
        self.output_vocabulary = vocabulary
        self.kb_vocabulary=kb_vocabulary
        self.kbfile = "./data/normalised_kbtuples.csv"
        self.file_name = file_name
    def kb_out(self):
        df=pd.read_csv(self.kbfile, encoding="ISO-8859-1", sep=',')
        self.kbs=list(df["subject"]+" "+df["relation"])
        # self.kbs = np.array(list(
        #     map(self.kb_vocabulary.string_to_int, self.kbs)))
        # TODO: See if it gets better when kbs are not three words but rather one word concatenated with "_"
        self.kbs = np.array(list(
            map(self.kb_vocabulary.string_to_int, [re.sub("p.f.", "p. f.",
                                                          re.sub("traffic_info", "traffic info", kb.lower()))
                                                   for kb in self.kbs])))


    def load(self):
        """
            Loads data from a file
        """
        self.inputs = []
        self.targets = []

        with io.open(self.file_name, 'r', encoding="ISO-8859-1") as f:
            reader = csv.reader(f, delimiter=',')
            for index, row in enumerate(reader):
                #print(row)
                #print(row[1],row[2])
                if not(index == 0 and (row[0] == "inputs" or row[0] == "input")):
                    self.inputs.append(row[0])
                    self.targets.append(row[1])


    def transform(self):
        """
            Transforms the data as necessary
        """
        # @TODO: use `pool.map_async` here?
        self.inputs = np.array(list(
            map(self.input_vocabulary.string_to_int, self.inputs)))
        self.targets = np.array(list(map(self.output_vocabulary.string_to_int, self.targets)))
        # self.inputs = np.array(list(
        #     map(self.input_vocabulary.string_to_int, [dialog.lower() for dialog in self.inputs])))
        # self.targets = np.array(list(map(self.output_vocabulary.string_to_int,
        #                                  [dialog.lower() for dialog in self.targets])))


    def generator(self, batch_size):
        """
            Creates a generator that can be used in `model.fit_generator()`
            Batches are generated randomly.
            :param batch_size: the number of instances to include per batch
        """
        instance_id = range(len(self.inputs))
        while True:
            try:
                batch_ids = random.sample(instance_id, batch_size)
                targets=np.array(self.targets[batch_ids])
                targets = np.array(list(map(lambda x: to_categorical(x,num_classes=self.output_vocabulary.size()),targets)))
                return ([np.array(self.inputs[batch_ids], dtype=int),np.repeat(self.kbs[np.newaxis,:,:],batch_size,axis=0)],np.array(targets))
            except Exception as e:
                print('EXCEPTION OMG')
                print(e)
                return None, None,None

if __name__ == '__main__':
    vocab = Vocabulary('./data/vocabulary.json', padding=20)
    kb_vocabulary = Vocabulary('./data/vocabulary.json', padding=4)
    print(vocab.string_to_int("find the address to a hospital or clinic. hospital#poi is at Stanford_Express_Care#address. thank you."))
    ds = Data('./data/train_data.csv', vocab,kb_vocabulary)
    ds.kb_out()
    g = ds.generator(32)
    ds.load()
    ds.transform()
    #print(vocab.string_to_int("find starbucks <eos>"))
    for i in range(50):
         print(next(g)[0][0][0],next(g)[1][0])
         print(vocab.int_to_string(list(next(g)[0][0][0])))
         break
