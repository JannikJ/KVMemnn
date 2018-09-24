"""
    Runs a simple Neural Machine Translation model
    Type `python run.py -h` for help with arguments.
"""
import os
import argparse
import numpy as np
import os
import keras
from keras.layers import Lambda
#from keras import backend as k
from keras.models import Model
from keras.layers import Dense, Embedding, Activation, Permute
from keras import regularizers, constraints, initializers, activations
from keras.layers import Input, Flatten, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.callbacks import ModelCheckpoint, Callback
from reader import Data,Vocabulary
from model.memnn import memnn


# create a directory if it doesn't already exist
if not os.path.exists('./weights'):
    os.makedirs('./weights/')


class TestCallback(Callback):
    best_loss = 100
    best_acc = 0.0
    training_file_name = ""

    def __init__(self, test_data, training_file_name):
        # Callback.__init__(self)
        self.test_data = test_data
        self.training_file_name = training_file_name

    def on_epoch_end(self, epoch, logs={}):
        # if len(self.test_data) == 2:
        #     x, y = self.test_data
        # elif len(self.test_data) == 3:
        #     x, y, sample_weight = self.test_data
        # else:
        #     print("ERROR: Expected 2 or 3 values packed in test_data but got " + str(len(self.test_data)))
        # x, y = self.test_data.send(None)
        # loss, acc = self.model.evaluate(x, y, verbose=0)
        saved = False
        # if loss < self.best_loss and acc > self.best_acc:
        #     self.model.save_weights("model_weights_nkbb-" + self.training_file_name + "-epoch-" + str(epoch) + "-with-best-loss-and-accuracy.hdf5")
        #     self.best_loss = loss
        #     self.best_acc = acc
        #     saved = True
        #     print("BEST LOSS YET: " + str(loss))
        #     print("BEST ACCURACY YET: " + str(acc))
        # else:
        #     if loss < self.best_loss:
        #         self.model.save_weights("model_weights_nkbb-" + self.training_file_name + "-epoch-" + str(epoch) + "-with-best-loss.hdf5")
        #         self.best_loss = loss
        #         saved = True
        #         print("BEST LOSS YET: " + str(loss))
        #     if acc > self.best_acc:
        #         self.model.save_weights("model_weights_nkbb-" + self.training_file_name + "-epoch-" + str(epoch) + "-with-best-accuracy.hdf5")
        #         self.best_acc = acc
        #         saved = True
        #         print("BEST ACCURACY YET: " + str(acc))
        if epoch % 20 == 0 and not saved:
            self.model.save_weights("model_weights_nkbb-" + self.training_file_name[14:21] + "-epoch-" + str(epoch) + ".hdf5")

        # self.model.save_weights("model_weights_nkbb-epoch-" + str(epoch) + "-with-loss-" + str(loss) + "-and-accuracy-"
        #                         + str(acc) + "-.hdf5")


def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # Dataset functions
    vocab = Vocabulary('./data/vocabulary.json', padding=args.padding)
    vocab = Vocabulary('./data/vocabulary.json',
                              padding=args.padding)
    kb_vocab=Vocabulary('./data/vocabulary.json',
                              padding=4) # 7  # 4
    print('Loading datasets.')
    training = Data(args.training_data, vocab,kb_vocab)
    validation = Data(args.validation_data, vocab, kb_vocab)
    training.load()
    validation.load()
    training.transform()
    training.kb_out()
    validation.transform()
    validation.kb_out()
    print('Datasets Loaded.')
    print('Compiling Model.')

    model = memnn(pad_length=args.padding,
                  embedding_size=args.embedding,
                  vocab_size=vocab.size(),
                  batch_size=args.batch_size,
                  n_chars=vocab.size(),
                  n_labels=vocab.size(),
                  embedding_learnable=True,
                  encoder_units=200,
                  decoder_units=200,trainable=True)

    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', ])
    print('Model Compiled.')
    print('Training. Ctrl+C to end early.')

    try:
        model.fit_generator(generator=training.generator(args.batch_size),
                            steps_per_epoch=300,
                            validation_data=validation.generator(args.batch_size),
                            validation_steps=10,
                            workers=1,
                            verbose=1,
                            epochs=args.epochs,
                            callbacks=[TestCallback(validation.generator(args.batch_size), args.training_data)])

    except KeyboardInterrupt as e:
        print('Model training stopped early.')
    model.save_weights("model_weights_nkbb.hdf5")

    print('Model training complete.')

    #run_examples(model, input_vocab, output_vocab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    named_args = parser.add_argument_group('named arguments')

    named_args.add_argument('-e', '--epochs', metavar='|',
                            help="""Number of Epochs to Run""",
                            required=False, default=130, type=int)
    named_args.add_argument('-es', '--embedding', metavar='|',
                            help="""Size of the embedding""",
                            required=False, default=200, type=int)

    named_args.add_argument('-g', '--gpu', metavar='|',
                            help="""GPU to use""",
                            required=False, default='1', type=str)

    named_args.add_argument('-p', '--padding', metavar='|',
                            help="""Amount of padding to use""",
                            required=False, default=20, type=int)

    named_args.add_argument('-t', '--training-data', metavar='|',
                            help="""Location of training data""",
                            required=False, default='./data/train_data.csv')

    named_args.add_argument('-v', '--validation-data', metavar='|',
                            help="""Location of validation data""",
                            required=False, default='./data/val_data.csv')

    named_args.add_argument('-b', '--batch-size', metavar='|',
                            help="""Location of validation data""",
                            required=False, default=100, type=int)
    args = parser.parse_args()
    print(args)
    main(args)
