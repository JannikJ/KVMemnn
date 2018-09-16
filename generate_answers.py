import argparse

from model.memnn import memnn
from reader import Vocabulary


def load_test_data():
    dialogs = []
    return dialogs


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
    vocab = Vocabulary('./data/vocabulary.json',
                              padding=args.padding)
    model = memnn(pad_length=args.padding,
                      embedding_size=args.embedding,
                      vocab_size=vocab.size(),
                      batch_size=args.batch_size,
                      n_chars=vocab.size(),
                      n_labels=vocab.size(),
                      embedding_learnable=True,
                      encoder_units=200,
                      decoder_units=200,trainable=True)
    model.load_weights("model_weights_nkbb.hdf5")

    test_data = load_test_data()
    model.predict(test_data)  # ("How is the weather in San Francisco today?")
