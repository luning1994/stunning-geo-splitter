from sys import version_info
if version_info[0] < 3:
    import cPickle
else:
    import _pickle as cPickle
import numpy as np
import nltk
import itertools
import sys
from random import shuffle
from classes import RNN, GRU
import config
from config import *
import nltk


# def softmax(x):
#     xt = np.exp(x - np.max(x))
#     return xt / np.sum(xt)

def save_model_parameters_rnn(outfile, model):
    U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
    try:
        np.savez(outfile, U=U, V=V, W=W)
        print ("Saved model parameters to %s." % outfile)
    except IOError as e:
        print('Cannot write to file!', e.errno)
    except:
        print('Other unexpected error')

    
def load_model_parameters_rnn(path, model):
    try:
        npzfile = np.load(path)
        U, V, W = npzfile["U"], npzfile["V"], npzfile["W"]
        model.hidden_dim = U.shape[0]
        model.word_dim = U.shape[1]
        model.U.set_value(U)
        model.V.set_value(V)
        model.W.set_value(W)
        print ("Loaded model parameters from {}. hidden_dim={} word_dim={}".format(
            path, U.shape[0], U.shape[1]))
    except IOError as e:
        print('File not found!', e.errno)
    except:
        print('Other unexpected error')

def save_model_parameters_gru(outfile, model):
    try:
        np.savez(outfile,
            E=model.E.get_value(),
            U=model.U.get_value(),
            W=model.W.get_value(),
            V=model.V.get_value(),
            b=model.b.get_value(),
            c=model.c.get_value())
        print("Saved model parameters to %s." % outfile)
    except IOError as e:
        print('Cannot write to file!', e.errno)
    except:
        print('Other unexpected error')

def load_model_parameters_gru(path, model):
    try:
        npzfile = np.load(path)
        E, U, W, V, b, c = npzfile["E"], npzfile["U"], npzfile["W"], npzfile["V"], npzfile["b"], npzfile["c"]
        hidden_dim, word_dim = E.shape[0], E.shape[1]
        print ("Building model model from %s with hidden_dim=%d word_dim=%d" % (
            path, hidden_dim, word_dim))
        sys.stdout.flush()
        model.hidden_dim = hidden_dim
        model.word_dim = word_dim
        model.E.set_value(E)
        model.U.set_value(U)
        model.W.set_value(W)
        model.V.set_value(V)
        model.b.set_value(b)
        model.c.set_value(c)
        return model
    except IOError as e:
        print('File not found!', e.errno)
    except:
        print('Other unexpected error')


def create_training_dataset(data_dir):
    # TODO => cross validation

    # read vietnam name and addr file then concatenate them at random
    # add in token for splitting
    print("loading data...")
    try:
        names = list(open(r'data\com.txt', 'r').readlines())
        addrs = list(open(r'data\add.txt', 'r').readlines())
        print("data loaded!")
        shuffle(names)
        shuffle(addrs)
        sentences = ["%s %s %s %s %s" % (SENT_START_TOKEN, name, SPLIT_TOKEN, addr, SENT_END_TOKEN)
                     for name, addr in zip(names, addrs)]
        print("number of sentences: %d" % len(sentences))

        # Tokenize the sentences into words
        tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
        print("Tokenized sentences {}".format(tokenized_sentences))
        # Count the word frequencies
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print("vocab size: %d " % len(word_freq.items()))

        # cut out the least 10 frequent items as UNK to ensure generalization
        vocab = word_freq.most_common(len(word_freq.items()) - 10)
        idx2word = [word for word, freq in vocab]
        idx2word.append(UNKNOWN_TOKEN)
        word2idx = {word: idx for idx, word in enumerate(idx2word)}

        # print("vocabulary size upper limit: %d." % VOCABULARY_SIZE_LIMIT)
        print("least frequent word: %s, freq: %d" % (vocab[-1][0], vocab[-1][1]))

        # Replace all words not in our vocabulary with the unknown token
        for idx, sent in enumerate(tokenized_sentences):
            tokenized_sentences[idx] = [word if word in word2idx else UNKNOWN_TOKEN for word in sent]

        # Create the training data
        X_train = np.asarray([[word2idx[word] for word in sent[:-1]] for sent in tokenized_sentences])
        y_train = np.asarray([[word2idx[word] for word in sent[1:]] for sent in tokenized_sentences])
        print(X_train)
        print(y_train)
        # uses actual vocab size if vocab is within limit
        vocab_size = min(len(word_freq.items()), VOCABULARY_SIZE_LIMIT)

        cPickle.dump([X_train, y_train, vocab_size, word2idx], open(data_dir, "wb"))

        print("dataset %s created" % data_dir)
    except IOError as e:
        print('Cannot find/write to file', e.errno)
    except EOFError as e:
        print('File empty!', e.errno)
    except:
        print('Other unexpected error')
        raise

'''
def load_trained_model(data_dir, model_dir):
    print("loading model %s..." % model_dir)
    try:
        X_train, y_train, vocab_size, word2idx = cPickle.load(open(data_dir, "rb"))
        model = RNN(vocab_size, hidden_dim=config.HIDDEN_DIM)
        load_model_parameters_theano(model_dir, model)
        print("model loaded!")
        return model, word2idx
    except IOError as e:
        print('File not found!', e.errno)
    except EOFError as e:
        print('File empty!', e.errno)
    except:
        print('Other unexpected error')
'''
if __name__ == "__main__":
    create_training_dataset("data\data.p")
