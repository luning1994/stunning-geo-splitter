#! /usr/bin/env python
from sys import version_info
if version_info[0] < 3:
    import cPickle
else:
    import _pickle as cPickle
import itertools
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
from process_data import save_model_parameters_gru, load_model_parameters_gru
from classes import RNN, GRU
from config import *
import sklearn
from sklearn.utils import shuffle



def train_rnn(model, X_train, y_train, learning_rate=0.005, nepoch=1, eval_every=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Shuffle the training data
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        # Optionally evaluate the loss
        if (epoch % eval_every == 0):
            print('------------------EPOCH %d--------------------' % epoch)
            loss = eval_model(model, num_examples_seen, X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learning_rate = learning_rate * 0.5
                print("Setting learning rate to %f" % learning_rate)
            sys.stdout.flush()
            # Saving model parameters
            try:
                save_model_parameters_gru("./data/rnn-epoch%d.npz" % epoch, model)
            except IOError as e:
                print('Cannot write to file!', e.errno)
            except:
                print('Other unexpected error')
            
        # Training
        for i in range(len(y_train)):
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
            
                    
def train_gru(model, X_train, y_train, learning_rate=0.001, nepoch=20, eval_every=5):
    """
    learning rate schdule no longer needed; using RMSProp/Adadelta for training
    """
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Shuffle the training data
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        if (epoch % eval_every == 0):
            print('------------------EPOCH %d--------------------' % epoch)
            eval_model(model, num_examples_seen, X_train, y_train)
            print('saving')
            save_model_parameters_gru("./data/gru-epoch%d.npz" % epoch, model)

        # Training
        for i in range(len(y_train)):
            print(i)
            model.sgd_step_adadelta(X_train[i], y_train[i])
            num_examples_seen += 1

            
def eval_model(model, num_examples_seen, X_train, y_train):
    """
    evaluate model using the last 10% of training data
    """
    dt = datetime.now().isoformat()
    print('calcloss')
    loss = model.calculate_loss(X_train, y_train)
    print('calclossdone')
    num_train = int(np.round(len(y_train) * 0.9))
    total_words = 0
    correct = 0
    print('evaluating')
    for x, y in zip(X_train[num_train:], y_train[num_train:]):
        print('rep')
        correct += sum(np.equal(model.pred_class(x), y))
        total_words += len(x)
    accuracy = correct * 1.0 / total_words
    print("%s| num_examples_seen:%d" % (dt, num_examples_seen))
    print("Loss: %f" % loss)
    print("Total Accuracy: %f" % accuracy)
    return loss

if __name__ == "__main__":
    print("loading data...")
    try:
        X_train, y_train, vocab_size, word2idx = cPickle.load(open("data\data.p", "rb"))
        print("data loaded!")

        print("initializing model...")
        # model = RNN(vocab_size, hidden_dim=HIDDEN_DIM)
        model = GRU(vocab_size)
        print("initialization finished!")

        # print("benchmarking training time...")
        # t1 = time.time()
        # 265ms per entry
        # model.sgd_step(X_train[10], y_train[10], LEARNING_RATE)
        # 280ms per entry
        # model.sgd_step_adadelta(np.asarray(X_train[10], 'int32'),
        #                         np.asarray(y_train[10], 'int32'))
        # t2 = time.time()
        # print("SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.))

        # load_model_parameters_rnn("data\\rnn-theano-80-3241-2016-09-01-17-08-54.npz", model)

        # train_rnn(model, X_train, y_train, nepoch=NEPOCH, learning_rate=LEARNING_RATE)
        train_gru(model, X_train, y_train, nepoch=NEPOCH, learning_rate=LEARNING_RATE)
    except IOError as e:
        print('File not found!', e.errno)
    except EOFError as e:
        print('File empty!', e.errno)
    except:
        print('Other unexpected error')
