# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import argparse
import tensorflow as tf
import time
import io
import random

parser = argparse.ArgumentParser(
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--rnn_size', type=int, default=2048,
                    help='Size of RNN hidden states')
parser.add_argument('--embedding_size', type=int, default=128,
                    help='Character embedding layer size')
parser.add_argument('--restore_path', type= str, default=None,
                    help='Path to a directory from which to restore a model from previous session')
parser.add_argument('--num_chars', type=int, default=100,
                    help='Option to specify how many chars to sample from the model')
parser.add_argument('--wn', type=int, default=1,
                    help='Switch for weight normalisation on the mLSTM parameters. Integer argument of 1 for ON and 0 for OFF')
parser.add_argument('--prime', type=str, default=None,
                    help='Prime the network with some bytes')
parser.add_argument('--save_samples', type=str, default=None,
                    help='Directory to save the generated samples')
parser.add_argument('--restore_weights', type=str, default=None,
                    help='Directory to save the generated samples')
parser.add_argument('--vocab_size', type=str, default=256,
                    help='Input size')
parser.add_argument('--sentiment_neuron_value', type=int, default=4,
                    help='Fix value of the sentiment Neuron')
parser.add_argument('--sentiment_neuron_index', type=int, default=3984,
                    help='Specify the index of the Sentiment Neuron')

args = parser.parse_args()

rnn_size = args.rnn_size
embedding_size = args.embedding_size
restore_path = args.restore_path
prime = args.prime
vocabulary_size = args.vocab_size # byte
sentiment_neuron_index = args.sentiment_neuron_index
sentiment_neuron_value = args.sentiment_neuron_value

weights_list = np.load(os.path.join(restore_path,'model.npy'), allow_pickle=True)

graph = tf.Graph()

with graph.as_default():

    # define all of the model variables
    W_embedding = tf.get_variable('W_embedding',initializer=tf.constant(weights_list[0]))

    # mt = (Wmxxt) ⊙ (Wmhht−1) - equation 18
    Wmx = tf.get_variable('Wmx', initializer=tf.constant(weights_list[1]))
    Wmh = tf.get_variable('Wmh', initializer=tf.constant(weights_list[2]))

    # hˆt = Whxxt + Whmmt
    Whx = tf.get_variable('Whx', initializer=tf.constant(weights_list[3]))
    Whm = tf.get_variable('Whm', initializer=tf.constant(weights_list[4]))
    Whb = tf.get_variable('Whb', initializer=tf.constant(weights_list[5]))

    # it = σ(Wixxt + Wimmt)
    Wix = tf.get_variable('Wix', initializer=tf.constant(weights_list[6]))
    Wim = tf.get_variable('Wim', initializer=tf.constant(weights_list[7]))
    Wib = tf.get_variable('Wib', initializer=tf.constant(weights_list[8]))

    # ot = σ(Woxxt + Wommt)
    Wox = tf.get_variable('Wox', initializer=tf.constant(weights_list[9]))
    Wom = tf.get_variable('Wom', initializer=tf.constant(weights_list[10]))
    Wob = tf.get_variable('Wob', initializer=tf.constant(weights_list[11]))

    # ft =σ(Wfxxt +Wfmmt)
    Wfx = tf.get_variable('Wfx', initializer=tf.constant(weights_list[12]))
    Wfm = tf.get_variable('Wfm', initializer=tf.constant(weights_list[13]))
    Wfb = tf.get_variable('Wfb', initializer=tf.constant(weights_list[14]))

    # define the g parameters for weight normalization if wn switch is on
    if args.wn == 1:

        gmx = tf.get_variable('gmx', initializer=tf.constant(weights_list[15]))
        gmh = tf.get_variable('gmh', initializer=tf.constant(weights_list[16]))

        ghx = tf.get_variable('ghx', initializer=tf.constant(weights_list[17]))
        ghm = tf.get_variable('ghm', initializer=tf.constant(weights_list[18]))

        gix = tf.get_variable('gix', initializer=tf.constant(weights_list[19]))
        gim = tf.get_variable('gim', initializer=tf.constant(weights_list[20]))

        gox = tf.get_variable('gox', initializer=tf.constant(weights_list[21]))
        gom = tf.get_variable('gom', initializer=tf.constant(weights_list[22]))

        gfx = tf.get_variable('gfx',  initializer=tf.constant(weights_list[23]))
        gfm = tf.get_variable('gfm', initializer=tf.constant(weights_list[24]))


        # normalized weights
        Wmx = tf.nn.l2_normalize(Wmx, dim=0)*gmx
        Wmh = tf.nn.l2_normalize(Wmh, dim=0)*gmh

        Whx = tf.nn.l2_normalize(Whx,dim=0)*ghx
        Whm = tf.nn.l2_normalize(Whm,dim=0)*ghm

        Wix = tf.nn.l2_normalize(Wix,dim=0)*gix
        Wim = tf.nn.l2_normalize(Wim,dim=0)*gim

        Wox = tf.nn.l2_normalize(Wox,dim=0)*gox
        Wom = tf.nn.l2_normalize(Wom,dim=0)*gom

        Wfx = tf.nn.l2_normalize(Wfx,dim=0)*gfx
        Wfm = tf.nn.l2_normalize(Wfm,dim=0)*gfm

    # classifier weights and biases.
    w = tf.get_variable('Classifier_w', initializer=tf.constant(weights_list[25]))
    b = tf.get_variable('Classifier_b', initializer=tf.constant(weights_list[26]))


    def mlstm_cell(x, h, c):
        """
        multiplicative LSTM cell. https://arxiv.org/pdf/1609.07959.pdf

        """
        # mt = (Wmxxt) ⊙ (Wmhht−1) - equation 18
        mt = tf.matmul(x,Wmx) * tf.matmul(h,Wmh)
        # hˆt = Whxxt + Whmmt
        ht = tf.tanh(tf.matmul(x,Whx) + tf.matmul(mt,Whm) + Whb)
        # it = σ(Wixxt + Wimmt)
        it = tf.sigmoid(tf.matmul(x,Wix) + tf.matmul(mt,Wim)+ Wib)
        # ot = σ(Woxxt + Wommt)
        ot = tf.sigmoid(tf.matmul(x,Wox) + tf.matmul(mt,Wom)+ Wob)
        # ft =σ(Wfxxt +Wfmmt)
        ft = tf.sigmoid(tf.matmul(x,Wfx) + tf.matmul(mt,Wfm)+ Wfb)

        c_new = (ft * c) + (it * ht)

        h_new = tf.tanh(c_new) * ot

        return h_new, c_new

    # Sampling code.
    sample_input = tf.placeholder(tf.int32, shape=(1,), name = 'sample_input')
    sample_embedding= tf.nn.embedding_lookup(W_embedding,sample_input)
    saved_sample_output = tf.Variable(tf.zeros([1, rnn_size]), name = 'saved_sample_output')
    saved_sample_state = tf.Variable(tf.zeros([1, rnn_size]), name = 'saved_sample_state')

    reset_sample_state = tf.group(saved_sample_output.assign(tf.zeros([1, rnn_size])), saved_sample_state.assign(tf.zeros([1,           rnn_size])),name='reset_sample_state_op')

    sample_state_var= tf.Variable(saved_sample_state,name='sample_state_var')

    sample_output, sample_state = mlstm_cell(sample_embedding, saved_sample_output, sample_state_var)



    with tf.control_dependencies([saved_sample_output.assign(sample_output),sample_state_var.assign(sample_state)]):

        # fix the value of the sentiment neuron
        if sentiment_neuron_index is not None:
            sample_state_var = tf.assign(sample_state_var[0, sentiment_neuron_index], sentiment_neuron_value)

        sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b), name = 'sample_prediction')

with tf.Session(graph=graph) as session:

        tf.global_variables_initializer().run()
        print('Variables Initialized')

        print('Sampling...')
        start = time.time()

        print('='*100)

        sentence = bytearray()

        if args.prime is not None:

            # prime the network with a sequence of bytes
            prime = bytearray(args.prime)
            sentence += prime
            for i in prime:
                feed = np.array(i, ndmin=1)
                prediction = session.run(sample_prediction, feed_dict = {sample_input: feed})

        else:

            # prime with a random byte
            feed = np.array(random.sample(range(vocabulary_size),1), dtype='int32')
            prediction = session.run(sample_prediction, feed_dict = {sample_input: feed})



        for _ in range(args.num_chars):

            # sequence is generated here
            feed = np.expand_dims(np.random.choice(range(vocabulary_size), p=prediction.ravel()),axis=0)
            sentence.append(int(feed))
            prediction, ssv = session.run([sample_prediction, sample_state_var],feed_dict = {sample_input: feed})

        # decode the bytes to get unicode representation
        sentence = sentence.decode('utf-8', errors='replace')

        print(sentence)
        end = time.time()

        print('='*100)
        print('Sampling time = ', end - start)


        if args.save_samples is not None:

            if not os.path.exists(args.save_samples):
                os.makedirs(args.save_samples)

            sample_file = os.path.join(args.save_samples,'samples')

            with io.open(sample_file, 'a+', encoding='utf-8') as f:
                f.write(sentence)
