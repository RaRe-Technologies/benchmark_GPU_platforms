#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Helping functions for loading twitter sentiments data.
Most of the code is directly borrowed from https://github.com/danielegrattarola/twitter-sentiment-cnn.

"""

import numpy as np
import re
import random, csv
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

POS_DATASET_PATH = 'data/tw-data.pos'
NEG_DATASET_PATH = 'data/tw-data.neg'
VOC_PATH = 'data/vocab.csv'
VOC_INV_PATH = 'data/vocab_inv.csv'


def clean_str(string):
    """Tokenizes common abbreviations and punctuation, removes unwanted characters.
    Returns the clean string.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r'(.)\1+', r'\1\1', string) 
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(dataset_fraction):
    """Loads data from files, processes the data and creates two lists, one of
    strings and one of labels.
    Returns the lists. 
    """
    logger.info("\tdata_helpers: loading positive examples...")
    positive_examples = list(open(POS_DATASET_PATH).readlines())
    positive_examples = [s.strip() for s in positive_examples]
    logger.info("\tdata_helpers: [OK]")
    logger.info("\tdata_helpers: loading negative examples...")
    negative_examples = list(open(NEG_DATASET_PATH).readlines())
    negative_examples = [s.strip() for s in negative_examples]
    logger.info("\tdata_helpers: [OK]")

    # for development, removed later.
    positive_examples = sample_list(positive_examples, dataset_fraction)
    negative_examples = sample_list(negative_examples, dataset_fraction)

    # Split by words
    x_text = positive_examples + negative_examples
    logger.info("\tdata_helpers: cleaning strings...")
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    logger.info("\tdata_helpers: [OK]")

    # Generate labels
    logger.info("\tdata_helpers: generating labels...")
    positive_labels = [1 for _ in positive_examples]
    negative_labels = [0 for _ in negative_examples]
    logger.info("\tdata_helpers: [OK]")
    logger.info("\tdata_helpers: concatenating labels...")
    y = np.concatenate([positive_labels, negative_labels], 0)
    logger.info("\tdata_helpers: [OK]")
    return [x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """Pads all sentences to the same length. The length is defined by the longest
    sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab():
    """Reads the vocabulary and its inverse mapping from the csv in the dataset
    folder.
    Returns a list with the vocabulary and the inverse mapping.
    """
    voc = csv.reader(open(VOC_PATH))
    voc_inv = csv.reader(open(VOC_INV_PATH))
    # Mapping from index to word
    vocabulary_inv = [x for x in voc_inv]
    # Mapping from word to index
    vocabulary = {x: i for x, i in voc}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """Maps sentencs and labels to vectors based on a vocabulary.
    Returns the mapped lists. 
    """
    x = np.array([[vocabulary[word] for word in sentence]
                  for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_all_data(dataset_fraction):
    """Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels(dataset_fraction)
    logger.info("\tdata_helpers: padding strings...")
    sentences_padded = pad_sentences(sentences)
    logger.info("\tdata_helpers: [OK]")
    logger.info("\tdata_helpers: building vocabulary...")
    vocabulary, vocabulary_inv = build_vocab()
    logger.info("\tdata_helpers: [OK]")
    logger.info("\tdata_helpers: building processed datasets...")
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    logger.info("\tdata_helpers: [OK]")
    return [x, y, vocabulary, vocabulary_inv]


def load_data(dataset_fraction):
    """Loads, preprocesses, shuffle and splits data.
    Returns final train and test vectors and labels, vocabulary, and inverse vocabulary.
    """
    x, y, vocabulary, vocabulary_inv = load_all_data(dataset_fraction)

    # Randomly shuffle data

    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    text_percent = 0.1
    test_index = int(len(x) * text_percent)
    x_train, x_test = x_shuffled[:-test_index], x_shuffled[-test_index:]
    y_train, y_test = y_shuffled[:-test_index], y_shuffled[-test_index:]

    return x_train, y_train, x_test, y_test, vocabulary, vocabulary_inv

# for development, removed later.
def sample_list(list, fraction):
    """Returns 1/dividend-th of the given list, randomply sampled.
    """
    return random.sample(list, int(len(list) * fraction))
