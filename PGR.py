# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Sat May 23 00:48:10 2020

@author: Administrator
"""

'''
Some of the following functions have been adapted from the code on the repository:
https://github.com/UKPLab/deeplearning4nlp-tutorial
'''
from model import myModel
import os
import re
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
filestrain = r'data/pgr/train_12.tsv'
filestest = r'data/pgr/clean_test.tsv'
def sentCleaner(sent):
    sent = re.sub(r"\.$", "", sent)
    sent = re.sub(r"-", " - ", sent)
    sent = re.sub(r"\.\s", " . ", sent)
    sent = re.sub(r",", " , ", sent)
    sent = re.sub(r"<", " ( ", sent)
    sent = re.sub(r">", " ( ", sent)
    sent = re.sub(r"\(", " ( ", sent)
    sent = re.sub(r"\)", " ) ", sent)
    sent = re.sub(r"/", " : ", sent)
    sent = re.sub(r":", " : ", sent)
    sent = re.sub(r";", " : ", sent)
    sent = re.sub(r"%", " % ", sent)
    sent = re.sub(r"\[", " [ ", sent)
    sent = re.sub(r"\]", " ] ", sent)
    sent = re.sub(r"=", " = ", sent)
    return sent
def getVocab(matrix, ndim=2, isClean=False, isUnknow=False):
    if isClean is True:
        matrix = sentClean(matrix)

    counter = Counter()
    if ndim == 1:
        for token in matrix:
            try:
                counter[token] += 1
            except KeyError as KE:
                print("vocab1Derror:", KE)
                exit()
    elif ndim == 2:
        for line in matrix:
            for token in line:
                try:
                    counter[token] += 1
                except KeyError as KE:
                    print("vocab2Derror:", KE)
                    exit()
    elif ndim == 3:
        for line in matrix:
            for token in line:
                for char in token:
                    try:
                        counter[str(char)] += 1
                    except KeyError as KE:
                        print("vocab2Derror:", KE)
                        exit()
    counter = sorted(counter.most_common(), key=lambda x: x[1], reverse=True)
    if isUnknow:
        vocab = {str(v[0]): i + 2 for i, v in enumerate(counter)}
        vocab["<unknow>"] = 1
        return vocab  # 0 as padding,1 as unknow
    return {str(v[0]): i + 1 for i, v in enumerate(counter)}  # 0 as padding,1 as unknow
def oneHot2D(matrix, vocab=None, returnvocab=False):
    '''
    one-hot encoding for word / pos
    '''

    # creat vocab
    if vocab is None:
        counter = Counter()
        for line in matrix:
            for token in line:
                try:
                    counter[token] += 1
                except KeyError as KE:
                    pass
        counter = sorted(counter.most_common(), key=lambda x: x[1], reverse=True)
        vocab = {v[0]: i + 1 for i, v in enumerate(counter)}

    # replace token to token-id
    matrixTags = []
    for l, line in enumerate(matrix):
        lineTag = []
        for t, token in enumerate(line):
            try:
                lineTag.append(vocab[token])
            except KeyError as KE:
                lineTag.append(1)  # vocab["<unknow>"]
                pass
        matrixTags.append(lineTag)

    # return vocab
    if returnvocab:
        return matrixTags, vocab
    return matrixTags
# train["sent"] = oneHot2D(train["sent"],vocab=wordVocab)
# test["sent"] = oneHot2D(test["sent"],vocab=wordVocab)
def readtxt(pathtain,pathtest):
    # GENE
    # PHENOTYPE
    train = {"sent": [], "gene": [], "phenotype": [], "rel": []}
    test = {"sent": [], "gene": [], "phenotype": [], "rel": []}
    for line in open(pathtain, "rb").read().decode("utf-8").strip().split("\n"):

        splits = line.strip().split('\t')
        # print(splits)
        # print(len(splits))

        train['sent'].append(splits[1])
        train['rel'].append(splits[-1])
        train['gene'].append(splits[2])
        train['phenotype'].append(splits[3])
    for line in open(pathtest, "rb").read().decode("utf-8").strip().split("\n"):

        splits = line.strip().split('\t')
        # print(splits)


        test['sent'].append(splits[1])
        test['rel'].append(splits[-1])
        test['gene'].append(splits[2])
        test['phenotype'].append(splits[3])
    # print(len(train['rel']))

    return train, test
def getSentWithRel(data):
    # print(len(data['rel']))
    # process sent
    sents = []
    for sent in data["sent"]:
        sent = sentCleaner(sent)
        sents.append(sent.split())

    # process rel
    # print("change")
    rels = []
    for rel in data["rel"]:
        if str(rel).lower() == "true": rels.append([1, 0])
        elif str(rel).lower() == "false": rels.append([0, 1])
        else:print(rel)

    # print(len(rels))

    return {"sent": sents, "gene": data["gene"], "phenotype": data["phenotype"], "rel": rels}
def wordCleaner(word):
    if re.search(r"^\d+\.\d+$", word) or str(word).isdigit():
        return "<num>"
    else:
        return str(word).lower().strip()
def sentClean(sents):
    sentTags = []
    for sent in sents:
        sentTag = []
        for word in sent:
            sentTag.append(wordCleaner(word))
        sentTags.append(sentTag)
    return sentTags
#        wf.write('do you want to take a trip.')
def loaddata(pathtain,pathtest):
    train, test = readtxt(pathtain,pathtest)
    # print(len(train['rel']))


    train = getSentWithRel(train)
    # print(len(train['rel']))
    test = getSentWithRel(test)

    train["sent"] = sentClean(train["sent"])
    test["sent"] = sentClean(test["sent"])
    
    #    print(train['sent'])

    # prepar for word
    wordVocab = getVocab(train["sent"] + test["sent"])
    train["sent"] = oneHot2D(train["sent"], vocab=wordVocab)
    test["sent"] = oneHot2D(test["sent"], vocab=wordVocab)
#    print(train["sent"].shape)
    # print(train['gene'][1])

    # prepar for window
    train["gene"], train["phenotype"] = oneHot2D(train["gene"], wordVocab), oneHot2D(train["phenotype"], wordVocab)
    test["gene"], test["phenotype"] = oneHot2D(test["gene"], wordVocab), oneHot2D(test["phenotype"], wordVocab)

    return train, test, wordVocab
print('load data...')
train, test, wordVocab = loaddata(filestrain,filestest)
print('load data done')
print('data preprocess...')
maxSentLen = 180
# train_sent = pad_sequences(train["sent"], maxlen=maxSentLen, padding="post")
# train_gene = pad_sequences(train["gene"], maxlen=5, padding="post")
# train_phenotype = pad_sequences(train["phenotype"], maxlen=5, padding="post")
test_sent = pad_sequences(test["sent"], maxlen=maxSentLen, padding="post")
test_gene = pad_sequences(test["gene"], maxlen=maxSentLen, padding="post")
test_phenotype = pad_sequences(test["phenotype"], maxlen=maxSentLen, padding="post")
train_rel = np.array(train["rel"])
test_rel = np.array(test["rel"])

# print(len(train_sent),len(train_rel),train_rel.shape)
# print(test_rel.shape)
# print('data preprocess done')
print("loading embedding...")
embedding_size = 200
embeddingVocab_size = len(wordVocab)

w2v_dir_path = "wikipedia-pubmed-and-PMC-w2v.bin"  # 2 0,934 unknow_per:  0.18722500379305113  unkownwords:  1234  vocab_size:  6591
#    w2v_dir_path = "PubMed-and-PMC-w2v.bin"

word2vec = KeyedVectors.load_word2vec_format(w2v_dir_path, binary=True, unicode_errors='ignore')

print("build embedding weights...")
embedding_weights = np.zeros((embeddingVocab_size + 1, embedding_size))
unknow_words = []
know_words = []
for word, index in wordVocab.items():
    try:
        embedding_weights[index, :] = word2vec[word.lower()]
        know_words.append(word)
    except KeyError as E:
        # print(E)
        unknow_words.append(word)
        embedding_weights[index, :] = np.random.uniform(-0.025, 0.025, embedding_size)
print("unknow_per: ", len(unknow_words) / embeddingVocab_size, " unkownwords: ", len(unknow_words), " vocab_size: ",
      embeddingVocab_size)
print('model...')
model = myModel(sent_lenth=maxSentLen)
model.attn(embedding_weights)
# model = myModel(sent_lenth=maxSentLen)
# model.attn()
print('train...')
model.train(inputs=train,
            batch_size=32,
            epochs=5,
            maxSentLen=maxSentLen
            )
print('predict')
y_ = model.predict([test_sent, test_gene, test_phenotype])


def get_Result(y):
    result = np.zeros(shape=y.shape)
    for i in range(y.shape[0]):
        if y[i, 0] >= y[i, 1]:
            result[i, 0] = 1
        else:
            result[i, 1] = 1
    return result
y_ = get_Result(y_)
from sklearn.metrics import f1_score
print(f1_score(y_, test_rel,average='micro'))
