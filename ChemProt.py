
# -*- coding: utf-8 -*-

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
import re
import numpy as np

#import _pickle as cPickle
from keras.utils.np_utils import *
from sklearn.preprocessing import LabelEncoder
#import gzip,json
from gensim.models import KeyedVectors
from collections import Counter
from keras.preprocessing.sequence import pad_sequences

Train_files = r'data/ChemProt/ftrain.tsv'
Test_files = r'data/ChemProt/ftest.tsv'
def sentCleaner(sent):
    sent = re.sub(r"\.$","",sent)
    sent = re.sub(r"-"," - ",sent)
    sent = re.sub(r"\.\s"," . ",sent)
    sent = re.sub(r","," , ",sent)
    sent = re.sub(r"<"," ( ",sent)
    sent = re.sub(r">"," ( ",sent)
    sent = re.sub(r"\("," ( ",sent)
    sent = re.sub(r"\)"," ) ",sent)
    sent = re.sub(r"/"," : ",sent)
    sent = re.sub(r":"," : ",sent)
    sent = re.sub(r";"," : ",sent)
    sent = re.sub(r"%"," % ",sent)
    sent = re.sub(r"\["," [ ",sent)
    sent = re.sub(r"\]"," ] ",sent)
    sent = re.sub(r"="," = ",sent)
    return sent
def getVocab(matrix,ndim=2,isClean=False,isUnknow=False):
    if isClean is True:
        matrix = sentClean(matrix)

    counter = Counter()
    if ndim == 1:
        for token in matrix:
            try:
                counter[token]+=1
            except KeyError as KE:
                print("vocab1Derror:",KE)
                exit()
    elif ndim == 2:
        for line in matrix:
            for token in line:
                try:
                    counter[token]+=1
                except KeyError as KE:
                    print("vocab2Derror:",KE)
                    exit()
    elif ndim == 3:
        for line in matrix:
            for token in line:
                for char in token:
                    try:
                        counter[str(char)]+=1
                    except KeyError as KE:
                        print("vocab2Derror:",KE)
                        exit()
    counter = sorted(counter.most_common(), key=lambda x: x[1], reverse=True)
    if isUnknow:
        vocab = {str(v[0]):i+2 for i,v in enumerate(counter)}
        vocab["<unknow>"] = 1
        return vocab  # 0 as padding,1 as unknow
    return {str(v[0]):i+1 for i,v in enumerate(counter)}  # 0 as padding,1 as unknow
def oneHot2D(matrix,vocab=None,returnvocab=False):
    '''
    one-hot encoding for word / pos
    '''

    # creat vocab
    if vocab is None:
        counter = Counter()
        for line in matrix:
            for token in line:
                try:
                    counter[token]+=1
                except KeyError as KE:
                    pass
        counter = sorted(counter.most_common(), key=lambda x: x[1], reverse=True)
        vocab = {v[0]:i+1 for i,v in enumerate(counter)}

    # replace token to token-id
    matrixTags = []
    for l,line in enumerate(matrix):
        lineTag = []
        for t,token in enumerate(line):
            try:
                lineTag.append(vocab[token])
            except KeyError as KE:
                lineTag.append(vocab["<unknow>"])
                pass
        matrixTags.append(lineTag)

    # return vocab
    if returnvocab:
        return matrixTags,vocab
    return matrixTags
def readtxt(path):
    train = {"sent":[],"rel":[]}
    for line in open(path,"rb").read().decode("utf-8").strip().split("\n"):
        splits = line.strip().split('\t')

        train['sent'] .append (splits[1])
        train['rel'] .append(splits[2])
    
    return train
def getSentWithRel(data):
    # process sent
    sents = []
    for sent in data["sent"]:
        sent = sentCleaner(sent)
        sents.append(sent.split())

    # process rel
    encoder = LabelEncoder()
    rels = to_categorical(encoder.fit_transform(data['rel']), 5)#'DDI-false,DDI-effect,DDI-mechanism,DDI-advise,DDI-int
#    rels = []
#    for rel in data["rel"]:
#        if rel == "rd-disab": rels.append([1,0])
#        if rel == 'None': rels.append([0,1])

    return {"sent":sents,"rel":rels}
def wordCleaner(word):
    if re.search(r"^\d+\.\d+$",word) or str(word).isdigit():
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
def getGenEmbedding(sents,vocab):
    genEmbedding,disEmbedding = [],[]
    for sent in sents:
        chemical,gene = [],[]
        for w,word in enumerate(sent):
            if word == vocab['@chemical$']:
                for i in range(3):
                    try:
                        chemical.append(sent[w-2+i])
                    except IndexError as IE:
                        chemical.append(0)
                        continue
                for i in range(2):
                    try:
                        chemical.append(sent[w+1+i])
                    except IndexError as IE:
                        chemical.append(0)
                        continue
            
            elif word == vocab["@gene$"]:
                for i in range(3):
                    try:
                        gene.append(sent[w-3+i])
                    except IndexError as IE:
                        gene.append(0)
                        continue
                for i in range(2):
                    try:
                        gene.append(sent[w+1+i])
                    except IndexError as IE:
                        gene.append(0)
                        continue
            else:
                pass
#        print(chemical,gene)
        genEmbedding.append(chemical)
        disEmbedding.append(gene)

    return genEmbedding,disEmbedding
#readtxt(r'data\\DDI\\train.tsv')
def loaddata(Train_path,Test_path):
    train = readtxt(Train_path)
    test = readtxt(Test_path)
#    print(train['rel'])
    train = getSentWithRel(train)
    test = getSentWithRel(test)
#    print(train['rel'])
    train["sent"] = sentClean(train["sent"])
    test["sent"] = sentClean(test["sent"])

    # prepar for word
    wordVocab = getVocab(train["sent"]+test["sent"])
#    print(wordVocab)
    train["sent"] = oneHot2D(train["sent"],vocab=wordVocab)
    test["sent"] = oneHot2D(test["sent"],vocab=wordVocab)

    # prepar for window
    train["chemical"],train["gene"] = getGenEmbedding(train["sent"],vocab=wordVocab)
    test["chemical"],test["gene"] = getGenEmbedding(test["sent"],vocab=wordVocab)

    return train,test,wordVocab
train,test,wordVocab=loaddata(Train_files,Test_files)
maxlen=150
print("max sentence len:",maxlen)
# train_sent = pad_sequences(train["sent"],maxlen=maxlen,padding="post")
# train_chemical = pad_sequences(train["chemical"],maxlen=5,padding="post")
# train_gene = pad_sequences(train["gene"],maxlen=5,padding="post")
test_sent = pad_sequences(test["sent"],maxlen=maxlen,padding="post")
test_chemical = pad_sequences(test["chemical"],maxlen=maxlen,padding="post")
test_gene = pad_sequences(test["gene"],maxlen=maxlen,padding="post")
# train_rel = np.array(train["rel"])
test_rel = np.array(test["rel"])
# print(train_rel.shape)
print(test_rel.shape)
print("loading embedding...")
embedding_size = 200
embeddingVocab_size = len(wordVocab)

w2v_dir_path = "wikipedia-pubmed-and-PMC-w2v.bin"#2 0,934 unknow_per:  0.18722500379305113  unkownwords:  1234  vocab_size:  6591
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
#pdict={Lclass:5,ativ:"softmax",gamma:2,alpha:0.2}

model = myModel(sent_lenth=maxlen,Lclass=5,ativ="softmax")
model.attn(embedding=embedding_weights)

# model = myModel(sent_lenth=maxlen,Lclass=5,ativ="softmax")
# model.attn()
print('train...')
model.train(inputs=train,
            batch_size=32,
            epochs=3,
            maxSentLen=maxlen
            )
from sklearn.metrics import f1_score,precision_recall_fscore_support
def get_Result(y):
    result = np.zeros(shape=y.shape)
    for i in range(y.shape[0]):
        result[i][np.argmax(y[i])]=1
#    data_write_csv('rtest.tsv',result)
    return result 
y_ = model.predict([test_sent,test_chemical,test_gene])
#print(y_)
y = get_Result(y_)
#print("F1score_weighted:%.3f"%f1_score(test_rel,y,average='weighted'))
#print("F1score_macro:%.3f"%f1_score(test_rel,y,average='macro'))
#print("F1score_micro:%.3f"%f1_score(test_rel,y,average='micro'))

p,r,f,_=precision_recall_fscore_support(test_rel,y,average='micro')
print("precision_recall_fscore_support--precision:%.3f,recall:%.3f,f1_score:%.3f"%(p,r,f))

