### -*- coding: utf-8 -*-
##"""
##Created on Fri May 29 14:47:54 2020
##
##@author: Administrator
##"""
import random


raw=[]
for line in open('test.tsv',"rb").read().decode("utf-8").strip().split("\n"):
    line = line.strip()#默认删除空白符
    line = line.split('\t')#使用|拆分成【1,2,3】
#    if line[2]=="DDI-false":
#        continue
    raw.append(line) 
#print(raw_data)
random.shuffle(raw)
import csv
import codecs

def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
  file_csv = codecs.open(file_name,'w+','utf-8')#追加
  writer = csv.writer(file_csv,delimiter='\t')
#  writer.writerow(['id','text','label'])
  for data in datas:
    writer.writerow(data)
data_write_csv('rtest.tsv',raw)
print("保存文件成功，处理结束")
#DDI-DrugBank.d63.s0.p46	Monoamine oxidase (MAO) inhibitors such as isocarboxazid (e.g., Marplan), phenelzine (e.g., Nardil), @DRUG$ (e.g., Matulane), @DRUG$ (e.g., Eldepryl), and tranylcypromine (e.g., Parnate): Using these medicines with L-tryptophan may increase the chance of side effects.	DDI-false

##import random
#from collections import Counter
#
##ADE_V2 = list(open('train-todo.txt', 'r',))
##label={}
maxl=0
raw=[]
la=[]
for line in open('ftrain.tsv',"rb").read().decode("utf-8").strip().split("\n"):
    line = line.strip()#默认删除空白符
    line = line.split('\t')#使用|拆分成【1,2,3】
    s=line[1].split(' ')
    if maxl<len(s):
        
        maxl=len(s)
    raw.append(line[2])
#    if line[2] not in label:
        
#print(raw_data)
print(maxl)
print(Counter(raw))
#
#
