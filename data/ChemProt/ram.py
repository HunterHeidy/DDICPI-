#### -*- coding: utf-8 -*-
###"""
###Created on Fri May 29 14:47:54 2020
###
###@author: Administrator
###"""
import random
##
maxl=0
raw=[]
for line in open('train.tsv',"rb").read().decode("utf-8").strip().split("\n"):
    line = line.strip()#默认删除空白符
    line = line.split('\t')#使用|拆分成【1,2,3】
    s=line[1].split(' ')
#    if maxl<len(s):
#        maxl=len(s)
    if line[2]=="false":
        continue
    raw.append(line) 
#print(raw_data)
#print(maxl)
random.shuffle(raw)

import csv
import codecs

def data_write_csv(file_name, datas):#file_name为写入CSV文件的路径，datas为要写入数据列表
  file_csv = codecs.open(file_name,'w+','utf-8')#追加
  writer = csv.writer(file_csv,delimiter='\t')
  
#  writer.writerow(['id','text','label'])
  for data in datas:
    writer.writerow(data)
data_write_csv('ndtrain.tsv',raw)
print("保存文件成功，处理结束")

##import random
#from collections import Counter
#
##ADE_V2 = list(open('train-todo.txt', 'r',))
##label={}
#raw=[]
#la=[]
#for line in open('ftrain.tsv',"rb").read().decode("utf-8").strip().split("\n"):
#    line = line.strip()#默认删除空白符
#    line = line.split('\t')#使用|拆分成【1,2,3】
#    raw.append(line[2])
##    if line[2] not in label:
#        
##print(raw_data)
#print(Counter(raw))


# -*- coding: utf-8 -*-

