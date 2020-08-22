
raw=[]

for line in open('train_1.tsv',"rb").read().decode("utf-8").strip().split("\n"):
    line = line.strip()#默认删除空白符
    line = line.split('\t')#使用|拆分成【1,2,3】
    if line[-1] == 'RELATION': continue
    x=[]
    x.append(line[0])
    x.append(line[1])
    x.append(line[2])
    x.append(line[3])
    x.append(line[-1])
    raw.append(x)
# print(raw[:8])
for line in open('train_2.tsv',"rb").read().decode("utf-8").strip().split("\n"):
    line = line.strip()#默认删除空白符
    line = line.split('\t')#使用|拆分成【1,2,3】
    if line[-1]=='RELATION':continue
    x=[]
    x.append(line[0])
    x.append(line[1])
    x.append(line[2])
    x.append(line[3])
    x.append(line[-1])
    raw.append(x)


import codecs,csv,random
random.shuffle(raw)
def data_write_csv(file_name, datas):  # file_name为写入CSV文件的路径，datas为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter='\t')

    #  writer.writerow(['id','text','label'])
    for data in datas:
        writer.writerow(data)


data_write_csv('train_12.tsv', raw)
print("保存文件成功，处理结束")
