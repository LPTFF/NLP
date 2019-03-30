import os
import numpy as np
import re
import jieba
n=[]
with open('C:\\Users\\hasee\\Documents\\stopwords.txt','r',encoding='utf8') as tf:
    for i in tf.readlines():
        n.append(i.strip())
    tf.close()






path='C:\\Users\\hasee\\Documents\\news\\'

if os.path.isdir(path):
    print('文件已经存在')
else:
    os.mkdir(path)

for i in os.listdir(path):
    
    with open(path+str(i),'r')as tf :
        sentence="".join(tf.readlines())
        

        s3=sentence.split('。')

        with open ('C:\\Users\\hasee\\Documents\\result\\'+str(i),'w',encoding='utf8') as te:
            for s in s3:
                s1= jieba.cut(s,cut_all=False)
                s2=''
                for i in s1:
                    if i not in n and i!='\t'and not i.isdigit():
                        s2=s2+i
                        s2=s2+' '
                te.write(s2.strip())
                te.write('\n')





