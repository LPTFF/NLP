import numpy as np
from urllib import request
from bs4 import BeautifulSoup
import re


import time

n=np.random.permutation(7)


headers_dict={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:61.0) Gecko/20100101 Fierfox/61.0'}
url1="http://news.sohu.com/"
res=request.Request(url=url1,headers=headers_dict)
target_response=request.urlopen(res).read().decode('utf-8')
soup=BeautifulSoup(target_response,'html.parser')

for new in soup.select('.focus-news'):
    for i in range(len(new.select('a'))):
        title_1=new.select('a')[i].attrs['title']
        url2=new.select('a')[i].attrs['href']
        res1=request.Request(url=url2,headers=headers_dict)
        try :
            target_response1=request.urlopen(res1).read().decode('utf-8')
        except:
            continue
        else:
            soup1=BeautifulSoup(target_response1,'html.parser')
            for ln in soup1.select('.article'):
                b=[]
                for i in range(2,len(ln.select('p'))-1):
                    b.append(str(ln.select('p')[i]))
                b="".join(b)

                p2 = re.compile('[\u4e00-\u9fa5\d，。、%；：""【】]')
                c=re.findall(p2,b)
                t=1
                while t:
                    i=-1

                    try:
                        if c[i]!='。':
                            del c[i]

                            i=i-1
                        else:
                            t=0
                    except:
                        break
                    else:
                        continue
                c="".join(c)
            p1=re.compile('[\u4e00-\u9fa5\d]')
            with open('C:\\Users\\hasee\\Documents\\news\\'+str(''.join(re.findall(p1,title_1)))+'.txt','w') as tf:
                tf.write(c)
                tf.close()
            time.sleep(2)
print('完成')


