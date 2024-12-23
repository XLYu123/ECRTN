import os
import pickle
# import pickle5 as pickle
import numpy as np
import pandas as pd
import scipy.io as scio
#定义保存文件
def save_pickle(path,name,x):

    with open(path+name, 'wb') as f:
        pickle.dump(x, f, protocol=4)
    print('save to path:',path)
    print('Save successfully!')


#加载原始文件

file =r"F:\桌面\无人潜航器\无人潜航器\滑槽松动3000.xlsx"
data = pd.read_excel(file,usecols=[0])  #DataFrame

# 从一维数据中随机生成样本

Train_sample = []

xx = len(data) #数据长度
length = 1024
sample_n = 1000 #样本数

for j in range(sample_n):
    random_start = np.random.randint(low=0, high=(xx - 2 * length))
    X1 = data[random_start: random_start + length]
    Train_sample.append(X1)



b = np.array(Train_sample)



path_out=r"F:\桌面\长度1024/"
os.makedirs(path_out,exist_ok=True) #如果没有该文件夹，则创建此文件夹
save_pickle(path_out,name='C6_1.pkl',x=b)
# with open(path_out,'wb') as f:
#     pickle.dump((path_out,f)
print('Next')
