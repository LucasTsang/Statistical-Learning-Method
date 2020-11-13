'''

Conditional Random Field

'''

from numpy import *
#这里定义T为转移矩阵列代表前一个y(ij)代表由状态i转到状态j的概率,Tx矩阵x对应于时间序列
#这里将书上的转移特征转换为如下以时间轴为区别的三个多维列表，维度为输出的维度
T1 = [[0.6, 1], [1, 0]]
T2 = [[0, 1], [1, 0.2]]
#将书上的状态特征同样转换成列表,第一个是为y1的未规划概率，第二个为y2的未规划概率
S0 = [1, 0.5]
S1 = [0.8, 0.5]
S2 = [0.8, 0.5]
Y = [1, 2, 2]  #即书上例一需要计算的非规划条件概率的标记序列
Y = array(Y) - 1  #这里为了将数与索引相对应即从零开始
P = exp(S0[Y[0]])
for i in range(1, len(Y)):
    P *= exp((eval('S%d' % i)[Y[i]]) + eval('T%d' % i)[Y[i - 1]][Y[i]])
print(P)
print(exp(3.2))


# Exercise
import numpy as np

# 创建随机矩阵
M1 = [[0, 0], [0.5, 0.5]]
M2 = [[0.3, 0.7], [0.7, 0.3]]
M3 = [[0.5, 0.5], [0.6, 0.4]]
M4 = [[0, 1], [0, 1]]
M = [M1, M2, M3, M4]
print(M)

# 生成路径
path = [2]
for i in range(1, 4):
    paths = []
    for _, r in enumerate(path):
        temp = np.transpose(r)
        paths.append(np.append(temp, 1))
        paths.append(np.append(temp, 2))
    path = paths.copy()

path = [np.append(r, 2) for _, r in enumerate(path)]
print(path)

# 计算概率

pr = []
for _, row in enumerate(path):
    p = 1
    for i in range(len(row) - 1):
        a = row[i]
        b = row[i + 1]
        p *= M[i][a - 1][b - 1]
    pr.append((row.tolist(), p))
pr = sorted(pr, key=lambda x: x[1], reverse=True)
print(pr)

# 打印结果
print("以start=2为起点stop=2为终点的所有路径的状态序列y的概率为：")
for path, p in pr:
    print("    路径为：" + "->".join([str(x) for x in path]), end=" ")
    print("概率为：" + str(p))
print("概率[" + str(pr[0][1]) + "]最大的状态序列为:",
      "->".join([str(x) for x in pr[0][0]]))
      
