'''

Latent Semantic Analytic


'''
import numpy as np
from sklearn.decomposition import TruncatedSVD
X = [[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 2, 3], [0, 0, 0, 1], [1, 2, 2, 1]]
X = np.asarray(X);X
# 奇异值分解
U,sigma,VT=np.linalg.svd(X)
# 截断奇异值分解

svd = TruncatedSVD(n_components=3, n_iter=7, random_state=42)
svd.fit(X)


# Non-negative Matrix Factorize(NMF)

def inverse_transform(W, H):
    # 重构
    return W.dot(H)

def loss(X, X_):
    #计算重构误差
    return ((X - X_) * (X - X_)).sum()
    
class MyNMF:
    def fit(self, X, k, t):
        m, n = X.shape
        
        W = np.random.rand(m, k)
        W = W/W.sum(axis=0)
        
        H = np.random.rand(k, n)
        
        i = 1
        while i < t:
            
            W = W * X.dot(H.T) / W.dot(H).dot(H.T)
            
            H = H * (W.T).dot(X) / (W.T).dot(W).dot(H)
            
            i += 1
            
        return W, H
        
model = MyNMF()
W, H = model.fit(X, 3, 200)
# 重构误差

loss(X, X_)

# scikit-learn

from sklearn.decomposition import NMF
model = NMF(n_components=3, init='random', max_iter=200, random_state=0)
W = model.fit_transform(X)
H = model.components_