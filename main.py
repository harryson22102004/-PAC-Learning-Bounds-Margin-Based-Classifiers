import numpy as np
def sample_complexity_pac(epsilon, delta, vc_dim):
    """VC dimension based sample complexity bound."""
    return int((2/epsilon)*(np.log(2/delta)+vc_dim*np.log(2/epsilon)))+1
 
def rademacher_complexity_linear(n, d, B=1):
    """Rademacher complexity of linear classifiers."""
    return B/np.sqrt(n)*np.sqrt(d)
 
class SVMPrimalSGD:
    def __init__(self, C=1.0, n_iter=1000, lr=0.01):
        self.C=C; self.n=n_iter; self.lr=lr; self.w=None; self.b=0
    def fit(self, X, y):
        self.w=np.zeros(X.shape[1])
        for t in range(1, self.n+1):
            i=np.random.randint(len(X)); xi=X[i]; yi=y[i]
            lr=self.lr/t
            if yi*(self.w@xi+self.b)<1:
                self.w=(1-lr)*self.w+lr*self.C*yi*xi
                self.b+=lr*self.C*yi
            else:
                self.w=(1-lr)*self.w
    def predict(self, X): return np.sign(X@self.w+self.b)
    def margin(self): return 1/np.linalg.norm(self.w) if np.linalg.norm(self.w)>0 else 0
 
np.random.seed(42)
X=np.vstack([np.random.randn(100,2)+[2,2], np.random.randn(100,2)-[2,2]])
y=np.array([1]*100+[-1]*100)
svm=SVMPrimalSGD(C=1.0)
svm.fit(X,y)
acc=(svm.predict(X)==y).mean()
print(f"SVM accuracy: {acc:.3f}, margin: {svm.margin():.3f}")
print(f"PAC bound (eps=0.05, delta=0.05, VC=3): n≥{sample_complexity_pac(0.05,0.05,3)}")
