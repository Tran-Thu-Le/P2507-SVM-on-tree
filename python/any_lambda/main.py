import numpy as np
from svm_on_tree import SVMOnTreeLambda
from sklearn.svm import LinearSVC

def toy():
    rng = np.random.RandomState(0)
    X0 = rng.randn(50,2) + np.array([-3,0])
    X1 = rng.randn(50,2) + np.array([+3,0])
    X = np.vstack([X0,X1]).astype(float)
    y = np.hstack([np.zeros(50,int), np.ones(50,int)])

    model = SVMOnTreeLambda(lamda=1.0)
    res = model.fit(X,y)
    _, acc2n, _ = model.predict_on_2n(X,y)
    print("Tree:", res, "acc(2N)=", acc2n)

    svc = LinearSVC(max_iter=10000).fit(X,y)
    from svm_on_tree import project_spine
    Xproj,_,_ = project_spine(X,y)
    y2 = np.hstack([y,y])
    acc_svc = (svc.predict(np.vstack([X,Xproj])) == y2).mean()
    print("LinearSVC acc(2N)=", acc_svc)

if __name__ == "__main__":
    toy()
