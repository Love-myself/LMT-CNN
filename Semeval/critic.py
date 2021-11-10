import numpy as np

def Euclidean(P,Q):
    t=np.subtract(P,Q)
    t=np.multiply(t,t)
    t=np.sum(t,axis=1)
    t=np.sqrt(t)
    return np.mean(t)

def Srensen(p,q):
    x=np.abs(np.subtract(p,q))
    x=np.sum(x,axis=1)
    y=np.add(p,q)
    y=np.sum(y,axis=1)
    return np.mean(np.divide(x,y))

def Squared(p,q):
    x=np.subtract(p,q)
    x=np.multiply(x,x)
    y=np.add(p,q)
    t=np.divide(x,y)
    return np.mean(np.sum(t,axis=1))

def K_L(p,q): #约定 0*log(0/q(x))=0;   p(x)*log(p(x)/0)=infinity;
    m,n=p.shape
    t=np.array([],dtype=float)
    for i in range(m):
        y=0.
        for j in range(n):
            if p[i][j]==0:
                y+=0.
            elif q[i][j]==0:
                y=float('inf')
            else:
                y+=p[i][j]*np.log(p[i][j]/q[i][j])
        if y!=float('inf'):
            t=np.append(t,y)
    return np.mean(t)

def Intersection(p,q):
    t=np.minimum(p,q)
    return np.mean(np.sum(t,axis=1))

def Cosine(p,q):
    x=np.multiply(p,q)
    x=np.sum(x,axis=1)
    y1=np.multiply(p,p)
    y1=np.sum(y1,axis=1)
    y1=np.sqrt(y1)
    y2=np.multiply(q,q)
    y2=np.sum(y2,axis=1)
    y2=np.sqrt(y2)
    y=np.multiply(y1,y2)

    t=np.array([],dtype=float)
    for i in range(len(y)):
        if y[i]!=0:
            t=np.append(t,x[i]/y[i])

    return np.mean(t) #t=np.divide(x,y)
