# Simulate the evolution of the number of black ants

import random
import matplotlib.pyplot as plt

N=100
eps=0.05
delta=0.4
av=[]
rec=[]
for i in range (N+1):
    av.append((1-i/N)*(eps+(1-delta)*i/(N-1)))
    rec.append((i/N)*(eps+(1-delta)*(N-i)/(N-1)))
T=[]
k0=N//2
T.append(k0)

for i in range (9999):
    p=random.uniform(0,1)
    if T[i]==0 and p<eps:
        T.append(1)
    elif T[i]==N and p<eps:
        T.append(N-1)
    elif p<av[T[i]]:
        T.append(T[i]+1)
    elif p<av[T[i]]+rec[T[i]]:
        T.append(T[i]-1)
    else:
        T.append(T[i])

plt.plot(T)

# Bayesian inference with Stan

import pystan
import cython

Kir_code = """
functions {
    

        real dist_lpdf(real y,real Fj, real N, real eps,real delta) {
            real P1;
            real P2;
            P1=(1-Fj/N)*(eps+(1-delta)*Fj/(N-1));
            P2=(Fj/N)*(eps+(1-delta)*(N-Fj)/(N-1));
            if (y !=Fj && (y !=Fj-1) && (y!=Fj+1)) reject("dist_lpdf:  illegal value for y: ", y);
            if (y == Fj-1) return log(P2);
            else if (y == Fj+1) return log(P1);
            else return log(1-(P1+P2));
            }
}
data {
    int < lower =2> N;
    int < lower =0> T;
    vector [T] F;
}

parameters {
    real <lower =0, upper=1> eps;
    real <lower =0, upper =1> delta ;
}
transformed parameters {
    
}
model {
    eps ~ beta(1,4) ;
    delta ~ beta(2,2) ;
    for (j in 2:T)
        F[j]~dist(F[j-1],N,eps,delta);
    

}

"""

model = pystan.StanModel(model_code=Kir_code)

Kir_data = {'N': 100,
           'T': 10000,
           'F': T}

fit = model.sampling(data=Kir_data, iter=1000)

fit.plot()
