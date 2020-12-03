import numpy as np
import matplotlib.pyplot as plt
import random
import math

#Simulate normalized matrix A randomly

N=2000 #Number of agents
T=1000 #Number of process iterations
def pro(x):
    return((1+math.tanh(x))/2)

gamma=0.5  #define parameter Gamma

L=np.zeros((N,T))
D=[]
A=[] 
D.append(0)
for i in range (N):
    p=random.uniform(0,1)
    if p<0.5:
        L[i,0]=1
    else:
        L[i,0]=-1

A.append(sum(L[:,0]))

for j in range(1,T):
    for i in range(N):
        p=random.uniform(0,1)
        if p>pro(gamma*D[j-1]):
            L[i,j]=-1
        else:
            L[i,j]=1
    A.append(sum(L[:,j]))
    D.append(D[j-1]-A[j]/N)

L=np.array(L,dtype=int)
A1=np.array(A)
A1=A1/math.sqrt(N)  # normalize matrix A

# Do Bayesian inference with Stan

import pystan
import cython

MG_code = """

data {
    int < lower =50> N;
    int < lower =500> T;
    vector [T] A1;
    vector [T] D;
}
transformed data {
}
parameters {
    real <lower=0.01,upper=1.5> gamma;
}

model {
    gamma~uniform(0.01,1.5);
    for (j in 2:T) {
        A1[j]~normal(sqrt(N)*tanh(gamma*D[j-1]),1/(cosh(gamma*D[j-1]))^2);
    }    
}

"""

model = pystan.StanModel(model_code=MG_code)   #training model
MG_data = {'N': 2000,
           'T': 1000,
           'A1': A1,
           'D': D}

fit = model.sampling(data=MG_data, iter=1000)   #fitting the model

#Plot results
plt.figure()
fit.plot()
