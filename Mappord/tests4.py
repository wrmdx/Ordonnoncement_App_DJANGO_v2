import copy
import random
import numpy as np
pop=[[1, 2, 3, 4, 5, 6, 7],
     [1, 2, 3, 4, 5, 6, 7],
     [1, 2, 3, 4, 5, 6, 7],
     [1, 2, 3, 4, 5, 6, 7],
     [1, 2, 3, 4, 5, 6, 7],
     [1, 2, 3, 4, 5, 6, 7],
     [1, 2, 3, 4, 5, 6, 7],
     [1, 3, 2, 4, 5, 6, 7],
     [1, 2, 3, 4, 5, 6, 7],
     [1, 2, 3, 4, 5, 6, 7]]
P=[1,1,1,1,1,1,1,1,1,1]
def generate(pop,P):
    n=0

    newpop=[]


    while n<10:
        i = random.randint(0, len(pop) - 1)
        if random.random()<=P[i]:
            m=0
            while m==0:
                j=i
                while j==i:
                    j=random.randint(0,len(pop)-1)
                if random.random()<=P[j]:
                    sigma1=copy.deepcopy(pop[i])
                    sigma2=copy.deepcopy(pop[j])
                    newpop.append(LOX(sigma1,sigma2))
                    m=1
                    n+=1
                    print(n)

    return newpop

def LOX(sigma1,sigma2):


    i=random.randint(0,len(sigma1)-1)
    j=i
    while j==i:
        j=random.randint(0,len(sigma1)-1)
    sigma=sigma1[i:j+1]
    if i>j:
        i,j=j,i

    n=len(sigma1)
    k=0
    while k<n:
        if sigma2[k] in sigma:
            sigma2[k:k+1]=[]
            n-=1
        else:
            k+=1
    sigma=sigma2[0:i]+sigma+sigma2[i:]
    if random.random()<=0.05:
        i=random.randint(0,len(sigma1)-1)
        e=sigma[i]
        sigma[i:i+1]=[]
        d=random.randint(1,100)
        i=(i+d)%len(sigma1)
        sigma=sigma[0:i]+[e]+sigma[i:]

    return sigma

print(generate(pop,P))