import copy
import random
import numpy as np

def fact(n):
    if n==0:
        return 1
    return n*fact(n-1)

M=[[5,7,3],
   [6,8,5]]
M=np.array(M)
c=np.zeros((len(M[:,0]),len(M[0,:])))
s=np.zeros((len(M[:,0]),len(M[0,:])))
cont='None'
R=np.zeros(len(M[0,:]))
S=np.zeros((len(M[:,0]),len(M[0,:]),len(M[0,:])))
def GA(s,c,M,cont,R,S):
    pop=CDS_gen(M)
    n=len(pop)
    i=0
    while i<n:
        if pop[i] in pop[i+1:]:
            pop[i:i+1]=[]
            n-=1
        else:
            i+=1
    n=len(pop)
    while n<min(20,fact(len(pop[0]))):
        i=random.randint(0,len(pop)-1)
        sigma=copy.deepcopy(pop[i])
        while sigma in pop:
            sigma=rswap(sigma)
        pop.append(sigma)
        n+=1
    print(pop)



    for i in range(len(pop)):
        sigma=copy.deepcopy(pop[i])
        l=Cmax(M,sigma,R,S)
        if i==0 or cmin>l:
            cmin=l
            sigmam=pop[i]

    Cb=cmin
    sigmab=sigmam

    g=1
    while g<200:
        P=[]
        Fit=fitness(M,pop,R,S)
        SF=sum(Fit)
        for i in range(len(pop)):
            P.append((Fit[i]+1)/(SF+1))

        pop=generate(pop,P)
        for i in range(len(pop)):
            l=Cmax(M,pop[i],R,S)
            if i==0 or cmin>l:
                cmin=l
                sigmam=copy.deepcopy(pop[i])
        if Cb>cmin:
            Cb=cmin
            sigmab=sigmam
        g+=1
    s,c=gantt_None(s,c,M,R,sigmab,S)
    return (s,c)


def generate(pop,P):
    n=0

    newpop=[]
    i=random.randint(0,len(pop)-1)

    while n<60:
        if random.random()<=P[i]:
            m=0
            while m==0:
                j=i
                while j==i:
                    j=random.randint(0,len(pop)-1)
                if random.random()<=P[j]:
                    sigma1=copy.deepcopy(pop[i])
                    sigma2 = copy.deepcopy(pop[j])
                    newpop.append(LOX(sigma1,sigma2))
                    m=1
                    n+=1
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






def fitness(M,pop,R,S):
    Fit=[]
    cpmax=Cpmax(M,pop,R,S)
    for i in range(len(pop)):
        Fit.append(cpmax-Cmax(M,pop[i],R,S))
    return Fit


def Cpmax(M,pop,R,S):
    for i in range(len(pop)):
        l=Cmax(M,pop[i],R,S)
        if i==0 or cpmax<l:
            cpmax=l

    return cpmax






def rswap(sigma):
    i=random.randint(0,len(sigma)-1)
    j=i
    while j==i:
        j=random.randint(0,len(sigma)-1)

    sigma[i],sigma[j]=sigma[j],sigma[i]
    return sigma







def CDS_gen(M):
    sigmas=[]

    M12 = np.zeros((3, len(M[0, :])))
    for k in range(0, len(M[:, 0]) - 1):
        nU = 0
        nV = 0
        U = []
        V = []
        M12[0] = sum(M[0:k + 1, :])
        M12[1] = sum(M[len(M[:, 0]) - (k + 1):len(M[:, 0]), :])
        M12[2] = np.array([i + 1 for i in range(len(M[0, :]))])
        for j in range(len(M[0, :])):
            if M12[0, j] < M12[1, j]:
                U.append([M12[0, j], M12[2, j]])
                nU += 1
            else:
                V.append([M12[1, j], M12[2, j]])
                nV += 1
        U = np.array(U).T
        V = np.array(V).T
        for j in range(nU - 1):
            for l in range(j + 1, nU):
                if U[0, j] > U[0, l]:
                    U[:, [j, l]] = U[:, [l, j]]

        for j in range(nV - 1):
            for l in range(j + 1, nV):
                if V[0, l] > V[0, j]:
                    V[:, [j, l]] = V[:, [l, j]]
        if nU * nV == 0:
            if nU == 0:
                sigma = V[1, :]
            else:
                sigma = U[1, :]
        else:
            sigma = np.concatenate((U[1, :], V[1, :]))

        sigmas.append(sigma.tolist())



    return sigmas

def Cmax(M,sigma,R,S):

    c=np.zeros((len(M[:,0]),len(M[0,:])))



    c[0,int(sigma[0]-1)]=R[int(sigma[0]-1)]+S[0,int(sigma[0] - 1),int(sigma[0] - 1)]+M[0,int(sigma[0]-1)]
    for i in range(1, len(M[:, 0])):
        c[i, int(sigma[0] - 1)] = max(c[i - 1, int(sigma[0] - 1)], S[i, int(sigma[0] - 1), int(sigma[0] - 1)])+ M[i, int(sigma[0] - 1)]

    for j in range(1, len(M[0, :])):
        c[0, int(sigma[j] - 1)] = max(c[0, int(sigma[j - 1] - 1)], R[int(sigma[j] - 1)]) + M[0, int(sigma[j] - 1)] + S[0, int(sigma[j - 1] - 1), int(sigma[j] - 1)]


    for i in range(1, len(M[:, 0])):
        for j in range(1, len(M[0, :])):
            c[i, int(sigma[j] - 1)] = max(c[i - 1, int(sigma[j] - 1)],
                                          c[i, int(sigma[j - 1] - 1)] + S[i, int(sigma[j - 1] - 1), int(sigma[j] - 1)])+ M[i, int(sigma[j] - 1)]

    return max(c[-1,:])

def gantt_None(s,c,M,R,sigma,S):
    s[0, int(sigma[0] - 1)] = R[int(sigma[0]-1)]+S[0,int(sigma[0] - 1),int(sigma[0] - 1)]
    c[0, int(sigma[0] - 1)] = s[0, int(sigma[0] - 1)] + M[0, int(sigma[0] - 1)]
    for i in range(1, len(M[:, 0])):
        s[i, int(sigma[0] - 1)] = max(c[i - 1, int(sigma[0] - 1)],S[i,int(sigma[0] - 1),int(sigma[0] - 1)])
        c[i, int(sigma[0] - 1)] = s[i, int(sigma[0] - 1)] + M[i, int(sigma[0] - 1)]
    for j in range(1,len(M[0,:])):
        c[0,int(sigma[j]-1)]=max(c[0,int(sigma[j-1]-1)],R[int(sigma[j]-1)])+M[0,int(sigma[j]-1)]+S[0,int(sigma[j-1] - 1),int(sigma[j] - 1)]
        s[0, int(sigma[j] - 1)]=c[0,int(sigma[j]-1)]-M[0,int(sigma[j]-1)]

    for i in range(1,len(M[:, 0])):
        for j in range(1, len(M[0, :])):
            s[i, int(sigma[j] - 1)] = max(c[i - 1, int(sigma[j] - 1)], c[i, int(sigma[j - 1] - 1)]+S[i,int(sigma[j-1] - 1),int(sigma[j] - 1)])
            c[i, int(sigma[j] - 1)] = s[i, int(sigma[j] - 1)] + M[i, int(sigma[j] - 1)]
    return (s,c)

print(GA(s,c,M,cont,R,S))
#print(rswap([3, 1, 5, 2, 4,7,6]))
#print(rswap([2,3,1]) in [[2,3,1],[3,2,1]])
#for i in range(1000):
    #print(i,LOX([1,2,3,4,5,6,7],[1,2,3,4,5,6,7]))
