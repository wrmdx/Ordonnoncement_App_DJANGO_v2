import numpy as np



def Jackson(O,p):
    M = []
    E1 = []
    E2 = []
    E12 = []
    E21=[]
    for i in range(len(O)):
        if len(O[i])==1:
            if O[i][0]==1:
                E1.append(i+1)
            else:
                E2.append(i + 1)
        elif O[i][0]==1:
            E12.append(i+1)
        else:
            E21.append(i+1)
    for i in range(len(E12)):
        M.append([p[E12[i]-1][0],p[E12[i]-1][1],E12[i]])
    M=np.array(M)
    M=M.T
    if len(E12)!=0:
        E12=gantt_CDS(M)
    M=[]
    for i in range(len(E21)):
        M.append([p[E21[i]-1][0],p[E21[i]-1][1],E21[i]])
    M=np.array(M)
    M=M.T
    if len(E21)!=0:
        E21=gantt_CDS(M)
    E1=np.array(E1)
    E2=np.array(E2)
    sigma1=np.concatenate((E12,E1,E21),axis=0)
    sigma2=np.concatenate((E21,E2,E12),axis=0)
    if len(sigma1)>len(sigma2):
        while len(sigma1)>len(sigma2):
            sigma2=np.append(sigma2,0)
    elif len(sigma2)>len(sigma1):
        while len(sigma2)>len(sigma1):
            sigma1=np.append(sigma1,0)

    sigma=np.vstack((sigma1,sigma2))
    return sigma




def gantt_CDS(MS):

    M=MS[0:-1,:]
    M12=np.zeros((3,len(M[0,:])))
    for k in range(0,len(M[:,0])-1):
        nU = 0
        nV = 0
        U=[]
        V=[]
        M12[0]=sum(M[0:k+1,:])
        M12[1]=sum(M[len(M[:,0])-(k+1):len(M[:,0]),:])
        M12[2]=MS[2,:]
        for j in range(len(M[0,:])):
            if M12[0,j]<M12[1,j]:
                U.append([M12[0,j],M12[2,j]])
                nU+=1
            else:
                V.append([M12[1,j], M12[2,j]])
                nV+=1
        U=np.array(U).T
        V = np.array(V).T
        for j in range(nU-1):
            for l in range(j+1,nU):
                if U[0, j]>U[0, l]:
                    U[:,[j,l]]=U[:,[l,j]]

        for j in range(nV-1):
            for l in range(j+1,nV):
                if V[0, l]>V[0, j]:
                    V[:,[j,l]]=V[:,[l,j]]
        if nU*nV==0:
            if nU==0:
                sigma=V[1,:]
            else:
                sigma = U[1, :]
        else:
            sigma=np.concatenate((U[1,:],V[1,:]))



    return sigma



