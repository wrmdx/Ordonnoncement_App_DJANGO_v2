import numpy as np
import pulp as lp

Q=100000

def MILP_None(p,cont,S):

    sigma=[]
    # Create a PuLP problem
    problem = lp.LpProblem("My_LINGO_Problem", lp.LpMinimize)
    m=len(p[:,0])
    job=len(p[0,:])
    # Sets
    M = list(range(1, m+1))
    JOB = list(range(1, job+1))
    p = p.tolist()
    S=S.tolist()


    # Define binary decision variables
    x  = lp.LpVariable.dicts("X", ((i, j) for i in JOB for j in JOB), cat=lp.LpBinary)
    µ=lp.LpVariable.dicts("µ",((m,i,j) for m in JOB for i in JOB for j in JOB),cat=lp.LpBinary)

    # Parameters


    c = lp.LpVariable.dicts("c", ((i, j) for i in M for j in JOB), lowBound=0)
    SM = len(M)
    n = len(JOB)
    # Define the objective function
    problem += c[SM,n]

    # Constraints
    problem+=lp.lpSum(K*x[(K,1)] for K in JOB)==lp.lpSum(K*µ[(1,K,K)] for K in JOB)

    for J in range(2, job+1):
        problem+=lp.lpSum(K*x[(K,J)] for K in JOB)==lp.lpSum(lp.lpSum(K*µ[(J,L,K)] for L in JOB) for K in JOB)

    for J in range(2, job+1):
        problem+=lp.lpSum(K*x[(K,J-1)] for K in JOB)==lp.lpSum(lp.lpSum(L*µ[(J,L,K)] for L in JOB) for K in JOB)

    for J in JOB:
        problem+=lp.lpSum(lp.lpSum(µ[(J,L,K)] for L in JOB) for K in JOB)==1
    # Constraints for each JOB
    for K in JOB:
        problem += lp.lpSum(x[(K,J)] for J in JOB) == 1

    # Constraints for each POS
    for J in JOB:
        problem += lp.lpSum(x[(K,J)] for K in JOB) == 1

    # Constraints for c(1,J)

    problem += c[1,1] >= lp.lpSum(x[(K,1)] * (p[0][K-1]+S[0][K-1][K-1]) for K in JOB)

    # Constraints for c(I,J) with I >= 2
    for I in range(2, m+1):
        for J in JOB:
            problem += c[I,J] >= c[I - 1,J] + lp.lpSum(x[(K,J)] * p[I-1][K-1] for K in JOB)

    # Constraints for c(I,J) with J >= 2
    for I in M:
        for J in range(2, job+1):
            problem += c[I,J] >= c[I,J - 1] + lp.lpSum(x[(K,J)]*p[I-1][K-1]+ lp.lpSum(µ[(J,L,K)]*S[I-1][L-1][K-1] for L in JOB) for K in JOB)

    if cont=='no-idle':
        for I in M:
            for J in range(2, job + 1):
                problem += c[I, J] <= c[I, J - 1] + lp.lpSum(x[(K, J)] * p[I - 1][K - 1] for K in JOB)
    if 'no-wait' in cont:
        for I in range(2, m + 1):
            for J in JOB:
                problem += c[I, J] <= c[I - 1, J] + lp.lpSum(x[(K, J)] * p[I - 1][K - 1] for K in JOB)





    # Solve the problem
    problem.solve()

    # Print the results
    for J in JOB:
        for K in JOB:

            if x[(K,J)].varValue==1:
                print(int(K))
                sigma.append(int(K))

    for I in M:
        for J in JOB:
            print(c[(I,J)].varValue)


    # Print the objective value
    print(f"Objective Value: {problem.objective.value()}")




    return sigma

def MILP_JOB(O,p,op,M):
    C=[]
    problem = lp.LpProblem("My_LINGO_Problem", lp.LpMinimize)
    job=len(p)
    JOB=range(1,len(p)+1)
    M=range(1,M+1)
    Opm=max(op)
    Op=[]
    for i in range(len(op)):
        Op.append(range(1,op[i]+1))

    x = lp.LpVariable.dicts("X", ((j,k) for j in JOB for k in JOB), cat=lp.LpBinary)
    y = lp.LpVariable.dicts("Y", ((j, o, l) for j in JOB for o in range(1,Opm+1) for l in M), cat=lp.LpBinary)

    c = lp.LpVariable.dicts("c", ((j,o) for j in JOB for o in range(1,Opm+1)), lowBound=0)

    Cmax = lp.LpVariable("Cmax", lowBound=0)

    problem += Cmax

    for j in JOB:
        problem+=Cmax>=c[j,op[j-1]]

    for j in JOB:
        for o in Op[j-1]:
            problem+=y[(j,o,O[j-1][o-1])]==1

    for j in JOB:
        for o in Op[j-1]:
            problem+=lp.lpSum(y[(j,o,l)] for l in M)==1

    for j in JOB:
        for o in Op[j-1]:
            problem+=c[j,o]>=p[j-1][o-1]

    for j in JOB:
        for o in range(2,op[j-1]+1):
            problem+=c[j,o]>=c[j,o-1]+p[j-1][o-1]

    for j in JOB:
        for k in range(j+1,job+1):
            for o in Op[j-1]:
                for l in M:
                    problem+=c[j,o]>=c[k,o]+p[j-1][o-1]-Q*(2-y[(j,o,l)]-y[(k,o,l)]+x[(j,k)])

    for j in JOB:
        for k in range(j+1,job+1):
            for o in Op[j-1]:
                for l in M:
                    problem+=c[k,o]>=c[j,o]+p[k-1][min(o-1,op[k-1]-1)]-Q*(3-y[(j,o,l)]-y[(k,o,l)]-x[(j,k)])

    for j in JOB:
        for k in JOB:
            for o in range(2,op[j-1]+1):
                for w in range(1,o):
                    for l in M:
                        problem+=c[j,o]>=c[k,w]+p[j-1][o-1]-Q*(2-y[(j,o,l)]-y[(k,w,l)])

    problem.solve()

    for j in JOB:
        for k in JOB:
            print(f'X[{j},{k}]=',x[(j,k)].varValue)

    for j in JOB:
        for o in range(1,Opm+1):
            for l in M:
                print(f'Y[{j},{o},{l}]=',y[(j,o,l)].varValue)

    for j in JOB:
        for o in range(1,Opm+1):
            print(f'c[{j},{o}]=',c[j,o].varValue)

    for j in JOB:
        h=[]
        for o in Op[j-1]:
            h.append(c[(j,o)].varValue)
        C.append(h)

    print(C)




    print(f"Objective Value: {problem.objective.value()}")
    return C

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





