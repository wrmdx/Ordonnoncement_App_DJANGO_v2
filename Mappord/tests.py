
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

#p=[[[8,6,9]],[[3,2,5],[6,5,10]]]

#ME=[1,2]
# def MILP_HYB(p,ME):
#     sigma=[]
#     problem = lp.LpProblem("My_LINGO_Problem", lp.LpMinimize)
#
#     et=len(ME)
#     M=[]
#     for i in range(et):
#         M.append(range(1,ME[i]+1))
#     job=len(p[0][0][:])
#     ET=list(range(1,et+1))
#     JOB=list(range(1,job+1))
#
#
#     x = lp.LpVariable.dicts("X", ((j, e, l) for j in JOB for e in ET for l in M[e-1]), cat=lp.LpBinary)
#     y = lp.LpVariable.dicts("Y", ((j, k, e) for j in JOB for k in JOB for e in ET), cat=lp.LpBinary)
#
#     c = lp.LpVariable.dicts("c", ((e, j) for e in ET for j in JOB), lowBound=0)
#
#     Cmax=lp.LpVariable("Cmax",lowBound=0)
#
#     problem+=Cmax
#
#     for E in ET:
#         for J in JOB:
#             problem+=lp.lpSum(x[(J,E,L)] for L in M[E-1])==1
#
#
#     for J in JOB:
#         problem+=c[1,J]>=lp.lpSum(x[(J,1,L)]*p[0][L-1][J-1] for L in M[0])
#
#     for J in JOB:
#         problem+=Cmax>=c[et,J]
#
#     for E in range(2,et+1):
#         for J in JOB:
#             problem+=c[E,J]>=c[E-1,J]+lp.lpSum(x[(J,E,L)]*p[E-1][L-1][J-1] for L in M[E-1])
#
#     for E in ET:
#         for L in M[E-1]:
#             for J in JOB:
#                 for K in range(J+1,job+1):
#                     problem+=c[E,J]>=c[E,K]+p[E-1][L-1][J-1]-Q*(2-x[(J,E,L)]-x[(K,E,L)]+y[(J,K,E)])
#
#
#     for E in ET:
#         for L in M[E-1]:
#             for J in JOB:
#                 for K in range(J+1,job+1):
#                     problem+=c[E,K]>=c[E,J]+p[E-1][L-1][K-1]-Q*(3-x[(J,E,L)]-x[(K,E,L)]-y[(J,K,E)])
#
#     problem.solve()
#     for E in ET:
#         for L in M[E-1]:
#             for J in JOB:
#                 if x[(J,E,L)].varValue == 1:
#                     print(f'X[{J},{E},{L}]=1')
#
#     for E in ET:
#         for J in JOB:
#             print(f'c[{E},{J}]={c[E,J].varValue}')
#
#     for E in ET:
#         SIGMA=[]
#         for J in JOB:
#             for L in M[E-1]:
#                 if x[(J,E,L)].varValue==1:
#                     SIGMA.append([c[E,J].varValue-p[E-1][L-1][J-1],c[E,J].varValue,J,L])
#         SIGMA=np.array(SIGMA)
#         SIGMA=SIGMA.T
#         for i in range(len(SIGMA[0, :]) - 1):
#             for j in range(i + 1, len(SIGMA[0, :])):
#                 if SIGMA[0, i] > SIGMA[0, j]:
#                     SIGMA[:, [i, j]] = SIGMA[:, [j, i]]
#
#         sigma.append(SIGMA.tolist())
#
#
#     return sigma





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







