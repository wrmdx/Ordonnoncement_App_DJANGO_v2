import os.path
import os
import random
import copy
from .tests import MILP_None,MILP_JOB
from .tests5 import Jackson
from django.shortcuts import render,redirect
import numpy as np
import matplotlib.pyplot as plt



def home_page(request):
    return render(request,'home_page.html')


def home2(request):
    if request.method=='POST':
        if request.POST.get('type')=='flowshop':
            return redirect('flowshop')
        else:
            return redirect('jobform1')
    else:
        return render(request,'TypeForm.html',locals())

def home(request):     #Flowhshop
    if request.method=='POST':
        rows=int(request.POST.get('rows'))
        columns = int(request.POST.get('columns'))
        cont=request.POST.get('contrainte')
        return redirect('flowform2',rows=rows,columns=columns,cont=cont)
    else:
        return render(request,'FlowForm.html',locals())

def flowform2(request,rows,columns,cont):
    rows=int(rows)
    columns=int(columns)
    y=0
    if 'SDST' in cont:
        y=1
    if request.method=='POST':
        r=[]
        R=[]
        d=[]
        S=[]
        crit=request.POST.get('critere')
        for j in range(columns):
            R.append(str(request.POST.get('r_'+str(j+1))))
            d.append(str(request.POST.get('d_'+str(j+1))))
        for i in range(rows):
            for j in range(columns):
                n=str(i+1)+str(j+1)
                v=request.POST.get(n)
                r.append(str(v))
        if 'SDST' in cont:
            for l in range(rows):
                for i in range(columns):
                    for j in range(columns):
                        S.append(str(request.POST.get('S_'+str(l+1)+str(i+1)+str(j+1))))

            if crit!='GA':
                return redirect('showgantt',M=','.join(r),r=rows,c=columns,crit=crit,cont=cont,R=','.join(R),d=','.join(d),S=','.join(S),N=0,NG=0,Pm=0)
            else:
                return redirect('GAform', M=','.join(r), r=rows, c=columns, crit=crit, cont=cont, R=','.join(R),d=','.join(d), S=','.join(S))
        else:
            if crit=='GA':
                return redirect('GAform', M=','.join(r), r=rows, c=columns, crit=crit, cont=cont, R=','.join(R),d=','.join(d), S='-1')
            else:
                return redirect('showgantt', M=','.join(r), r=rows, c=columns, crit=crit, cont=cont, R=','.join(R),d=','.join(d), S='-1',N=0,NG=0,Pm=0)

    else:
        rows=range(1,int(rows)+1)
        columns=range(1,int(columns)+1)
        return render(request,'FlowForm2.html',{'rows':rows,'columns':columns,'SDST':y})

def GA_form(request,M,r,c,crit,cont,R,d,S):
    if request.method=='POST':
        N=int(request.POST.get('N'))
        NG=int(request.POST.get('NG'))
        Pm=float(request.POST.get('Pm'))
        return redirect('showgantt',M=M,r=r,c=c,crit=crit,cont=cont,R=R,d=d,S=S,N=N,NG=NG,Pm=Pm)
    else:
        return render(request,'GAForm.html')


import numpy as np
import matplotlib.pyplot as plt
import os
import random


def gantt(N, r, c, crit, cont, R, d, T, Np, NG, Pm):
    r = int(r)
    C = int(c)
    R = R.split(',')
    d = d.split(',')
    N = N.split(',')

    M = []
    S = []

    # Data preprocessing
    for i in range(C):
        R[i] = int(R[i])
        d[i] = int(d[i])
    for i in range(r):
        row = []
        for j in range(C):
            row.append(int(N[i * C + j]))
        M.append(row)

    if T != '-1':
        T = T.split(',')
        for l in range(r):
            matrix = []
            for i in range(C):
                row = []
                for j in range(C):
                    row.append(int(T[l * C * C + i * C + j]))
                matrix.append(row)
            S.append(matrix)
        S = np.array(S)
    else:
        S = np.zeros((r, C, C))

    M = np.array(M)
    R = np.array(R)
    d = np.array(d)

    s = np.zeros((r, C))
    c = np.zeros((r, C))

    if crit in ['LPT', 'SPT']:
        p = np.zeros(C)
        for i in range(len(M[:, 0])):
            p += M[i, :]
        sigma = np.array([p, [i + 1 for i in range(len(M[0, :]))]])
        for i in range(len(sigma[1, :]) - 1):
            for j in range(i + 1, len(sigma[1, :])):
                if sigma[0, i] > sigma[0, j]:
                    sigma[:, [i, j]] = sigma[:, [j, i]]
        sigma = sigma[1, :]
        if crit == 'LPT':
            sigma = sigma[::-1]

    elif crit in ['FIFO', 'LIFO']:
        sigma = np.array([R, [i + 1 for i in range(len(M[0, :]))]])
        for i in range(len(sigma[1, :]) - 1):
            for j in range(i + 1, len(sigma[1, :])):
                if sigma[0, i] > sigma[0, j]:
                    sigma[:, [i, j]] = sigma[:, [j, i]]
        sigma = sigma[1, :]
        if crit == 'LIFO':
            sigma = sigma[::-1]

    elif crit == 'EDD':
        sigma = np.array([d, [i + 1 for i in range(len(M[0, :]))]])
        for i in range(len(sigma[1, :]) - 1):
            for j in range(i + 1, len(sigma[1, :])):
                if sigma[0, i] > sigma[0, j]:
                    sigma[:, [i, j]] = sigma[:, [j, i]]
        sigma = sigma[1, :]

    elif crit == 'CDS':
        s, c, sigma = gantt_CDS(s, c, M, R, S)

    elif crit == 'MILP':
        s, c, sigma = LINGO_None(s, c, M, R, cont, S)

    elif crit == 'GA':
        s, c, sigma = GA(s, c, M, cont, R, S, Np, NG, Pm)

    # Gantt chart calculation
    if crit not in ['CDS', 'MILP', 'GA']:
        if cont == 'None' or cont == 'SDST':
            s, c = gantt_None(s, c, M, R, sigma, S)
        elif cont == 'no-idle':
            s, c = gantt_no_idle(s, c, M, R, sigma)
        elif 'no-wait' in cont:
            s, c = gantt_no_wait(s, c, M, R, S, sigma)
        elif 'blocking' in cont:
            s, c = gantt_blocking(s, c, M, R, S, sigma)

    # Visualization with enhanced style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 6))

    ticks = []
    K = (0.3 / len(M[:, 0])) * max(c[-1, :])

    for j in range(len(M[0, :])):
        col = "#{:02x}{:02x}{:02x}".format(random.randint(100, 255), random.randint(50, 255), random.randint(0, 150))
        gray = "#{:02x}{:02x}{:02x}".format(128, 128, 128)
        for i in range(len(M[:, 0])):
            ax.barh(K * i + 1, c[i, int(sigma[j] - 1)] - s[i, int(sigma[j] - 1)], left=s[i, int(sigma[j] - 1)],
                    height=K / 2, align='center', alpha=1, color=col)
            ax.text((c[i, int(sigma[j] - 1)] + s[i, int(sigma[j] - 1)]) / 2, K * i + 1, f'J{int(sigma[j])}',
                    ha='center', va='center', color='black', fontsize=12)
            if T != '-1':
                if j == 0:
                    ax.barh(K * i + 1, S[i, int(sigma[j] - 1), int(sigma[j] - 1)],
                            left=s[i, int(sigma[j] - 1)] - S[i, int(sigma[j] - 1), int(sigma[j] - 1)], height=K / 2,
                            align='center', alpha=1, color=gray)
                    ax.text(s[i, int(sigma[j] - 1)] - S[i, int(sigma[j] - 1), int(sigma[j] - 1)] / 2, K * i + 1,
                            f'S{int(i + 1)}{int(sigma[j])}{int(sigma[j])}', ha='center', va='center',
                            color='black', fontsize=12)
                else:
                    ax.barh(K * i + 1, S[i, int(sigma[j - 1] - 1), int(sigma[j] - 1)],
                            left=s[i, int(sigma[j] - 1)] - S[i, int(sigma[j - 1] - 1), int(sigma[j] - 1)], height=K / 2,
                            align='center', alpha=1, color=gray)
                    ax.text(s[i, int(sigma[j] - 1)] - S[i, int(sigma[j - 1] - 1), int(sigma[j] - 1)] / 2,
                            K * i + 1, f'S{int(i + 1)}{int(sigma[j - 1])}{int(sigma[j])}', ha='center',
                            va='center', color='black', fontsize=12)

    # Labels and grid
    ax.set_xlabel('Time')
    ax.set_title('Gantt Chart')
    Cmax = max(c[-1, :])
    ax.text(max(c[-1, :]) / 3, (2 / 5) * max(c[-1, :]), f'Cmax={int(Cmax)}', color='black')
    labels = []
    for i in range(len(M[:, 0])):
        ticks.append(K * i + 1)
        labels.append(f'M{i + 1}')
    plt.yticks(ticks, labels)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    plt.axis('equal')

    # Metrics calculations
    TFT = np.sum(c[-1, :])  # Total Flow Time
    CT = c[-1, :]  # Completion Times
    TT = np.mean(CT)  # Mean Flow Time
    TE = np.maximum(CT - np.array(d), 0)  # Tardiness
    EMAX = np.max(TE)  # Maximum Tardiness
    TFR = np.sum(TE)  # Total Tardiness
    TAR = np.mean(TE)  # Mean Tardiness

    metrics = {
        'Cmax': max(c[-1, :]),
        'TFT': float(TFT),
        'TT': float(TT),
        'EMAX': float(EMAX),
        'TFR': float(TFR),
        'TAR': float(TAR)
    }

    # Save the plot
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'images/gantt_img.png')
    delete_images()  # Assuming a function to delete old images exists
    plt.savefig(plot_path)
    plt.close()

    return metrics


def showGantt(request,M,r,c,crit,cont,R,d,S,N=0,NG=0,Pm=0):
    metrics = gantt(M,r,c,crit,cont,R,d,S,N,NG,Pm)
    return render(request, 'GanttChart.html', {
        'metrics': metrics,
        'image_url': metrics['Cmax']  # Keep backward compatibility
    })




def delete_images():
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static','images')
    for file_name in os.listdir(plot_path):
        os.remove(os.path.join(plot_path,file_name))


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


def gantt_no_idle(s,c,M,R,sigma):
    d = [0]
    for k in range(1, len(M[0, :])):
        a = 0
        for l in range(k):
            a += M[0, int(sigma[l] - 1)]
        d.append(R[int(sigma[k] - 1)] - a)

    c[0, int(sigma[0] - 1)] = R[int(sigma[0] - 1)] + max(d) + M[0, int(sigma[0] - 1)]
    s[0, int(sigma[0] - 1)] = c[0, int(sigma[0] - 1)] - M[0, int(sigma[0] - 1)]
    for j in range(1, len(M[0, :])):
        c[0, int(sigma[j] - 1)] = c[0, int(sigma[j - 1] - 1)] + M[0, int(sigma[j] - 1)]
        s[0, int(sigma[j] - 1)] = c[0, int(sigma[j] - 1)] - M[0, int(sigma[j] - 1)]
    for i in range(1, len(M[:, 0])):
        d = [0]
        for k in range(1, len(M[0, :])):
            a = 0
            b = 0
            for l in range(1, k + 1):
                a += M[i - 1, int(sigma[l] - 1)]
            for l in range(0, k):
                b += M[i, int(sigma[l] - 1)]
            d.append(a - b)
        c[i, int(sigma[0] - 1)] = c[i - 1, int(sigma[0] - 1)] + max(d) + M[i, int(sigma[0] - 1)]
        s[i, int(sigma[0] - 1)] = c[i, int(sigma[0] - 1)] - M[i, int(sigma[0] - 1)]
    for i in range(1, len(M[:, 0])):
        for j in range(1, len(M[0, :])):
            c[i, int(sigma[j] - 1)] = c[i, int(sigma[j - 1] - 1)] + M[i, int(sigma[j] - 1)]
            s[i, int(sigma[j] - 1)] = c[i, int(sigma[j] - 1)] - M[i, int(sigma[j] - 1)]
    return (s,c)


def gantt_no_wait(s,c,M,R,S,sigma):
    d=[0]
    for k in range(1,len(M[:, 0])):
        a=0
        for l in range(k):
            a+=M[l,int(sigma[0] - 1)]
        d.append(S[k,int(sigma[0]-1),int(sigma[0]-1)]-S[0,int(sigma[0]-1),int(sigma[0]-1)]-a)
    c[0, int(sigma[0] - 1)] = S[0, int(sigma[0] - 1),int(sigma[0] - 1)] + M[0, int(sigma[0] - 1)]+max(d)
    s[0, int(sigma[0] - 1)]=c[0, int(sigma[0] - 1)]-M[0,int(sigma[0] - 1)]

    for i in range(1,len(M[:,0])):
        c[i, int(sigma[0] - 1)] = c[i - 1, int(sigma[0] - 1)] + M[i, int(sigma[0] - 1)]
        s[i, int(sigma[0] - 1)] = c[i, int(sigma[0] - 1)] - M[i, int(sigma[0] - 1)]
    for j in range(1, len(M[0, :])):
        d = [0]
        for k in range(1, len(M[:, 0])):
            a = 0
            b = 0
            for l in range(1, k + 1):
                a += M[l, int(sigma[j - 1] - 1)]
            for l in range(0, k):
                b += M[l, int(sigma[j] - 1)]
            d.append(a - b+S[k,int(sigma[j-1]-1),int(sigma[j]-1)]-S[0,int(sigma[j-1]-1),int(sigma[j]-1)])
        c[0, int(sigma[j] - 1)] = c[0, int(sigma[j - 1] - 1)] + max(d) + M[0, int(sigma[j] - 1)]+S[0,int(sigma[j-1]-1),int(sigma[j]-1)]
        s[0, int(sigma[j] - 1)] = c[0, int(sigma[j] - 1)] - M[0, int(sigma[j] - 1)]
    for i in range(1, len(M[:, 0])):
        for j in range(1, len(M[0, :])):
            c[i, int(sigma[j] - 1)] = c[i - 1, int(sigma[j] - 1)] + M[i, int(sigma[j] - 1)]
            s[i, int(sigma[j] - 1)] = c[i, int(sigma[j] - 1)] - M[i, int(sigma[j] - 1)]
    return (s,c)

def gantt_blocking(s,c,M,R,S,sigma):
    print(S)
    c[0,int(sigma[0]-1)]=M[0,int(sigma[0]-1)]+S[0,int(sigma[0]-1),int(sigma[0]-1)]
    s[0, int(sigma[0] - 1)]=c[0,int(sigma[0]-1)]-M[0,int(sigma[0]-1)]

    for i in range(1,len(M[:,0])):
        c[i,int(sigma[0]-1)]=max(c[i-1,int(sigma[0]-1)],S[i,int(sigma[0]-1),int(sigma[0]-1)])+M[i,int(sigma[0]-1)]
        s[i, int(sigma[0] - 1)] = c[i, int(sigma[0] - 1)] - M[i, int(sigma[0] - 1)]



    for j in range(1, len(M[0, :])):
        for i in range(len(M[:,0])):
            if i==0:
                c[0, int(sigma[j] - 1)] = max(c[0, int(sigma[j - 1] - 1)],c[1, int(sigma[j - 1] - 1)] - M[1, int(sigma[j - 1] - 1)]) + S[0, int(sigma[j - 1] - 1), int(sigma[j] - 1)] + M[0, int(sigma[j] - 1)]
                s[0, int(sigma[j] - 1)] = c[0, int(sigma[j] - 1)] - M[0, int(sigma[j] - 1)]
            elif i==len(M[:,0])-1:
                c[len(M[:, 0]) - 1, int(sigma[j] - 1)] = max(c[len(M[:, 0]) - 2, int(sigma[j] - 1)],c[len(M[:, 0]) - 1, int(sigma[j - 1] - 1)] + S[len(M[:, 0]) - 1, int(sigma[j - 1] - 1), int(sigma[j] - 1)]) + M[len(M[:, 0]) - 1, int(sigma[j] - 1)]
                s[len(M[:, 0]) - 1, int(sigma[j] - 1)] = c[len(M[:, 0]) - 1, int(sigma[j] - 1)] - M[len(M[:, 0]) - 1, int(sigma[j] - 1)]

            else:
                c[i,int(sigma[j]-1)]=max(c[i-1,int(sigma[j]-1)],c[i,int(sigma[j-1]-1)]+S[i,int(sigma[j-1]-1),int(sigma[j]-1)],c[i+1,int(sigma[j-1]-1)]-M[i+1,int(sigma[j-1]-1)]+S[i,int(sigma[j-1]-1),int(sigma[j]-1)])+M[i,int(sigma[j]-1)]
                s[i, int(sigma[j] - 1)]=c[i,int(sigma[j]-1)]-M[i,int(sigma[j]-1)]



    return (s,c)



def gantt_CDS(s,c,M,R,S):
    SIGMA=np.array([i+1 for i in range(len(M[0,:]))])
    M12=np.zeros((3,len(M[0,:])))
    for k in range(0,len(M[:,0])-1):
        nU = 0
        nV = 0
        U=[]
        V=[]
        M12[0]=sum(M[0:k+1,:])
        M12[1]=sum(M[len(M[:,0])-(k+1):len(M[:,0]),:])
        M12[2]=np.array([i+1 for i in range(len(M[0,:]))])
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
        s,c=gantt_None(s,c,M,R,sigma,S)
        if k==0 or max(c[-1,:])<Copt:
            Copt=max(c[-1,:])
            SIGMA=sigma
    S,C=gantt_None(s,c,M,R,SIGMA,S)
    return (S,C,SIGMA)





def LINGO_None(s,c,M,R,cont,S):
    sigma=MILP_None(M,cont,S)
    if cont=='None' or cont=='SDST':
        s,c=gantt_None(s,c,M,R,sigma,S)
    elif cont=='no-idle':
        s,c=gantt_no_idle(s,c,M,R,sigma)
    elif 'no-wait' in cont:
        s,c=gantt_no_wait(s,c,M,R,S,sigma)
    return (s,c,sigma)


def fact(n):
    if n==0:
        return 1
    return n*fact(n-1)

def GA(s,c,M,cont,R,S,N,NG,Pm):
    N=int(N)
    NG=int(NG)
    Pm=float(Pm)
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
    while n<min(N,fact(len(pop[0]))):
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
    while g<NG:
        P=[]
        Fit=fitness(M,pop,R,S)
        SF=sum(Fit)
        for i in range(len(pop)):
            P.append((Fit[i]+1)/(SF+1))

        pop=generate(pop,P,Pm)
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
    return (s,c,sigmab)


def generate(pop,P,Pm):
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
                    newpop.append(LOX(sigma1,sigma2,Pm))
                    m=1
                    n+=1
    return newpop



def LOX(sigma1,sigma2,Pm):


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
    if random.random()<=Pm:
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


def jobform1(request):
    if request.method=='POST':
        J=int(request.POST.get('J'))
        m=int(request.POST.get('m'))
        return redirect('jobform2',J=J,m=m)
    else:
        return render(request,'JobForm1.html')

def jobform2(request,J,m):
    if request.method=='POST':
        J=int(J)
        o=[]
        for j in range(J):
            o.append(request.POST.get(f'o{j+1}'))
        return redirect('jobform3',J=J,m=m,o=','.join(o))

    else:
        Jobs=range(1,int(J)+1)
        return render(request,'JobForm2.html',{'Jobs':Jobs})

def jobform3(request,J,m,o):
    if request.method=='POST':
        J=int(J)
        O=[]
        o=o.split(',')
        p=[]
        for j in range(J):
            for op in range(int(o[j])):
                O.append(request.POST.get(f'O{j+1}{op+1}'))
                p.append(request.POST.get(f'p{j+1}{op+1}'))
        crit=request.POST.get('critere')
        cont=request.POST.get('contrainte')
        return redirect('showjobgantt',J=J,m=m,o=','.join(o),O=','.join(O),p=','.join(p),crit=crit,cont=cont)


    else:

        N=o.split(',')
        for i in range(len(N)):
            if i==0 or maxO<int(N[i]):
                maxO=int(N[i])
            N[i]=range(1,int(N[i])+1)
        Jobs=range(1,int(J)+1)
        N=zip(Jobs,N)
        Op=range(1,maxO+1)

        return render(request,'JobForm3.html',{'N':N,'Op':Op,'y':int(m)==2})

# def jobgantt(J,M,o,Ol,pl,crit,cont):
#     M=int(M)
#     J=int(J)
#     o=o.split(',')
#     Ol=Ol.split(',')
#     pl = pl.split(',')
#     for i in range(len(o)):
#         o[i]=int(o[i])
#     O=[]
#     p=[]
#     for j in range(J):
#         v=[]
#         h=[]
#         for op in range(o[j]):
#             v.append(int(Ol[sum(o[0:j])+op]))
#             h.append(int(pl[sum(o[0:j])+op]))
#         O.append(v)
#         p.append(h)
#
#     if crit=='Jackson':
#         sigma=Jackson(O,p)
#
#     c=np.zeros((M,J))
#     s=np.zeros((M,J))
#
#     for m in range(M):
#         if sigma[m,0]!=0:
#             c[m,int(sigma[m,0]-1)]=p[int(sigma[m,0]-1)][0]
#             s[m, int(sigma[m, 0] - 1)]=0
#         for j in range(1,len(sigma[m,:])):
#             if O[int(sigma[m,j]-1)][0]==m+1:
#                 if sigma[m,j] != 0:
#                     c[m,int(sigma[m,j]-1)]=c[m,int(sigma[m,j-1]-1)]+p[int(sigma[m,j]-1)][0]
#                     s[m, int(sigma[m, j] - 1)]=c[m,int(sigma[m,j]-1)]-p[int(sigma[m,j]-1)][0]
#
#
#     for m in range(M):
#         for j in range(len(sigma[m,:])):
#             if len(O[int(sigma[m,j]-1)])!=1:
#                 if O[int(sigma[m,j]-1)][1]==m+1:
#                     if sigma[m,j]!=0:
#                         c[m, int(sigma[m, j] - 1)] = max(c[m, int(sigma[m, j - 1] - 1)],c[1-m,int(sigma[m, j] - 1)]) + p[int(sigma[m, j] - 1)][1]
#                         s[m, int(sigma[m, j] - 1)]=c[m, int(sigma[m, j] - 1)]-p[int(sigma[m, j] - 1)][1]
#
#
#     fig, ax = plt.subplots()
#     ticks = []
#     K = (0.3 / M) * np.max(c)
#     for j in range(J):
#         col = "#{:02x}{:02x}{:02x}".format(random.randint(100, 255), random.randint(50, 255), random.randint(0, 150))
#         for op in range(o[j]):
#             i=int(O[j][op]-1)
#             ax.barh(K * i + 1, c[i, j] - s[i, j], left=s[i,j],height=K / 2, align='center', alpha=1, color=col)
#             ax.text((c[i, j] + s[i, j]) / 2, K * i + 1, f'J{j+1}',ha='center', va='center', color='black', fontsize=12)
#
#
#     ax.set_xlabel('time')
#     ax.set_title('Gantt Chart')
#     Cmax = np.max(c)
#     ax.text(np.max(c) / 3, (2 / 5) * np.max(c), f'Cmax={int(Cmax)}', color='black')
#     labels = []
#     for i in range(M):
#         ticks.append(K * i + 1)
#         labels.append(f'M{i + 1}')
#     plt.yticks(ticks, labels)
#     ax.yaxis.grid(True)
#     ax.xaxis.grid(True)
#     plt.axis('equal')
#
#     plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'images/gantt_img.png')
#     delete_images()
#     plt.savefig(plot_path)
#     plt.close()
#     return np.max(c)


# def MILPjobgantt(J,Ol,pl,op,M):
#     M = int(M)
#     J=int(J)
#     op = op.split(',')
#     Ol = Ol.split(',')
#     pl = pl.split(',')
#     for i in range(len(op)):
#         op[i] = int(op[i])
#
#
#     O = []
#     p = []
#     for j in range(J):
#         v = []
#         h = []
#         for o in range(op[j]):
#             v.append(int(Ol[sum(op[0:j]) + o]))
#             h.append(int(pl[sum(op[0:j]) + o]))
#         O.append(v)
#         p.append(h)
#
#     c=MILP_JOB(O,p,op,M)
#     s=[]
#     for j in range(len(p)):
#         v=[]
#         if j==0 or Cmax<c[j][-1]:
#             Cmax=c[j][-1]
#         for o in range(op[j]):
#             v.append(c[j][o]-p[j][o])
#         s.append(v)
#
#     fig, ax = plt.subplots()
#     ticks = []
#     K = (0.3 / M) * Cmax
#     for j in range(len(p)):
#         col = "#{:02x}{:02x}{:02x}".format(random.randint(100, 255), random.randint(50, 255), random.randint(0, 150))
#         for o in range(op[j]):
#             i = int(O[j][o] - 1)
#             ax.barh(K * i + 1, c[j][o] - s[j][o], left=s[j][o], height=K / 2, align='center', alpha=1, color=col)
#             ax.text((c[j][o] + s[j][o]) / 2, K * i + 1, f'J{j + 1}', ha='center', va='center', color='black',fontsize=12)
#
#     ax.set_xlabel('time')
#     ax.set_title('Gantt Chart')
#
#     ax.text(Cmax / 3, (2 / 5) * Cmax, f'Cmax={int(Cmax)}', color='black')
#     labels = []
#     for i in range(M):
#         ticks.append(K * i + 1)
#         labels.append(f'M{i + 1}')
#     plt.yticks(ticks, labels)
#     ax.yaxis.grid(True)
#     ax.xaxis.grid(True)
#     plt.axis('equal')
#
#     plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'images/gantt_img.png')
#     delete_images()
#     plt.savefig(plot_path)
#     plt.close()
#     return Cmax

def MILPjobgantt(J, Ol, pl, op, M):
    M = int(M)
    J = int(J)
    op = op.split(',')
    Ol = Ol.split(',')
    pl = pl.split(',')
    for i in range(len(op)):
        op[i] = int(op[i])

    O = []
    p = []
    for j in range(J):
        v = []
        h = []
        for o in range(op[j]):
            v.append(int(Ol[sum(op[0:j]) + o]))
            h.append(int(pl[sum(op[0:j]) + o]))
        O.append(v)
        p.append(h)

    c = MILP_JOB(O, p, op, M)
    s = []
    Cmax = 0
    completion_times = np.zeros(J)  # Array to store completion time of each job

    for j in range(len(p)):
        v = []
        completion_times[j] = c[j][-1]  # Store completion time for job j
        if j == 0 or Cmax < c[j][-1]:
            Cmax = c[j][-1]
        for o in range(op[j]):
            v.append(c[j][o] - p[j][o])
        s.append(v)

    # Calculate metrics
    TFT = np.sum(completion_times)  # Total Flow Time
    TT = np.mean(completion_times)  # Mean Flow Time

    # Using mean completion time as reference for tardiness
    due_dates = np.full(J, np.mean(completion_times))
    TE = np.maximum(completion_times - due_dates, 0)  # Tardiness
    EMAX = np.max(TE)  # Maximum Tardiness
    TFR = np.sum(TE)  # Total Tardiness
    TAR = np.mean(TE)  # Mean Tardiness

    # Visualization
    fig, ax = plt.subplots()
    ticks = []
    K = (0.3 / M) * Cmax
    for j in range(len(p)):
        col = "#{:02x}{:02x}{:02x}".format(random.randint(100, 255), random.randint(50, 255), random.randint(0, 150))
        for o in range(op[j]):
            i = int(O[j][o] - 1)
            ax.barh(K * i + 1, c[j][o] - s[j][o], left=s[j][o], height=K / 2, align='center', alpha=1, color=col)
            ax.text((c[j][o] + s[j][o]) / 2, K * i + 1, f'J{j + 1}', ha='center', va='center', color='black',
                    fontsize=12)

    ax.set_xlabel('time')
    ax.set_title('Gantt Chart')
    ax.text(Cmax / 3, (2 / 5) * Cmax, f'Cmax={int(Cmax)}', color='black')
    labels = []
    for i in range(M):
        ticks.append(K * i + 1)
        labels.append(f'M{i + 1}')
    plt.yticks(ticks, labels)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    plt.axis('equal')

    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'images/gantt_img.png')
    delete_images()
    plt.savefig(plot_path)
    plt.close()

    return {
        'Cmax': float(Cmax),
        'TFT': float(TFT),
        'TT': float(TT),
        'EMAX': float(EMAX),
        'TFR': float(TFR),
        'TAR': float(TAR)
    }


def jobgantt(J, M, o, Ol, pl, crit, cont):
    # Data processing
    M = int(M)
    J = int(J)
    o = o.split(',')
    Ol = Ol.split(',')
    pl = pl.split(',')
    for i in range(len(o)):
        o[i] = int(o[i])
    O = []
    p = []
    for j in range(J):
        v = []
        h = []
        for op in range(o[j]):
            v.append(int(Ol[sum(o[0:j]) + op]))
            h.append(int(pl[sum(o[0:j]) + op]))
        O.append(v)
        p.append(h)

    if crit == 'Jackson':
        sigma = Jackson(O, p)

    # Calculations
    c = np.zeros((M, J))
    s = np.zeros((M, J))

    for m in range(M):
        if sigma[m, 0] != 0:
            c[m, int(sigma[m, 0] - 1)] = p[int(sigma[m, 0] - 1)][0]
            s[m, int(sigma[m, 0] - 1)] = 0
        for j in range(1, len(sigma[m, :])):
            if O[int(sigma[m, j] - 1)][0] == m + 1:
                if sigma[m, j] != 0:
                    c[m, int(sigma[m, j] - 1)] = c[m, int(sigma[m, j - 1] - 1)] + p[int(sigma[m, j] - 1)][0]
                    s[m, int(sigma[m, j] - 1)] = c[m, int(sigma[m, j] - 1)] - p[int(sigma[m, j] - 1)][0]

    for m in range(M):
        for j in range(len(sigma[m, :])):
            if len(O[int(sigma[m, j] - 1)]) != 1:
                if O[int(sigma[m, j] - 1)][1] == m + 1:
                    if sigma[m, j] != 0:
                        c[m, int(sigma[m, j] - 1)] = max(c[m, int(sigma[m, j - 1] - 1)],
                                                         c[1 - m, int(sigma[m, j] - 1)]) + p[int(sigma[m, j] - 1)][1]
                        s[m, int(sigma[m, j] - 1)] = c[m, int(sigma[m, j] - 1)] - p[int(sigma[m, j] - 1)][1]

    # Calculate metrics
    Cmax = np.max(c)
    completion_times = np.max(c, axis=0)
    TFT = np.sum(completion_times)
    TT = np.mean(completion_times)
    due_dates = np.full(J, np.mean(completion_times))
    TE = np.maximum(completion_times - due_dates, 0)
    EMAX = np.max(TE)
    TFR = np.sum(TE)
    TAR = np.mean(TE)

    # Enhanced visualization
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set background colors
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#ffffff')

    # Calculate bar positions
    ticks = []
    K = (0.3 / M) * np.max(c)

    # Color palette
    colors = plt.cm.Pastel1(np.linspace(0, 1, J))

    # Draw bars
    for j in range(J):
        for op in range(o[j]):
            i = int(O[j][op] - 1)
            ax.barh(K * i + 1,
                    c[i, j] - s[i, j],
                    left=s[i, j],
                    height=K / 2,
                    align='center',
                    color=colors[j],
                    edgecolor='white',
                    alpha=0.9,
                    zorder=3)

            ax.text((c[i, j] + s[i, j]) / 2,
                    K * i + 1,
                    f'J{j + 1}',
                    ha='center',
                    va='center',
                    color='#2c3e50',
                    fontweight='bold',
                    fontsize=10)

    # Style axes and labels
    ax.set_xlabel('Time', fontsize=12, color='#2c3e50', labelpad=10)
    ax.set_title('Gantt Chart', pad=20, fontsize=14, fontweight='bold', color='#37517e')

    # Add Cmax text
    ax.text(np.max(c) / 3,
            (2 / 5) * np.max(c),
            f'Cmax = {int(Cmax)}',
            color='#37517e',
            fontsize=12,
            fontweight='bold',
            bbox=dict(facecolor='white',
                      edgecolor='#37517e',
                      boxstyle='round,pad=0.5',
                      alpha=0.8))

    # Add machine labels
    labels = []
    for i in range(M):
        ticks.append(K * i + 1)
        labels.append(f'M{i + 1}')
    plt.yticks(ticks, labels, fontsize=10)
    plt.xticks(fontsize=10)

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='#cccccc', zorder=1)
    ax.xaxis.grid(True, linestyle='--', alpha=0.7, color='#cccccc', zorder=1)

    # Style spines
    for spine in ax.spines.values():
        spine.set_color('#e0e0e0')
        spine.set_linewidth(1)

    # Adjust layout
    plt.axis('equal')
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'images/gantt_img.png')
    delete_images()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return {
        'Cmax': float(Cmax),
        'TFT': float(TFT),
        'TT': float(TT),
        'EMAX': float(EMAX),
        'TFR': float(TFR),
        'TAR': float(TAR)
    }

def showjobgantt(request, J, m, o, O, p, crit, cont):
    if crit == 'Jackson':
        metrics = jobgantt(J, m, o, O, p, crit, cont)
    elif crit == 'MILP':
        metrics = MILPjobgantt(J, O, p, o, m)

    return render(request, 'GanttChart.html', {'metrics': metrics})
