import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import solve

def datagen():
    df = pd.read_csv('economic_freedom.csv')
    df=df.sample(frac=1).reset_index(drop=True)
    df=df.dropna()
    #df.describe()

    count_col=len(df.columns)

    count_replace=0
    count_float_int=0
    count_pb=0

    for i in range(count_col):
        t=df.iloc[0,i]
        if(type(t)==type('a')) :
            count_replace=count_replace+1
            df.iloc[:,i] = df.iloc[:,i].str.replace('$', '', regex=True).str.replace(',', '', regex=True)
            df.iloc[:,i] = df.iloc[:,i].astype(float)
        elif(isinstance(t, (np.floating, float))):
            count_float_int=count_float_int+1
        else: count_pb=count_pb+1
    
    ind_selection=np.array([1,2,3,6,7,8,9,10,11])

    A_=df.iloc[:,ind_selection].to_numpy()

    y_=df.iloc[:,0].to_numpy()
    
    
    return A_,y_


def datatreat(A_,y_,n_train):
    if((int(n_train)==n_train) & (n_train!=1)):
        n,p=A_.shape
        n_test=n-n_train
        
    else:
        n,p=A_.shape
        n_train=int(n*n_train)
        n_test=n-n_train
    
    print("Number of obs:",n)
    print("n train:",n_train)
    print("n test:",n_test)
    print("Number of explicative variables:",p)
    
    A=A_[:n_train]
    y=y_[:n_train]
    
    A_test=A_[n_train:]
    y_test=y_[n_train:]
    mA = A.mean(axis=0)
    sA = A.std(axis=0)
    
    A = (A-mA)/sA
    A_test = (A_test-mA)/sA

    m = y.mean()
    
    y = y-m
    y_test = y_test-m
    
    #print(A.mean(),np.mean(A.std(axis=0)))
    #print(y.mean())
    
    return n,p,n_train,n_test,A,y,A_test,y_test
    
def PCA(A,y):
    n,p = np.shape(A)
    fig, (ax0,ax1) = plt.subplots(nrows=1,ncols=2)
    
    C = A.T@A
    ax0.imshow(C);
    ax0.set_title("Correlation Matrix");
    ax0.set_xlabel("ID Explicative Variable")


    pca=A.T@y
    
    ax1.set_title("PCA")
    ax1.set_xlabel("ID Explicative Variable")
    ax1.bar(np.arange(1,p+1),pca.flatten())
    ax1.axis('tight');
    ax1.set_xticks(np.arange(1,p+1,1));



    fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
    plt.subplots_adjust(left=None, bottom=None, right=2, top=1, wspace=0.2, hspace=None)
    
    
def eigen_val(A,lbd=0,show=0):
    n,p=A.shape
    C=A.T@A/n+lbd*np.eye(p)
    eigen_val=np.flip(np.linalg.eigvalsh(C))
    mu=eigen_val[-1]
    L=eigen_val[0]
    if (show==1):
        plt.plot(eigen_val, '.-');
        plt.title("Eigen values")
        plt.xlabel("ID Explicative Variable")
        print("Eigen values:",np.sort(eigen_val))
        print("\n")
        print('mu =',mu)
        print('L =',L)
        print('mu/L =',mu/L)
    return mu,L

def tau_GD(A,lbd=0):
    mu,L=eigen_val(A,lbd)
    tau_opt=2/(mu+L)
    tau_max=2/L
    return tau_max,tau_opt
 

def solve_ridge(A,y,lbd=0):
    n,p=A.shape
    C=(A.T@A/n+lbd*np.eye(p))/2
    x_star=solve(C, A.T@y/(2*n), assume_a="sym")
    f_star=np.linalg.norm(A@x_star-y)**2/(2*n)+0.5*lbd*np.linalg.norm(x_star)**2
    return x_star,f_star
    
   
def GD(tau,x_0,f,df,n_iter):
    x=np.zeros((n_iter,x_0.size))
    x[0]=x_0
    
    fx=np.full(n_iter,f(x[0]))
    
    tau=tau*np.ones(n_iter)
    
    for i in range(1,n_iter):
        x[i]=x[i-1]-tau[i]*df(x[i-1])
        fx[i]=f(x[i])
        
    return x,fx    

def SGD(tau,x_0,n_size,f,df_i,n_iter,batch_size=1):
    x=np.zeros((n_iter,x_0.size))
    x[0]=x_0
    
    fx=np.full(n_iter,f(x[0]))
    
    tau=tau*np.ones(n_iter)
    
    for i in range(1,n_iter):
        ind=np.random.choice(np.arange(0,n_size), batch_size, replace = False)
        x[i]=x[i-1]-tau[i]*df_i(x[i-1],ind)
        fx[i]=f(x[i])
        
    return x,fx
    
    
def SAGA(tau,x_0,n_size,f,df_i,n_iter):
    p=x_0.size
    x=np.zeros((n_iter,p))
    fx=np.full(n_iter,f(x[0]))
    x[0]=x_0
    
    z=np.zeros((n_size,p))
    for i in range(n_size): z[i]= df_i(x_0,np.array([i]))
    
    tau=tau*np.ones(n_iter)
    
    for i in range(1,n_iter):
        ind=np.random.choice(np.arange(0,n_size), 1, replace = False)
        x[i]=x[i-1]-tau[i]*( df_i(x[i-1],ind)-z[ind]+np.mean(z,axis=0) )
        fx[i]=f(x[i])
        z[ind]=df_i(x[i-1],ind)
    return x,fx   
    
def HB(tau,gamma,x_0,f,df,n_iter):
    x=np.zeros((n_iter,x_0.size))
    x[0]=x_0
    
    fx=np.full(n_iter,f(x[0]))
    
    m=df(x_0)
    
    for i in range(1,n_iter):
        m=gamma*m+(1-gamma)*df(x[i-1])
        x[i]=x[i-1]-tau*m
        fx[i]=f(x[i])
        
    return x,fx 

def GD_ncvx(tau,x_0,f,df,n_iter):
    x=np.zeros((n_iter,x_0.size))
    x[0]=x_0
    
    x_min=np.zeros((n_iter,x_0.size))
    x_min[0]=x_0
    
    fx_min=np.full(n_iter,f(x_0))
    
    fx=np.full(n_iter,f(x_0))
    
    tau=tau*np.ones(n_iter)
    
    for i in range(1,n_iter):
        x[i]=x[i-1]-tau[i]*df(x[i-1])
        fx[i]=f(x[i])
        
        if( np.linalg.norm(df(x[i])) <= np.linalg.norm(df(x_min[i-1])) ):
            x_min[i]=x[i]
        else:
            x_min[i]=x_min[i-1]
            
        fx_min[i]=f(x_min[i])
    return x,fx,x_min,fx_min

def CGD(tau,x_0,n_size,f,df_j,n_iter,block_size=1):
    x=np.zeros((n_iter,x_0.size))
    x[0]=x_0
    
    fx=np.full(n_iter,f(x[0]))
    
    tau=tau*np.ones(n_iter)
    
    for i in range(1,n_iter):
        ind=np.random.choice(np.arange(0,x_0.size), block_size, replace = False)
        x[i]=x[i-1]-tau[i]*df_j(x[i-1],ind)
        fx[i]=f(x[i])
        
    return x,fx

def SCGD(tau,x_0,n_size,f,df_ij,n_iter,batch_size=1,block_size=1):
    x=np.zeros((n_iter,x_0.size))
    x[0]=x_0
    
    fx=np.full(n_iter,f(x[0]))
    
    tau=tau*np.ones(n_iter)
    
    for i in range(1,n_iter):
        ind_i=np.random.choice(np.arange(0,n_size), batch_size, replace = False)
        ind_j=np.random.choice(np.arange(0,x_0.size), block_size, replace = False)
        x[i]=x[i-1]-tau[i]*df_ij(x[i-1],ind_i,ind_j)
        fx[i]=f(x[i])
        
    return x,fx

def PGD(proj,tau,x_0,f,df,n_iter):
    x=np.zeros((n_iter,x_0.size))
    x[0]=x_0
    
    fx=np.full(n_iter,f(x[0]))
    
    tau=tau*np.ones(n_iter)
    
    for i in range(1,n_iter):
        x[i]=proj(x[i-1]-tau[i]*df(x[i-1]))
        fx[i]=f(x[i])        
    return x,fx    

def FWGD(iter_FW,theta,x_0,f,df,n_iter):
    x=np.zeros((n_iter,x_0.size))
    x[0]=x_0
    
    fx=np.full(n_iter,f(x[0]))
        
    stop_cond=0
    
    for i in range(1,n_iter):
        s=iter_FW(x[i-1])
        if(np.dot(df(x[i-1]),s-x[i-1]) >= 0):
            stop_cond=i
            break
    
        theta_k=theta(x[i-1],s,i)
        x[i]=theta_k*s+(1-theta_k)*x[i-1]
        fx[i]=f(x[i])

    if(stop_cond > 0): 
        x[stop_cond:]=x[stop_cond-1]
        fx[stop_cond:]=f(x[stop_cond-1])
    
    return x,fx   