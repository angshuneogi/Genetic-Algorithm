import numpy as np
import math
import matplotlib.pyplot as plt

def func(x1,x2):            
    return 1/(1+(x1+x2-2*(x1**2)-(x2**2)+x1*x2))    # Fitness Function

def inv_func(f):
    return 1-(1/f)      # Inverse of the Fitness Function
    
n=20            # String Size of each variable
N=70            # Population Size of Solution Matrix
Pc = 1        # Crossover Probability
Pm = 0.05        # Mutation Probability
Gen=50          # Number Of Generations

Avg_FE = np.zeros([Gen,2])
MM_FE = np.zeros([Gen,3])
Opt_X = np.zeros([Gen,3])
inv_FE = np.zeros([Gen,2])

x1_max = 0.5    # Maximum Value of the Varialble x1
x1_min = 0.0    # Minimum Value of the Varialble x1
x2_max = 0.5    # Maximum Value of the Varialble x2
x2_min = 0.0    # Minimum Value of the Varialble x2

S=np.zeros([N,2*n])         # Solution Matrix
b2=np.zeros([n,1])          # Decoding Vector

X = np.zeros([N,2])
norm_X = np.zeros([N,2])    # Normalized of X
FE=np.zeros([N,1])          # Fitness Evaluation Vector

mut=0                       # Mutation Count
cso=0                       # Crossover Count


for i in range (0,n-1):
    b2[i]=pow(2,n-i-1)      # Binary String Decoding Vector

# Assigning the Initial Fitness Values

for i in range (0,N):
    x1=np.random.randint(2, size=n)
    x2=np.random.randint(2, size=n)
    s=np.concatenate((x1,x2))
    S[i,:]=s
    for j in range (0,n-1):
        X[i,0]=X[i,0]+b2[j]*x1[j]
        X[i,1]=X[i,1]+b2[j]*x2[j]
    norm_X[i,0]=x1_min+((x1_max-x1_min)/(pow(2,n)-1))*X[i,0]
    norm_X[i,1]=x2_min+((x2_max-x2_min)/(pow(2,n)-1))*X[i,1]
    FE[i,0]=func(norm_X[i,0],norm_X[i,1])       # Initial Fitness Evaluation

print("Gen \t\t optimal x1 \t\t optimal x2 \t\t Avg. Fitness \t Inverted Fitness\n")

for g in range (0,Gen):
    
    # Roulette Wheel Selection ----------------------------------------- (Operation 1)
    
    sum1=0
    rws_p=np.zeros([N,1])       # Roulette Wheel Selection Probability
    cuv_FE=np.zeros([N,1])      # Cumilative Fitness Probability Values

    for i in range (0,N):
        sum1=sum1+FE[i,0]
    for i in range (0,N):
        rws_p[i,0]=(FE[i,0]/sum1)
    for i in range (0,N):
        for j in range (0,i+1):
            cuv_FE[i,0]=cuv_FE[i,0]+rws_p[j,0]
            
    MP = np.zeros([N,2*n])      # Mating Pool

    for i in range (0,N):
        rp = np.random.rand()
        for j in range(0,N):
            if(cuv_FE[j,0]>rp):
                MP[i,:]=S[j,:]
                break

    # Two Point CrossOver Operation -------------------------------------- (Operation 2)

    ch1 = np.zeros([1,2*n])     # Children Solution 1
    ch2 = np.zeros([1,2*n])     # Children Solution 2

    for i in range (0,int(N/2)):
        Rc = np.random.rand(1)
        if (Rc<Pc):
            rc1=np.random.randint(1,2*n-1)
            rc2=np.random.randint(1,2*n-1)
            rs1=np.random.randint(0,N-1)
            rs2=np.random.randint(0,N-1)

            ch1[0,:]=MP[rs1,:]
            ch2[0,:]=MP[rs2,:]
            if (rc1>rc2):
                ch1[0,rc2:rc1]=MP[rs2,rc2:rc1]
                ch2[0,rc2:rc1]=MP[rs1,rc2:rc1]
            if (rc2>rc1):
                ch1[0,rc1:rc2]=MP[rs2,rc1:rc2]
                ch2[0,rc1:rc2]=MP[rs1,rc1:rc2]
            MP[rs1,:]=ch1[0,:]
            MP[rs2,:]=ch2[0,:]
            cso=cso+1

    # Mutation Operation -------------------------------------------------- (Operation 3)

    for i in range (0,N):
        RV=np.random.rand(1,2*n)
        for j in range (0,2*n):
            if(RV[0,j]<Pm):
                if(MP[i,j]==0):
                    MP[i,j]=1
                if(MP[i,j]==1):
                    MP[i,j]=0
                mut=mut+1

    # Decoding the fitness Values 
    avg_fe=0
    opt_x1=0
    opt_x2=0
    avg_inv = 0
    FE_new=np.zeros([N,1])       # Updated Fitness Evaluation Vector
    norm_X_new = np.zeros([N,2]) # Updated Normalized Vector X
    X_n=np.zeros([N,2])          # Norm of X      

    for i in range (0,N):
        for j in range (0,2*n):
            if(j<n):
               X_n[i,0]=X_n[i,0]+MP[i,j]*b2[j]
            else:
                X_n[i,1]=X_n[i,1]+MP[i,j]*b2[j-n]
        norm_X_new[i,0]=x1_min+((x1_max-x1_min)/(pow(2,n)-1))*X_n[i,0]
        norm_X_new[i,1]=x2_min+((x2_max-x2_min)/(pow(2,n)-1))*X_n[i,1]
        FE_new[i,0]=func(norm_X_new[i,0],norm_X_new[i,1])
        avg_fe=avg_fe+(1/N)*(FE_new[i,0])
        opt_x1=opt_x1+(1/N)*(norm_X_new[i,0])
        opt_x2=opt_x2+(1/N)*(norm_X_new[i,1])
        avg_inv = avg_inv + (1/N)*(inv_func(FE_new[i,0]))
    MM_FE[g,0]=g+1
    MM_FE[g,1]=max(FE_new)
    MM_FE[g,2]=min(FE_new)
    Opt_X[g,0]=g+1
    Opt_X[g,1]=opt_x1
    Opt_X[g,2]=opt_x2
    Avg_FE[g,0]=g+1
    Avg_FE[g,1]=avg_fe
    inv_FE[g,0]=g+1
    inv_FE[g,1]=math.fabs(avg_inv)
    FE=FE_new
    S=MP
    print(g+1,"\t",opt_x1,"\t",opt_x2,"\t",avg_fe,"\t",math.fabs(avg_inv))


print("No.of Generations | Average Fitness : \n",Avg_FE)
print("No. of Gen | Max Fitness | Min Fitness Values \n ",MM_FE)
print("\nNumber of Times Mutation is performed : ",mut)
print("Number of Cross Over Operations : ",cso)

##plot1 - Maximum | Average | Minimum Fitness Values
plt.plot(MM_FE[:,0],MM_FE[:,1])
plt.plot(Avg_FE[:,0],Avg_FE[:,1])
plt.plot(MM_FE[:,0],MM_FE[:,2])
plt.legend(['Maximum FItness','Average Fitness','Minimum Fitness'])
plt.title('Max - Avg - Min Fitness v/s Number of Generations ')
plt.xlabel('Number of Generations')
plt.ylabel('Fitness Values')
plt.show()

##plot2 - Optimal x1 | Optimal x2
plt.plot(Opt_X[:,0],Opt_X[:,1],Opt_X[:,0],Opt_X[:,2])
plt.legend(['Optimal_x1','Optimal_x2'])
plt.title('Optimal Values of x1 and x2 v/s Number of Generations ')
plt.xlabel('Number of Generations')
plt.ylabel('X1 & X2')
plt.show()

##plot3 - Inverted Fitness Values (Actual Function)
plt.plot(inv_FE[:,0],inv_FE[:,1])
plt.xlabel('Number of Generations')
plt.ylabel('Minimization Function Values (Inverted Fitness Function)')
plt.legend(['Inverted Values'])
plt.title('Inverted Values of Fitness Function v/s No. of Generations ')
plt.show()


















