#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:53:12 2019

@author: alandanielroblesaguilar
"""

# Program for modeling a stochastic and dynamical game for an advertising model among two firms 
# competing for market share.

from random import uniform

from scipy import *
from numpy import *

from scipy import linalg
import numpy as np

R=zeros((101,4,4))
S=zeros((101,4,4))

Q=zeros((8,101))
Q1=zeros((10,101))

T=zeros((101,101,4,4))
U=zeros((101,101,4,4))
V=zeros((101,101,4,4))

V0=zeros((101,1))
V1=zeros((101,1))
W0=zeros((101,1))
W1=zeros((101,1))

H=zeros((101,4,4))
I=zeros((101,4,4))

PF1=zeros((10,10,10))
PF2=zeros(10)

PG1=zeros((10,10,10))
PG2=zeros(10)

EQ1=zeros((1,8))
# A[] and B[] are vectores of the available actions for firms 1 and 2 respectively
# This actions corresponds to actions  for  advertising  effort by each firm.


A=zeros(4)
B=zeros(4)

PF=zeros(10)
PG=zeros(10)

S1=zeros(23) # Two more than the states number !
F=zeros(10)
G=zeros(10)

M=zeros((101,101,4,4))
N=zeros((4,4))
N1=zeros((4,4))


# Actions  for  advertising  effort Firm 1
A[0]=0.01
A[1]=0.02
A[2]=0.03
A[3]=0.04

# Actions  for  advertising  effort Firm 2
B[0]=0.01
B[1]=0.02
B[2]=0.03
B[3]=0.04




# MAIN PARAMETERS FOR RUNNING THE FULL-INFORMATION CASE:
# ntry=1   
# nstg=25  
# nest=0   
# nfig=3   
# beta1=0.95  
# ns=21  
###############################################################################

# MAIN PARAMETERS FOR RUNNING ONE TRAJECTORY IN CASE OF EMPIRICAL ESTIMATION:
# ntry=1   
# nstg=25  
# nest=7   
# nfig=2   
# beta1=0.75  
# ns=11  
###############################################################################

# MAIN PARAMETERS 
ntry=1   #Number of Trajectories
nstg=25  #Number of Stages of calculation
nest=0   #Number of Estimations
nfig=3   #Number of decimal figures in mixed strategies calculation
beta1=0.95  # beta1 is the discount factor
ns=21    #Number of States
###############################################################################

#OTHER PARAMETERS
pf=0.4 # pf is the value of probability for the Binomial Distribution for Firm 1
pg=0.4 # pg is the value of probability for the Binomial Distribution for Firm 2

si=1.0 # Initial State
sf=0.0 # Final State

nf=10   #Number of values taken for de random variable for Firm 1
fi=0.95 # Initial value
ff=1.05 # Final value

ng=10 #Number of values taken for de random variable for Firm 2
gi=0.95 # Initial value
gf=1.05 # Final value
###############################################################################


inc=(sf-si)/(ns-1)
S1[0]=si-inc
for i in range(ns+1):
    S1[i+1]=si+inc*i

incf=(ff-fi)/(nf-1)
for i in range(nf):
    F[i]=fi+incf*i

incg=(gf-gi)/(ng-1)
for i in range(ng):
    G[i]=gi+incg*i

# PF1[] is a vector of probabilities for Firm 1                   
# PG1[] is the vector of probabilities for Firm 2    

R=payoffs14(A,S1,ns,1) #Payoffs for Firm 1
S=payoffs14(B,S1,ns,2) #Payoffs for Firm 2


t=0

while t<ntry: # "ntry" number of trajectories, each one for a fixed Omega, 
          
    PF2=zeros(10)
    PG2=zeros(10)    
    
    c=0
    z=0
    
    
    while z<10**(nest-1): # "nest",  number of estimations for the Empriric Distribution
        u1 = uniform(0,1)
        c1 = float(pf)/float((1-pf))
        pr1 = ((1-pf)**(nf-1)) 
        F1 = pr1
        for l in range(nf):
            if u1<F1:
                PF2[l]+=1
                break
            pr1=pr1*c1*float(nf-1-l)/float(l+1) 
            F1=F1+pr1               


        u2 = uniform(0,1)
        c2 = float(pg)/float((1-pg))
        pr2 = ((1-pg)**(ng-1)) 
        F2 = pr2
        for i in range(ng):
            if u2<F2:
                PG2[i]+=1
                break
            pr2=pr2*c2*float(ng-1-i)/float(i+1) 
            F2=F2+pr2               


        z+=1

        if z==10**c:

            for m in range(nf):
                PF1[m,c,t]=float(PF2[m])/float((z))
            for j in range(ng):
                PG1[j,c,t]=float(PG2[j])/float((z))
            c+=1

   #PF1[l,nest,t] and PG1[l,nest,t] correspond to calculate of Probabilities for Binomial Distribution in Full-Information case
 
    for l in range(nf):
        PF1[l,nest,t]=(math.factorial(nf-1)/(math.factorial(nf-1-l)* \
         math.factorial(l)))*(pf**l)*((1-pf)**(nf-1-l)) 

    for l in range(ng):
        PG1[l,nest,t]=(math.factorial(ng-1)/(math.factorial(ng-1-l)* \
         math.factorial(l)))*(pg**l)*((1-pg)**(ng-1-l))    


    t+=1                     


# The function tran16(A,B,S1,F,G,PF,PG,ns,nf,ng,k,h) calculates all transition matrices
    #A and B are vectors of actions  for  advertising  effort for each firm.
    #S1 of the states of market share, ns is the number os states.
    #F and G the vectors of values for random variables for both firms.
    #nf and ng the number of random variables for each vector F and G.
    # k is number of estimation for empirical distribution, h is the number of the trajectory. 

def tran16(A,B,S1,F,G,PF,PG,ns,nf,ng,k,h):   
        
    E=zeros((101,101,4,4))
    E1=zeros((101,101))
            
    alfa=0.5
    r=0.0    
          
    for l in range(4):
        for m in range(4):                
            for f in range(nf):
                for g in range(ng):
                    for i in range(ns):
                        r=min(max(S1[i+1]+(1-S1[i+1])*F[f]*A[l]**alfa-S1[i+1]*G[g]*B[m]**alfa,0),1)
                        for j in range(ns):
                            if ((S1[j]+S1[j+1])/2 > r >= (S1[j+1]+S1[j+2])/2):
                                E[i,j,l,m]=E[i,j,l,m]+(PF[f,k,h]*PG[g,k,h])                    

                                             
    return E

# The function payoffs14(A,S1,ns,pl) calculate the payoffs matrices for both firms.
# Considers as arguments the of actions for the firm.
# Includes a vector S1 of the states of market share, ns is the number os states.
# pl=1 are payoffs for firm 1, pl=2 are payoffs for firm 2

def payoffs14(A,S1,ns,pl):
    
    U=zeros((101,4,4))
    U1=zeros((4,4))
    p1=1.2
    p2=1.2
    
    for i in range(ns):        
        for j in range(4):            
            for k in range(4):
                if pl==1:
                    U[i,j,k]=p1*S1[i+1]-A[j]                
                if pl==2:
                    U[i,k,j]=p2*(1-S1[i+1])-A[j] 
                                                                                
    return U

        


#  The function sig13(A,n,u,v,w,x,y,z) corresponds to the calculation of the value
# of a game and the strategies of both firms as a matricial multiplication.
# A is the game matrix, n is the current state, u,v,w are strategies for firm 1; x,y,z 
# are strategies for firm 2.

def sig13(A,n,u,v,w,x,y,z):    
    u1=x*(u*A[n,0,0]+v*A[n,1,0]+w*A[n,2,0]+(1-u-v-w)*A[n,3,0])+ \
       y*(u*A[n,0,1]+v*A[n,1,1]+w*A[n,2,1]+(1-u-v-w)*A[n,3,1]) + \
       z*(u*A[n,0,2]+v*A[n,1,2]+w*A[n,2,2]+(1-u-v-w)*A[n,3,2]) + \
       (1-x-y-z)*(u*A[n,0,3]+v*A[n,1,3]+w*A[n,2,3]+(1-u-v-w)*A[n,3,3])
    return u1

# The function val13(A,B,n,u,v,w,x,y,z) corresponds to the McKelvey formula utilized
# in calculation of Nash Equilibrium.
# A and B are payoffs matrices for both firms 1 and 2 respectively, n is the current state, u,v,w are strategies for firm 1; x,y,z 
# are strategies for firm 2.
    
def val13(A,B,n,u,v,w,x,y,z):
    w1=(max(0,sig13(A,n,1,0,0,x,y,z)-sig13(A,n,u,v,w,x,y,z)))**2+ \
       (max(0,sig13(A,n,0,1,0,x,y,z)-sig13(A,n,u,v,w,x,y,z)))**2+ \
       (max(0,sig13(A,n,0,0,1,x,y,z)-sig13(A,n,u,v,w,x,y,z)))**2+ \
       (max(0,sig13(A,n,0,0,0,x,y,z)-sig13(A,n,u,v,w,x,y,z)))**2+ \
       (max(0,sig13(B,n,u,v,w,1,0,0)-sig13(B,n,u,v,w,x,y,z)))**2+ \
       (max(0,sig13(B,n,u,v,w,0,1,0)-sig13(B,n,u,v,w,x,y,z)))**2+ \
       (max(0,sig13(B,n,u,v,w,0,0,1)-sig13(B,n,u,v,w,x,y,z)))**2+ \
       (max(0,sig13(B,n,u,v,w,0,0,0)-sig13(B,n,u,v,w,x,y,z)))**2

    return w1


# The function  eq12(A,B,n,ns) helps to calculate the minimum values for the McKelvey 
# function, wich are Nash Equilibriums.
# The arguments are the payoffs matrices A and B for both players, n is the current state, ns the number of states.
# nfig is the number of decimal positions in case of mixed strategies.

def eq13(A,B,n,ns,nfig):
    EQ=zeros((1,8))
    t1=0
    a2=0.0
    b2=0.0
    c2=0.0
    d2=0.0
    e2=0.0
    f2=0.0


    s3=0

    for i in range(2):
        for j in range(2-i):
            for k in range(2-i-j):
                for l in range(2):
                    for h in range(2-l):
                        for r in range(2-l-h):                    
                    
                            mc=val13(A,B,n,i,j,k,l,h,r)

                            if mc==0.0:
                                if s3==0:
                                    a2=i
                                    b2=j
                                    c2=k
                                    d2=l
                                    e2=h
                                    f2=r
                                if s3>0 and n<(ns-1)/2:
                                    a2=i
                                    b2=j
                                    c2=k
                                    d2=l
                                    e2=h
                                    f2=r 
                                    
                                s3=s3+1



    if s3==0:

        i1=0.0
        i2=1.0 
        j1=0.0 
        j2=1.0 
        k1=0.0 
        k2=1.0 
        l1=0.0 
        l2=1.0 
        h1=0.0 
        h2=1.0 
        r1=0.0 
        r2=1.0 


        g=0
        t1=0
        e=1.0
        h3=1.0

        a1=0.0
        b1=0.0
        c1=0.0
        d1=0.0
        e1=0.0
        f1=0.0
    
        a2=0.0
        b2=0.0
        c2=0.0
        d2=0.0
        e2=0.0
        f2=0.0    
        
        while g<nfig: # "nfig" corresponds to the precision for the calculation of Nash Equilibrium in decimal figures.
            g+=1
            s=1000.0
            i=i1
            while i <= i2: 
                j=j1
                while j<=min(j2,1.0-i):
                    k=k1
                    while k<=min(k2,1.0-i-j):
                        l=l1
                        while l <= l2:
                            h=h1
                            while h<=min(h2,1.0-l):                        
                                r=r1
                                while r<=min(r2,1.0-l-h):                        
                            
                                    t1+=1                                                
                                    fx=val13(A,B,n,i,j,k,l,h,r)
                                    if fx<s:
                                        s=fx
                                        a2=i
                                        b2=j
                                        c2=k
                                        d2=l                            
                                        e2=h
                                        f2=r                            
                                                      

                                    r+=10**(-g)
                                h+=10**(-g)
                            l+=10**(-g)
                        k+=10**(-g)
                    j+=10**(-g) 
                i+=10**(-g)  
        
        

            r1=max(f2-(10**(-g))/2,0.0)
            r2=min(f2+(10**(-g))/2,1.0)
            h1=max(e2-(10**(-g))/2,0.0)
            h2=min(e2+(10**(-g))/2,1.0)
            l1=max(d2-(10**(-g))/2,0.0)
            l2=min(d2+(10**(-g))/2,1.0)
            k1=max(c2-(10**(-g))/2,0.0)
            k2=min(c2+(10**(-g))/2,1.0)
            j1=max(b2-(10**(-g))/2,0.0)
            j2=min(b2+(10**(-g))/2,1.0)    
            i1=max(a2-(10**(-g))/2,0.0)
            i2=min(a2+(10**(-g))/2,1.0)
        
 
            f3=abs(val13(A,B,n,a1,b1,c1,d1,e1,f1)-val13(A,B,n,a2,b2,c2,d2,e2,f2))
            h3=val13(A,B,n,a2,b2,c2,d2,e2,f2)
            a1=a2
            b1=b2
            c1=c2
            d1=d2
            e1=e2
            f1=f2
        
    Q[0,n]=a2
    Q[1,n]=b2
    Q[2,n]=c2
    Q[3,n]=d2
    Q[4,n]=e2
    Q[5,n]=f2    
    Q[6,n]=sig13(A,n,a2,b2,c2,d2,e2,f2)
    Q[7,n]=sig13(B,n,a2,b2,c2,d2,e2,f2)

    Q1[0,n]=a2
    Q1[1,n]=b2
    Q1[2,n]=c2
    Q1[3,n]=1-a2-b2-c2
    Q1[4,n]=d2
    Q1[5,n]=e2
    Q1[6,n]=f2
    Q1[7,n]=1-d2-e2-f2    
    Q1[8,n]=Q[6,n]
    Q1[9,n]=Q[7,n]
    
    EQ[0,0]=a2
    EQ[0,1]=b2
    EQ[0,2]=c2
    EQ[0,3]=1-a2-b2-c2
    EQ[0,4]=d2
    EQ[0,5]=e2
    EQ[0,6]=f2
    EQ[0,7]=1-d2-e2-f2
 
    
    return EQ


# The output file correspond to a text file with rows for each Trajectory, Estimation and Stage of calculation, 
# the strategies and value of the game for Firms 1 and 2 for each state.
    
file= open("Advertising_Game_T.txt","w")    
    
t=0


while t<ntry: # "ntry", Number of Trajectories


    w=0

    while w<=nest: #"nest", Number of Estimations, The last one corresponds to the Full-Information case.
    
        T=tran16(A,B,S1,F,G,PF1,PG1,ns,nf,ng,w,t)          
    
        z=0
        f3=1.0


        for j in range(ns):    
            V0[j,0]=0.0
            W0[j,0]=0.0


        H=zeros((21,4,4))
        I=zeros((21,4,4))


        while z<nstg: # "nstg", number of stages in the determination of Nash Equilibriums
   
            z+=1
            print "Trajectory",t , "Estimation" ,w, "Stage",z

            for j in range(ns):
                for x in range(4):
                    for y in range(4):
                        sum1=0.0
                        sum2=0.0
                        for u in range(ns):
                            sum1=sum1+V0[u,0]*T[j,u,x,y]
                            sum2=sum2+W0[u,0]*T[j,u,x,y]
                        H[j,x,y]=R[j,x,y]+beta1*sum1
                        I[j,x,y]=S[j,x,y]+beta1*sum2
                
                

                EQ1=eq13(H,I,j,ns,nfig)
                if j>=0:                                       
                    print "State",j,"Strat.:", EQ1

    
    
        
            for i in range(ns):
                V1[i,0]=Q[6,i]
                W1[i,0]=Q[7,i]
    
    



    
            f3=0.0
            for i in range(ns):
                if abs(V0[i,0]-V1[i,0])>f3:
                    f3=abs(V0[i,0]-V1[i,0])

    
            for i in range(ns):
                V0[i,0]=V1[i,0]
                W0[i,0]=W1[i,0]

    
            file.write('% s' %t+ ' , ')
            file.write('% s' %w+ ' , ')
            file.write('% s' %z+ ' , ')

            for i in range(ns):
                file.write('% s' %Q1[0,i]+ ' , ')
                file.write('% s' %Q1[1,i]+ ' , ')
                file.write('% s' %Q1[2,i]+ ' , ')
                file.write('% s' %Q1[3,i]+ ' , ')
                file.write('% s' %Q1[4,i]+ ' , ')
                file.write('% s' %Q1[5,i]+ ' , ')
                file.write('% s' %Q1[6,i]+ ' , ')
                file.write('% s' %Q1[7,i]+ ' , ')
                file.write('% s' %Q1[8,i]+ ' , ')
                file.write('% s' %Q1[9,i]+ ' , ')

            file.write('\n')
    
        w+=1
    t+=1        
    
file.close()