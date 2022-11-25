#!/usr/bin/env python
# coding: utf-8

# In[130]:


#Import liabraries
from cmath import sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from time import process_time_ns
from tkinter.tix import COLUMN
import matplotlib as mpl

from scipy import polyfit


# In[131]:


#matrix multiplication
def multi_matrix(D,F):
    L = [0] * len(D)
    for x in range (len(D)):
        L[x] = [0] * len(F[0])
    for i in range(len(D)):
        for j in range(len(F[0])):
            for l in range(len(F)):
                L[i][j] += D[i][l] * F[l][j]
    return L

#transpose
def transpose(G):
    n = len(G)
    m= len(G[0])
    C = [0] * m
    for x in range (m):
        C[x] = [0] * n
    for i in range(m):
        for j in range(n):
            C[i][j] = G[j][i]
    return C   


#scalar multiplication
def scalar_multi(D,s):
    L = [0] * len(D)
    for x in range (len(D)):
        L[x] = [0] * len(D[0])
    for i in range(len(D)):
        for j in range(len(D[0])):
            L[i][j] += (s)*D[i][j]
    return L

#matrix sum   
def sum_matrix(D,F):
    L = [0] * len(D)
    for x in range (len(D)):
        L[x] = [0] * len(D[0])
    for i in range(len(D)):
        for j in range(len(D[0])):
            L[i][j] += D[i][j]+F[i][j]
    return L


#identity matrix  
def Id_n(t):
    I_n = [0] * t
    for i in range(t):
        I_n[i] = [0] *t
    for i in range(t):
        I_n[i][i] = 1
    return I_n  
 

# inner product A^T P B
def inner(A,B,P):
    return multi_matrix(multi_matrix(transpose(A),P),B)



# Consider the “DataExam.csv” data set. The data is collected by a psychiatrist assigning values for mental retardation and degree of distrust of doctors in 53 newly hospitalised patients. Six months later, a value for the degree of illness of each patient is assigned. There is a claim that the effect of a treatment can be related to these values and indices. Answer the following two questions. 
# Question 3 (100 marks). 
# 1.	(a)  Find the optimal parameters θMLE for the linear regression where the last column (level of illness) is dependent on the other two. The loss function to be considered is the square of residuals. 
# 1.	Week8&9_ MLE and MAP_problem sheet 8.py
# 2.	Week 10_ regression_PSheet5-to_wrok
# 3.	MSDA_Practices
# 4.	Week8&9_cather_to_MAP.py
# 5.	Week8&9_cather_computing errors.py
# 

# In[132]:


# Upload data
data_set = pd.read_csv('/Users/homecomputer/Data/DataExam.csv')


# 1.	(a)  Find the optimal parameters θMLE for the linear regression where the last column (level of illness) is dependent on the other two. The loss function to be considered is the square of residuals. 

# In[133]:


#for ele in file: del ele[0]
# X_1 = Degree of mental retardation
# X_2 = Degree of distrust of doctors
# Y = Degree of illness

data_matrix=data_set.to_numpy()

X_1 = data_matrix[:,1:2]
X_2 = data_matrix[:,2:3]
Y = data_matrix[:,3:4]

#print(data_matrix, X_1, X_2, Y)

n=len(data_matrix)
print(n)


# In[ ]:





# In[134]:


# one matrix
J_row = [1]*n
J = J_row
for i in range(n):
        J[i] = [1] *n

one_col = transpose([J_row[0]])
#Print one_col


# In[135]:


# Creating X

X = np.hstack ((one_col,X_1,X_2))


# In[136]:


X


# In[137]:


# regression matrix order one 

Term = np.matmul(transpose(X), X)
Theta_vector = np.matmul(np.matmul(np.linalg.inv(Term), transpose(X)),Y)

print('Coefficent of MLE:', Theta_vector )


# In[138]:


# predicted value of training data

Prediction = multi_matrix(X,Theta_vector)
Prediction


# # 2.(b)  Compute the empirical risk Rempf for the linear regression in (a). 

# In[140]:


#computing the error: 
# 2.(b)  Compute the empirical risk Rempf for the linear regression in (a). 

Residual = sum_matrix(Y,scalar_multi(Prediction,-1))
SSE= inner(Residual,Residual,Id_n(n))
avr_loss_lin = (1/n)*(SSE[0][0])

print('Sigma_mle:', avr_loss_lin)


# # 3.(c)  Compute the MAP parameters θMAP with prior Gaussian distribution N(0,bI) where b = 10 and the noise parameter equal to the average loss. 
# 

# In[141]:


##### MAP 
b = (avr_loss_lin)
sigma = 10

reguliser = b/(sigma**2) 
reguliser


# In[142]:


## computing the paramter for MAP
Theta_MAP= np.matmul(np.matmul(np.linalg.inv(sum_matrix(Term,scalar_multi(Id_n(4),reguliser))), transpose(X)),Y)

Prediction_MAP = multi_matrix(X,Theta_MAP)

Residual_MAP = sum_matrix(Y,scalar_multi(Prediction_MAP,-1))
SSE_MAP= inner(Residual_MAP,Residual_MAP,Id_n(n))
avr_loss_lin_MAP = (1/n)*(SSE_MAP[0][0])

print('sigma_MAP:', avr_loss_lin_MAP)

print('MAP=', Theta_MAP)


# # d)  Compute the empirical risk Rempf for θMAP in (c). 

# In[143]:


#computing the error: 
# 2.(b)  Compute the empirical risk Rempf for the linear regression in (a). 

Residual = sum_matrix(Y,scalar_multi(Prediction,-1))
SSE= inner(Residual,Residual,Id_n(n))
avr_loss_lin = (1/n)*(SSE[0][0])

print('Sigma_mle:', avr_loss_lin)


# In[ ]:





# # Question 4 (100 marks). 
# 1.	(a)  Compute the empirical mean x ̄ and the empirical covariance Σ of the data set. 
# 2.	(b)  Verify whether Σ is a positive definite matrix or not. 
# 3.	(c)  Provide the index of two data points with maximum Mahalanobis distance from each other.
# [Hint: The Mah􏰉alanobis distance of two data points xi and xj is computed using (xi − xj)TΣ−1(xi − xj).] 
# 4.	(d)  Use the method of PCA to project the data points from dimension 3 into dimension 2. Trace your steps carefully and provide the coordinates of the projected data of the data points with index 10 and index 50. 
# 5.	(e)  Consider the Euclidean distance in R3 and provide the index of the data point that has the closest distance to its projected coordinate in the PCA projection in item (d). 
# 

# In[144]:


X.shape, Y.shape,X_1.shape


# In[145]:


data_set.shape


# Definition 4.11. The empirical mean vector is the arithmetic average of the observation for each variable, and it is defined as
# 1 􏰓N
#  ̄x := N
# where xn ∈ R . Similarly empirical covariance matrix d × d is
# 1 􏰓N
# Σ:=N
# d
# i=1
# n=1
# xn,
# T (xn− ̄x)(xn− ̄x) .

# In[146]:


# a) Compute the empirical mean x ̄ for dataset 
# X_1 = Degree of mental retardation
# X_2 = Degree of distrust of doctors
# Y = Degree of illness

X_1_Mean = np.mean(X_1, dtype=np.float64)
X_2_Mean = np.mean(X_2, dtype=np.float64)
Y_Mean = np.mean(Y, dtype=np.float64)

print('Empirical mean of Degree of mental retardation:',X_1_Mean)
print('Empirical mean of Degree of distrust of doctors:',X_2_Mean)
print('Empirical mean of Degree of illness:',Y_Mean)


# In[147]:


# Covaraince of X_1

x = [2.8,3.1, 2.59, 3.36, 2.8, 3.35, 2.99, 2.99, 2.92, 3.23, 3.37, 2.72, 3.47, 2.7, 3.24, 2.65, 3.41, 2.58, 2.81, 2.8, 3.62, 2.74, 3.27, 3.78, 2.9, 3.7, 3.4, 2.63, 2.65, 3.26, 3.15, 2.6, 2.74, 2.72, 3.11, 2.79, 2.9, 2.74, 2.7, 3.08, 2.18, 2.88, 3.04, 3.32, 2.8, 3.29, 3.56, 2.74, 3.06, 2.54, 2.78, 2.81, 3.26]
y = [44, 25, 10, 28, 25, 72, 45, 25, 12, 24, 46, 8, 15, 28, 26, 27, 4, 14, 21, 22, 60, 10, 60, 12, 28, 39, 14, 8, 11, 7, 23, 16,26, 8, 11, 12, 50 ,9, 13, 22, 23, 31, 20, 66, 9, 12, 21, 13, 10, 4, 18, 10, 7]
Cov_X_1 = np.stack((x, y), axis=0)
np.cov(Cov_X_1)


# In[148]:


# Covaraince of X_2

x = [6.1, 5.1, 6, 6.9, 7, 5.6, 6.3, 7.2, 6.9, 6.5, 6.8, 6.6, 8.4, 5.9, 6, 6, 7.6, 6.2, 6, 6.4, 6.8, 8.4, 6.7, 8.3, 5.6, 7.3, 7, 6.9, 5.8, 7.2, 6.5, 6.3, 6.8, 5.9, 6.8, 6.7, 6.7, 5.5, 6.9, 6.3, 6.1, 5.8, 6.8, 7.3, 5.9, 6.8, 8.8, 7.1, 6.9, 6.7, 7.2, 5.2, 6.6]
y = [44, 25, 10, 28, 25, 72, 45, 25, 12, 24, 46, 8, 15, 28, 26, 27, 4, 14, 21, 22, 60, 10, 60, 12, 28, 39, 14, 8, 11, 7, 23, 16,26, 8, 11, 12, 50 ,9, 13, 22, 23, 31, 20, 66, 9, 12, 21, 13, 10, 4, 18, 10, 7]
Cov_X_2 = np.stack((x, y), axis=0)
np.cov(Cov_X_2)


# # 2. (b)  Verify whether Σ is a positive definite matrix or not. 

# In[150]:


X= np.array([data_matrix])
X_t = np.transpose(X)

#print(X_t)
X_1= X_t[1]
X_2= X_t[2]
Y=X_t[3]

#print(X_1,X_2,Y)


# In[151]:


# average
Ave_1 = np.average(X_1)
Ave_2 = np.average(X_2)
Ave_3 = np.average(Y)

Ave = [Ave_1,Ave_2, Ave_3]

Ave


# In[152]:


#zero matrix  
def zero(t,s):
    Z = [0] * t
    for i in range(t):
        Z[i] = [0] *s
    return Z  


# In[153]:


plt.scatter(X_1,X_2, s=80, c="blue")
plt.show()

# plt.scatter(ave_x,ave_y, s=100, c="red")

# New points with mean 0

X_c_1 = X_1 - Ave_1
X_c_2 = X_2 - Ave_2

Y = np.vstack((X_c_1,X_c_2))
Y_t = np.transpose(Y)

#print(transpose([Y_t[0]]),Y_t[0])

C = zero(106,106)

for i in range(n):
    C = C+ np.matmul(transpose([Y_t[i]]),[Y_t[i]])



# In[165]:


#transpose
def transpose(G):
    n = len(G)
    m= len(G[0])
    C = [0] * m
    for x in range (m):
        C[x] = [0] * n
    for i in range(m):
        for j in range(n):
            C[i][j] = G[j][i]
    return C   


# In[166]:


# covarience matrix
Sigma = (1/n)*C

print(Sigma)


# In[156]:


w, v = np.linalg.eig(Sigma)

print(w,v)


# In[157]:


# Proof that matrix is not definite postive

np.linalg.eig((Sigma)<0)


# In[ ]:





# # 3.	(c)  Provide the index of two data points with maximum Mahalanobis distance from each other.
# [Hint: The Mah􏰉alanobis distance of two data points xi and xj is computed using (xi − xj)TΣ−1(xi − xj).] 
# 

# In[ ]:





# # 4.	(d)  Use the method of PCA to project the data points from dimension 3 into dimension 2. Trace your steps carefully and provide the coordinates of the projected data of the data points with index 10 and index 50. 

# In[158]:


#plotting
plt.scatter(X_1,X_2, s=80, c="blue")

plt.scatter(X_c_1,X_c_2, s=20, c="red")

h_1= np.linspace(-1,1,4)
slop = B_1[1]/B_1[0]
line_1=slop*h_1

#plt.plot(h_1,line_1,'g-')
plt.show()

# computing projections t_i

C= np.transpose([B_1])


proj_star = zero(4,2)

for i in range(4):
   proj_star[i]= inner(C,transpose([Y_t[i]]),Id_n(2))[0][0]*B_1 

#print(proj_star)
#plotting the projection


P_1 = transpose(proj_star)[0]
P_2 = transpose(proj_star)[1]

#plt.scatter(X_c_1,X_c_2, s=20, c="red")
plt.scatter(X_c_1,X_c_2, s=20, c="red")
plt.scatter(P_1, P_2, s=10, c="black")
#plt.plot(h_1,line_1,'g-')

plt.show()

# shifting by the average

proj = zero(4,2)
for i in range(4):
   proj[i]= proj_star[i]+ Ave 

for i in range(4):
    print('Final projections of point p_',i+1, ':', proj[i])


plt.scatter(X_1,X_2, s=80, c="blue")



#print(proj_star)
h= np.linspace(1,3,4)
line_1=slop*h+(4-2*slop)

plt.scatter(X_1,X_2, s=80, c="blue")
plt.scatter(X_c_1,X_c_2, s=20, c="red")

plt.scatter(transpose(proj)[0], transpose(proj)[1], s=10, c="black")
#plt.plot(h,line_1,'g-')

plt.show()


# In[159]:


# 
V_1 = v[:,0]
V_2 = v[:,1]
V_3 = v[:,3]
V_4 = v[:,2]

# for details on why we consider it this way
#  please check https://scriptverse.academy/tutorials/python-eigenvalues-eigenvectors.html

C_1= np.transpose([V_1])
C_2= np.transpose([V_2])
#print('project', multi_matrix ([Y_t[0]],C_1)*V_1 + multi_matrix ([Y_t[0]],C_2)*V_2)

proj_star_1 = zero(n,4)
proj_star_2 = zero(n,4)

for i in range(n):
   proj_star_1[i]= inner(C_1,transpose([Y_t[i]]),Id_n(4))[0][0]*V_1 
   proj_star_2[i]= proj_star_1[i]+ inner(C_2,transpose([Y_t[i]]),Id_n(4))[0][0]*V_2
   
#proj_1 = zero(n, 4)
proj_2 = zero(n,4)
for i in range(n):
#   proj_1[i]= proj_star_1[i]+ Ave
   proj_2[i]= proj_star_2[i] + Ave 

    
#print(proj_2)
for i in range(n):
#    print('Final projections dim 1 of point p_',i+1, ':', proj_1[i])
    print('Original data point p_', i+1, ':', X_t[i])
    print('Projection to dim 2 of point p_',i+1, ':', proj_2[i])


#for i in range(n):
#    U = sum_matrix([X_t[i]], scalar_multi([proj_1[i]],-1))
#    W = transpose(U)
#    t = inner(W,W,Id_n(4))
#    print('Distance dim 1 of vector',i+1,':', t[0][0]**(0.5))

for i in range(n):
    U_2 = sum_matrix([X_t[i]], scalar_multi([proj_2[i]],-1))
    W_2 = transpose(U_2)
    t_2 = inner(W_2,W_2,Id_n(4))
    print('Distance dim 2 of vector',i+1,':', t_2[0][0]**(0.5))


plt.scatter(X[0],X[1], s=60, c="black", marker="^")
plt.scatter(transpose(proj_2)[0],transpose(proj_2)[1], s=60, c="blue")
plt.xlim((-30,30))
plt.ylim((-30,30))
plt.show()



# In[160]:


# Eignetvalues  [2.30320782e+02 8.81049662e+00 1.06257370e-01 1.19588404e+00]
# Associated eigenvalues based on the higher value

V_1 = v[:,1]
V_2 = v[:,2]
V_3 = v[:,3]


# In[161]:


# for details on why we consider it this way
#  please check https://scriptverse.academy/tutorials/python-eigenvalues-eigenvectors.html

C_1= np.transpose([V_1])
C_2= np.transpose([V_2])

#print('project', multi_matrix ([Y_t[0]],C_1)*V_1 + multi_matrix ([Y_t[0]],C_2)*V_2)


# In[162]:


proj_star_1 = zero(n,4)
proj_star_2 = zero(n,4)

for i in range(n):
   proj_star_1[i]= inner(C_1,transpose([Y_t[i]]),Id_n(4))[0][0]*V_1 
   proj_star_2[i]= proj_star_1[i]+ inner(C_2,transpose([Y_t[i]]),Id_n(4))[0][0]*V_2
   
#proj_1 = zero(n, 4)
proj_2 = zero(n,4)
for i in range(n):
#   proj_1[i]= proj_star_1[i]+ Ave
   proj_2[i]= proj_star_2[i] + Ave 


# In[ ]:





# In[163]:



#print(proj_2)
for i in range(n):
#    print('Final projections dim 1 of point p_',i+1, ':', proj_1[i])
    print('Original data point p_', i+1, ':', X_t[i])
    print('Projection to dim 2 of point p_',i+1, ':', proj_2[i])


#for i in range(n):
#    U = sum_matrix([X_t[i]], scalar_multi([proj_1[i]],-1))
#    W = transpose(U)
#    t = inner(W,W,Id_n(4))
#    print('Distance dim 1 of vector',i+1,':', t[0][0]**(0.5))

for i in range(n):
    U_2 = sum_matrix([X_t[i]], scalar_multi([proj_2[i]],-1))
    W_2 = transpose(U_2)
    t_2 = inner(W_2,W_2,Id_n(4))
    print('Distance dim 2 of vector',i+1,':', t_2[0][0]**(0.5))



# In[164]:


plt.scatter(X[0],X[1], s=60, c="black", marker="^")
plt.scatter(transpose(proj_2)[0],transpose(proj_2)[1], s=60, c="blue")
plt.xlim((-30,30))
plt.ylim((-30,30))
plt.show()




# # (e)  Consider the Euclidean distance in R3 and provide the index of the data point that has the closest distance to its projected coordinate in the PCA projection in item (d). 
# 
# 

# In[ ]:


import numpy
import perfplot
from scipy.spatial import distance


def linalg_norm(data):
    a, b = data[0]
    return numpy.linalg.norm(a - b, axis=1)


def linalg_norm_T(data):
    a, b = data[1]
    return numpy.linalg.norm(a - b, axis=0)


def sqrt_sum(data):
    a, b = data[0]
    return numpy.sqrt(numpy.sum((a - b) ** 2, axis=1))


def sqrt_sum_T(data):
    a, b = data[1]
    return numpy.sqrt(numpy.sum((a - b) ** 2, axis=0))


def scipy_distance(data):
    a, b = data[0]
    return list(map(distance.euclidean, a, b))


def sqrt_einsum(data):
    a, b = data[0]
    a_min_b = a - b
    return numpy.sqrt(numpy.einsum("ij,ij->i", a_min_b, a_min_b))


def sqrt_einsum_T(data):
    a, b = data[1]
    a_min_b = a - b
    return numpy.sqrt(numpy.einsum("ij,ij->j", a_min_b, a_min_b))


def setup(n):
    a = numpy.random.rand(n, 3)
    b = numpy.random.rand(n, 3)
    out0 = numpy.array([a, b])
    out1 = numpy.array([a.T, b.T])
    return out0, out1


b = perfplot.bench(
    setup=setup,
    n_range=[2 ** k for k in range(22)],
    kernels=[
        linalg_norm,
        linalg_norm_T,
        scipy_distance,
        sqrt_sum,
        sqrt_sum_T,
        sqrt_einsum,
        sqrt_einsum_T,
    ],
    xlabel="len(x), len(y)",
)
b.save("norm.png")
Share
Edit
Follow
edited Oct 25, 2021 at 18:05


# In[ ]:




