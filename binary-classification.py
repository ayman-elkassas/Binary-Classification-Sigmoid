import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

path='week_3-ex_2.csv'
data=pd.read_csv(path,header=None,names=['Exam 1','Exam 2','Admitted'])

# print('data = ')
# print(data.head(10))
# print()
# print('data.describe = ')
# print(data.describe())

fig1,ax0=plt.subplots(figsize=(5,5))
ax0.scatter(data['Exam 1'],data['Exam 2'],s=50,c='g',marker='o')
ax0.set_xlabel('Exam 1 Score')
ax0.set_ylabel('Exam 2 Score')

# print("admitted 1= \n",data[data['Admitted'].isin([1])])

# should make this step to show your data
positive=data[data['Admitted'].isin([1])]
negative=data[data['Admitted'].isin([0])]

fig,ax=plt.subplots(figsize=(5,5))
ax.scatter(positive['Exam 1'],positive['Exam 2'],s=50,c='b',marker='o',label='Admitted')
ax.scatter(negative['Exam 1'],negative['Exam 2'],s=50,c='r',marker='x',label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# test sigmoid classification function
nums=np.arange(-10,10,step=1)

# draw function
fig,ax=plt.subplots(figsize=(5,5))
ax.plot(nums,sigmoid(nums),'r')

# add a ones column - this makes the matrix multiplication work out easier
data.insert(0,'Ones',1)

# print('data = \n',data)

# separate X (training data) and y (target variable)
cols=data.shape[1]
X=data.iloc[:,0:cols - 1]
y=data.iloc[:,cols - 1:cols]

# convert to numpy arrays and initialize the parameter array theta
X=np.array(X.values)
y=np.array(y.values)
theta=np.zeros(3)


def cost(theta,X,y):
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=np.matrix(y)
    first=np.multiply(-y,np.log(sigmoid(X * theta.T)))
    second=np.multiply((1 - y),np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))


thiscost=cost(theta,X,y)
print()
print('cost before optimize cost function = ',thiscost)


def gradient(theta,X,y):
    theta=np.matrix(theta)
    X=np.matrix(X)
    y=np.matrix(y)

    parameters=int(theta.ravel().shape[1])
    grad=np.zeros(parameters)

    error=sigmoid(X * theta.T) - y

    for i in range(parameters):
        term=np.multiply(error,X[:,i])
        # grade is the value before subtract value from old theta
        # optimize function complete this step
        grad[i]=np.sum(term) / len(X)

    return grad

# scipy specialist in statistical problems like that
# using scipy opt can optimize function using another function like gradient and return coefficients thetas
# using this way another pure gradient optimize loop inside loop and nums of iterations
# opt.fmin_tnc get the best iterations and return theta vector
# but you can make like regression way loop inside loop
result=opt.fmin_tnc(func=cost,x0=theta,fprime=gradient,args=(X,y))
print('result = \n',result)

costafteroptimize=cost(result[0],X,y)

print()
print('cost after optimize cost function = ',costafteroptimize)
print()


def predict(theta,X):
    probability=sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


# result is tuple that first index is theta vector
theta_min=np.matrix(result[0])

predictions=predict(theta_min,X)
# print(predictions)

correct=[1 if a == b else 0 for (a,b) in zip(predictions,y)]

accuracy=(sum(correct) % len(correct))

print('accuracy = {0}%'.format(accuracy))

plt.show()
