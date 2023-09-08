#%%
import numpy as np
import matplotlib.pyplot as plt

# load data
data1 = np.load('dataset_1.npy')
data2 = np.load('dataset_2.npy')
print("data1 size:", data1.shape)
print("data2 size:", data2.shape)

# (a)(i)
input1  = data1[:,0]
output1 = data1[:,1]
fig, ax = plt.subplots()
ax.scatter(input1, output1)
ax.set_title("Dataset_1")
#plt.show()

input2  = data2[:,0]
output2 = data2[:,1]
fig, ax = plt.subplots()
ax.scatter(input2, output2)
ax.set_title("Dataset_2")
#plt.show()

#%%
# (a)(ii)
imean1 = np.sum(input1, axis=0) / input1.shape[0]
omean1 = np.sum(output1, axis=0) / output1.shape[0]
input1_ = input1 - imean1
output1_ = output1 - omean1
cor1 = np.dot(input1_, output1_) / (np.sqrt(np.sum(input1_ ** 2, axis=0)) * np.sqrt(np.sum(output1_ ** 2, axis=0)))
print("correlation coefficient 1:", cor1)

imean2 = np.sum(input2, axis=0) / input2.shape[0]
omean2 = np.sum(output2, axis=0) / output2.shape[0]
input2_ = input2 - imean2
output2_ = output2 - omean2
cor2 = np.dot(input2_, output2_) / (np.sqrt(np.sum(input2_ ** 2, axis=0)) * np.sqrt(np.sum(output2_ ** 2, axis=0)))
print("correlation coefficient 2:", cor2)


#%%
# (c)
wc= 1 / np.dot(input1, input1) * np.dot(input1, output1);
Jc = np.sum((input1 * wc- output1) **2, axis=0) / input1.shape[0]
fig, ax = plt.subplots()
ax.scatter(input1, output1)
X = np.linspace(0,10,100)
ax.plot(X,X*wc,'r')
ax.set_title("Dataset_1(c)")
#plt.show()

# %%
# (d)
input = np.vstack((input1, np.ones(input1.shape[0]))).T
wd= np.linalg.pinv(input.T @ input) @ input.T @ output1;
Jd = np.sum((input @ wd - output1) **2, axis=0) / input.shape[0]
fig, ax = plt.subplots()
ax.scatter(input1, output1)
X = np.linspace(0,10,100)
ax.plot(X,np.vstack((X,np.ones(X.shape[0]))).T@wd,'r')
ax.set_title("Dataset_1(d)")
#plt.show()

# %%
# (e)
input = np.vstack((input1 **2, input1, np.ones(input1.shape[0]))).T
we= np.linalg.pinv(input.T @ input) @ input.T @ output1;
Je = np.sum((input @ we - output1) **2, axis=0) / input.shape[0]
fig, ax = plt.subplots()
ax.scatter(input1, output1)
X = np.linspace(0,10,100)
ax.plot(X,np.vstack((X **2, X,np.ones(X.shape[0]))).T@we,'r')
ax.set_title("Dataset_1(e)")
#plt.show()

# %%
def feature_ext(X, number):
    if number  == 0:
        return X
    phi = np.ones(X.shape[0])
    for i in range(1, number+1):
        phi = np.vstack((np.power(X,i), phi))
    return phi.T
def solve_MSE(X,Y):
    if X.ndim == 1:
        return 1 / np.dot(X,X) * np.dot(X,Y)
    return np.linalg.pinv(X.T@X) @ X.T @ Y
def calc_MSE(X,Y,w):
    if X.ndim == 1:
        return np.sum((X*w-Y) **2, axis=0) / X.shape[0]
    return np.sum((X@w-Y)**2, axis=0) / X.shape[0]
def estimate(X,w):
    if X.ndim == 1:
        return X*w
    return X@w
# (f)
extNum = 2
input = input2
output = output2
fig, ax = plt.subplots()
ax.scatter(input, output)
input = feature_ext(input, extNum)
w = solve_MSE(input, output)
J = calc_MSE(input, output, w)
X = np.linspace(0,10,100)
feature = feature_ext(X, extNum)
ax.plot(X, estimate(feature,w),'r')
ax.set_title("Dataset_2(e), MSE: "+ str(J) +", w*: "+ str(w))
plt.show()

# %%
from sklearn.model_selection import KFold
def feature_ext(X, number):
    if number  == 0:
        return X
    phi = np.ones(X.shape[0])
    for i in range(1, number+1):
        phi = np.vstack((np.power(X,i), phi))
    return phi.T
def solve_MSE(X,Y):
    if X.ndim == 1:
        return 1 / np.dot(X,X) * np.dot(X,Y)
    return np.linalg.pinv(X.T@X) @ X.T @ Y
def calc_MSE(X,Y,w):
    if X.ndim == 1:
        return np.sum((X*w-Y) **2, axis=0) / X.shape[0]
    return np.sum((X@w-Y)**2, axis=0) / X.shape[0]
def estimate(X,w):
    if X.ndim == 1:
        return X*w
    return X@w
# (g)
extMaxNum = 5
k = 4
kf = KFold(n_splits=k, shuffle=True, random_state=5)
valMSE = np.zeros((extMaxNum,k))
trainMSE = np.zeros((extMaxNum,k))
for j, (train_idx, test_idx) in enumerate(kf.split(data2)):
    train_data = data2[train_idx, :]
    test_data = data2[test_idx, :]
    train_input = train_data[:, 0]
    train_output = train_data[:, 1]
    test_input = test_data[:,0]
    test_output = test_data[:,1]
    for i in range(1, extMaxNum+1):
        train_input_ext = feature_ext(train_input, i)
        test_input_ext = feature_ext(test_input, i)
        w = solve_MSE(train_input_ext, train_output)
        valMSE[i-1,j] = calc_MSE(test_input_ext, test_output, w)
        trainMSE[i-1,j] = calc_MSE(train_input_ext, train_output, w)
# plot
fig, ax = plt.subplots()
ax.plot(np.array([1,2,3,4,5]), np.log(np.mean(valMSE, axis=1)),'ro-', label='Validation MSE')
ax.plot(np.array([1,2,3,4,5]), np.log(np.mean(trainMSE, axis=1)),'bo-', label='Training MSE')
ax.set_title("k-fold")
ax.set_xlabel("highest exponent of feature extension")
ax.set_ylabel("log of MSE")
ax.legend()
plt.show()
# %%
