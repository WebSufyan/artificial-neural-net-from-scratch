import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('haberman.csv', header=None)

column = ['age', 'year-19xx', 'num_CancerCells_spread', 'survived_or_no']

df.columns = column

dataset = df.copy()

#quick exploration of dataset

#counting target
# sns.countplot(dataset['survived_or_no'])

# description = dataset.describe()

# # checking correlations between variables
# plt.scatter(dataset['age'], dataset['num_CancerCells_spread'])
# plt.scatter(dataset['num_CancerCells_spread'], dataset['survived_or_no'])
# plt.scatter(dataset['year-19xx'], dataset['num_CancerCells_spread'])

# corr = dataset.corr()

# sns.heatmap(corr, annot = True)

dataset = dataset.drop('year-19xx', 1)


x = dataset.drop('survived_or_no', 1).values
y = dataset['survived_or_no'].values

encoded_y = np.where(y==1, 0, 1)
test_encoded_y = encoded_y[:len(x) - 61]


#split data to train set and test set

test_size = 61
x_train = x[:len(x) - test_size]
y_train = encoded_y[:len(x) - test_size]
x_test = x[len(x) - test_size:]
y_test = encoded_y[len(x) - test_size:]



#one hot encoding target for softmax function
y2 = np.zeros((245, 2))
for i in range(245):
    y2[i, y_train[i]] = 1

#initiating weights
w1 = np.random.randn(2, 5)
b1 = np.random.randn(5)
w2 = np.random.randn(5, 3)
b2 = np.random.randn(3)
w3 = np.random.randn(3, 2)
b3 = np.random.randn(2)

#activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#forward propagation
def feed_forward(x, w1, b1, w2, b2, w3, b3):
    layer1 = np.dot(x, w1) +b1
    sig_layer1 = sigmoid(layer1)
    layer2 = np.dot(sig_layer1, w2) + b2
    sig_layer2 = sigmoid(layer2)
    layer3 = np.dot(sig_layer2, w3) + b3
    expA = np.exp(layer3)
    y_hat = expA / expA.sum(axis = 1, keepdims = True)
    return y_hat, sig_layer1, sig_layer2

#calculate derivatives for backpropagation
def derivative_sig(x):
    return sigmoid(x)*(1 - sigmoid(x))


def w3_deriv(y_hat, y, sig_layer2):
    return np.dot(sig_layer2.T, (y_hat - y))
    

def b3_deriv(y_hat, y):
    return (y_hat - y).sum(axis=0)

def w2_deriv(y_hat, y, sig_layer2, sig_layer1, w3):
    previous_deriv = np.dot((y_hat - y), w3.T) * derivative_sig(sig_layer2)
    return np.dot(sig_layer1.T, previous_deriv)

def b2_deriv(y_hat, y, sig_layer2, w3):
    return (np.dot((y_hat - y), w3.T) * derivative_sig(sig_layer2)).sum(axis=0)

def w1_deriv(y_hat, y, x, sig_layer1, w2, w3, sig_layer2):
    error = y, y_hat
    hidden3 = np.dot((y_hat - y), w3.T) * derivative_sig(sig_layer2)
    previous_deriv = np.dot(hidden3, w2.T) * derivative_sig(sig_layer1)
    return np.dot(x.T, previous_deriv)

def b1_deriv(y_hat, y, x, w2, sig_layer1, w3, sig_layer2):
    error = y - y_hat
    hidden3 = np.dot((y_hat - y), w3.T) * derivative_sig(sig_layer2)
    hidden2 = np.dot(hidden3, w2.T) * derivative_sig(sig_layer1)
    return (np.dot(hidden3.T, hidden2)).sum(axis=0)

def cost(y, y_hat):
    return -np.mean(y * np.log(y_hat))


lr = 0.00001
epochs = 100000
losses = []

for epoch in tqdm(range(epochs)):
    
    y_hat, sig_layer1, sig_layer2 = feed_forward(x_train, w1, b1, w2, b2, w3, b3)
    
    loss = cost(y2, y_hat)
    if epoch % 10000 == 0:
        print(loss)
    
    
    gw3 = w3_deriv(y_hat, y2, sig_layer2)
    gb3 = b3_deriv(y_hat, y2)
    
    gw2 = w2_deriv(y_hat, y2, sig_layer2, sig_layer1, w3)
    gb2 = b2_deriv(y_hat, y2, sig_layer2, w3)
    
    gw1 = w1_deriv(y_hat, y2, x_train, sig_layer1, w2, w3, sig_layer2)
    gb1 = b1_deriv(y_hat, y2, x_train, w2, sig_layer1, w3, sig_layer2)
    
    
    w3 = w3 - lr * gw3
    b3 = b3 - lr * gb3
    
    w2 = w2 - lr * gw2
    b2 = b2 - lr * gb2
    
    w1 = w1 - lr * gw1
    b1 = b1 - lr * gb1
    
    losses.append(loss)
    
    #evaluate the model

#train accuracy
train_correct = 0
concat = np.concatenate((y_hat, y2), 1)
predicted = np.argmax(y_hat, 1)
for idx, i in enumerate(predicted):
    if predicted[idx] == test_encoded_y[idx]:
        train_correct += 1

print(f'train_accuracy :{train_correct/len(test_encoded_y)}')

#test accuracy

y_test2 = np.zeros((61, 2))
for j in range(61):
    y_test2[j, y_test[j]] = 1

y_val, _, _ = feed_forward(x_test, w1, b1, w2, b2, w3, b3)

# since the data is not balanced the network is leaning into the survived status
# we need to adjust the threshhold to balance it!
threshold = 0.40
y_val_new = np.where(y_val[:, 1]>threshold, 1, 0)
concatenate_new_y_val = np.concatenate((y_val[:, 0].reshape(-1, 1), y_val_new.reshape(-1, 1)), 1)

test_correct = 0
test_pred = np.argmax(concatenate_new_y_val, 1)
for idx2, j in enumerate(test_pred):
    if test_pred[idx2] == y_test[idx2]:
        test_correct += 1
        
print(f'test accuracy :{test_correct/len(test_pred)}')

test_compare = np.concatenate((test_pred.reshape(-1, 1), y_test.reshape(-1, 1)), 1)

plt.plot(losses)


''' the test accuracy is 75% , the data is way small for such a deep network'''




















