import csv
import matplotlib.pyplot as plt
import numpy as np

data_path = './data_theory/data_set.csv'
iteration = 101
initial_alpha = 0.0000001

def forward(w, x):
    # w -> (5, 1)
    # x -> (n, 5)
    # return <- (n, 1)
    return np.matmul(x, w)

def loss(pd_y, gt_y):
    # pd_y -> (n, 1)
    # gt_y -> (n, 1)
    # return <- (1, 1)
    diff = pd_y - gt_y
    return np.matmul(diff.T, diff) / diff.shape[0]

def backward(w, x, gt_y, alpha):
    gradient = np.matmul(x.T, forward(w, x) - gt_y)
    return w - alpha * gradient

dataset = []
with open(data_path, encoding='utf-8') as f:
    for data in csv.DictReader(f):
        new_data = []
        for v in data.values():
            new_data.append(float(v))
        dataset.append(new_data)

dataset = np.asarray(dataset)
dataset_x = np.ones(dataset.shape)
dataset_x[:, 1:5] = dataset[:, 0:4]
dataset_y = dataset[:, 4:5]

## 5-train
train_num = 5
train_set_x = dataset_x[:train_num, :]
train_set_y = dataset_y[:train_num, :]
test_set_x = dataset_x[train_num:, :]
test_set_y = dataset_y[train_num:, :]
w = np.zeros([5, 1])
alpha = initial_alpha / train_num

# record
weight = []
train_loss = []
test_loss = []

for i in range(iteration):
    w = backward(w, train_set_x, train_set_y, alpha)
    weight.append(w)
    train_loss.append(float(loss(forward(w, train_set_x), train_set_y)))
    test_loss.append(float(loss(forward(w, test_set_x), test_set_y)))

print(w)

## plot
plt.axhline(train_loss[-1], 0, 0.45, color='blue', linestyle='dashed')
plt.axhline(test_loss[37], 0, 0.36, color='red', linestyle='dashed')
plt.text(-16, train_loss[-1]-15, '{:.2f}'.format(train_loss[-1]))
plt.text(-16, test_loss[37]-15, '{:.2f}'.format(test_loss[37]))

plt.semilogy(range(iteration), train_loss, color='blue', label='train_loss')
plt.semilogy(range(iteration), test_loss, color='red', label='test_loss')
plt.legend()

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('q1.jpg')

## 200-train
train_num = 200
train_set_x = dataset_x[:train_num, :]
train_set_y = dataset_y[:train_num, :]
test_set_x = dataset_x[train_num:, :]
test_set_y = dataset_y[train_num:, :]
w = np.zeros([5, 1])
alpha = initial_alpha / train_num

# record
weight = []
train_loss = []
test_loss = []

for i in range(iteration):
    w = backward(w, train_set_x, train_set_y, alpha)
    weight.append(w)
    train_loss.append(float(loss(forward(w, train_set_x), train_set_y)))
    test_loss.append(float(loss(forward(w, test_set_x), test_set_y)))

print(w)

## plot
plt.cla()
plt.axhline(train_loss[-1], 0, 0.5, color='blue', linestyle='dashed')
plt.axhline(test_loss[-1], 0, 0.45, color='red', linestyle='dashed')
plt.text(-16, train_loss[-1]-15, '{:.2f}'.format(train_loss[-1]))
plt.text(-16, test_loss[-1]-15, '{:.2f}'.format(test_loss[-1]))

plt.semilogy(range(iteration), train_loss, color='blue', label='train_loss')
plt.semilogy(range(iteration), test_loss, color='red', label='test_loss')

plt.legend()

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('q2.jpg')