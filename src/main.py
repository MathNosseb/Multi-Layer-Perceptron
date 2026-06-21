import random
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_inv(y):
    if y <= 0 or y >= 1:
        raise ValueError("y must be in (0, 1)")
    return math.log(y / (1 - y))

train = [
    [0,0,0],
    [1,0,0],
    [1,1,1],
    [0,1,0]
]


w1 = random.random()
w2 = random.random()

b = random.random()
lr = 0.1

#entrainement
for i in range(100000):

    dataset = random.choice(train)

    x1 = dataset[0]
    x2 = dataset[1]

    #propagation avant
    x3 = sigmoid( x1 * w1 + x2 * w2 + b)

    #erreur
    error = dataset[2] - x3

    #correction erreur
    grad = error * x3 * (1 - x3)

    w1 += lr * grad * x1
    w2 += lr * grad * x2
    b += lr * grad

#prediction
for dataset in train:
    x1 = dataset[0]
    x2 = dataset[1]

    x3 = sigmoid( x1 * w1 + x2 * w2 + b)
    print(round(x3,1), dataset[2])