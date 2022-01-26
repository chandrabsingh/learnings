import pandas as pd
import numpy as np

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_colwidth', None)

##############################################
# Model

# points = [(np.array([2]),4), (np.array([4]),2)]
# d = 1

# Generate data
iterationCount = 2000
true_w = np.array([1,2,3,4,5,]) # Reverse-engineer to get to this vector
d = len(true_w)
dfColNames = [f"w{s+1}" for s in range(d)]
dfColNames.append('F(w)')
points = []
for i in range(iterationCount):
    x = np.random.randn(d)
    y = true_w.dot(x) + np.random.randn()
    points.append((x,y))


def F(w):
    return sum((w.dot(x) - y)**2 for x, y in points) / len(points)

def dF(w):
    return sum(2*(w.dot(x) - y) * x for x, y in points) / len(points)

##############################################
# Algorithm

def gradientDescent(F, dF, d):
    w = np.zeros(d)
    eta = 0.01

    lst = []
    for t in range(iterationCount):
        l1 = []
        value = F(w)
        gradient = dF(w)
        w = w - eta * gradient
        l1.extend(w)
        l1.append(value)
        lst.append(l1)
    df = pd.DataFrame(lst, columns = dfColNames)
    df['Iteration'] = df.index
    return df

result = gradientDescent(F, dF, d)

# print(result)

