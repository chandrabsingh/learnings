import numpy as np

##########################################################################
# Modeling: what we want to capture

# points = [(np.array([2]),4), (np.array([4]),2)]
# d = 1

# Generate artificial data
true_w = np.array([1,2,3,4,5])
d = len(true_w)
points = []
iterationCount = 500
for i in range(iterationCount):
    x = np.random.randn(d)
    y = true_w.dot(x) + np.random.randn()
    #  print(x, y)
    points.append((x,y))


def F(w):
    return sum((w.dot(x) - y)**2 for x, y in points) / len(points)

def dF(w):
    return sum(2*(w.dot(x) - y)*x  for x, y in points) / len(points)

def sF(w, i):
    x,y = points[i]
    return (w.dot(x) - y)**2

def sdF(w, i):
    x,y = points[i]
    return 2*(w.dot(x) - y)*x 



##########################################################################
# Algorithms: how we compute it 

def gradientDescent(F, dF, d):
    # gradient descent
    w = np.zeros(d)
    eta = 0.01
    for t in range(iterationCount):
        value = F(w)
        gradient = dF(w)
        w = w - eta * gradient
        print('iteration {}: w = {}, F(w) = {}'.format(t, w, value))

def stochasticGradientDescent(sF, sdF, d, n):
    # stochastic gradient descent
    w = np.zeros(d)
    eta = 1
    numUpdates = 0
    for t in range(iterationCount):
        for i in range(n):
            value = sF(w, i)
            gradient = sdF(w, i)
            numUpdates += 1
            eta = 1 / numUpdates
            w = w - eta * gradient
        print('iteration {}: w = {}, F(w) = {}'.format(t, w, value))


##########################################################################
# gradientDescent(F, dF, d)
stochasticGradientDescent(sF, sdF, d, len(points))
