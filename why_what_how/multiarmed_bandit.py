import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class BanditClass:
    def __init__(self, epsilon, leverCount, probs):
        self.epsilon = epsilon
        self.leverCount = leverCount
        self.probs = probs
        self.redefineSettings()

    def calcAction(self):
        # Select action/lever as per explore-exploit probability
        if np.random.rand() < self.epsilon: # explore case
            return np.random.choice(self.leverCount)
        else: # exploit case
            return np.argmax(self.valueEstimateQ) # random.choice(self.leverCount)

    def calcReward(self, actionId):
        val = np.random.randn() + self.valueTrueQStar[actionId]
        # return val
        # return 1 if (np.random.random() < self.probs[actionId]) else 0
        return val if (np.random.random() < self.probs[actionId]) else 0

    def calcQEstimate(self, actionId, reward):
        self.actionCount[actionId] += 1
        self.valueEstimateQ[actionId] += (1/self.actionCount[actionId]) * (reward - self.valueEstimateQ[actionId])

    def redefineSettings(self):
        # Define individual lever probability
        self.valueTrueQStar = np.random.randn(self.leverCount) # Reset the valueTrueQStar before each incremental step
        self.valueEstimateQ = np.zeros(self.leverCount, dtype=float)
        self.actionCount = np.zeros_like(self.valueEstimateQ, dtype=int)

# def main():
epsilons = [0.1] # Define list of epsilons=exploration
leverCount = 10 # Define number of levers
runs = 2000
steps = 1000
probs = [0.10, 0.50, 0.60, 0.80, 0.10,
         0.25, 0.60, 0.45, 0.75, 0.65]
# Define list of arm probabilities

rewards = np.zeros((len(epsilons), runs, steps))
actions = np.zeros((len(epsilons), runs, steps))

for e, epsilon in enumerate(epsilons): # loop over all the bandits epsilons
    bandit = BanditClass(epsilon, leverCount, probs)
    for run in tqdm(range(runs)): # loop over all the runs
        bandit.redefineSettings()
        for step in range(steps): # loop over all the steps
            actionId = bandit.calcAction()
            reward = bandit.calcReward(actionId)
            bandit.calcQEstimate(actionId, reward)
            actions[e, run, step] = actionId
            rewards[e, run, step] = reward
        # print('Test')
avgActions, avgRewards = actions.mean(axis=1), rewards.mean(axis=1)

plt.subplot(2, 1, 1)
for eps, rewardsY in zip(epsilons, avgRewards):
    plt.plot(rewardsY, label=r'$\epsilon$ = {}'.format(eps), lw=1)
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.legend()

plt.subplot(2, 1, 2)
for eps, actionsX in zip(epsilons, avgActions):
    plt.plot(actionsX, label=r'$\epsilon$ = {}'.format(eps), lw=1)
plt.xlabel('Steps')
plt.ylabel('% Average action')
plt.legend()

plt.show()
