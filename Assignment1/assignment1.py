#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model
from matplotlib import pyplot as plt


# In[2]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[3]:


def readJSON(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        d = eval(l)
        u = d['userID']
        g = d['gameID']
        yield u,g,d


# In[4]:


allHours = []
for l in readJSON("train.json.gz"):
    allHours.append(l)


# In[5]:


hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]


# In[6]:


allHours[0]


# In[7]:


##################################################
# Play prediction                                #
##################################################


# In[8]:


usersPerGame = defaultdict(set) # Maps an item to the users who rated it
gamesPerUser = defaultdict(set) # Maps a user to the items that they rated
hoursPerUser = defaultdict(set)
hoursPerGame = defaultdict(set)

for user, game, data in hoursTrain:
    usersPerGame[game].add(user)
    gamesPerUser[user].add(game)
    hoursPerUser[user].add(data['hours'])
    hoursPerGame[game].add(data['hours'])


# In[9]:


newValidSet = []
allGames = set(value[1] for value in allHours)
for user, game, data in hoursValid:
    # list of games that has been played
    playedGames = set(value[1] for value in allHours if value[0] == user)
    unplayedGames = list(set(allGames) - set(playedGames))
    
    if unplayedGames:
        randomGame = random.choice(unplayedGames)
    else:
        randomGame = None
    newValidSet.append((user, randomGame, 0))
    if data['hours'] > 0:
        newValidSet.append((user, game, 1))
random.seed(42)
random.shuffle(newValidSet)


# In[10]:


newValidSet[:10]


# In[11]:


# baseline-model from baseline.py
gameCount = defaultdict(int)
totalPlayed = 0

for user,game,_ in newValidSet:
    gameCount[game] += 1
    totalPlayed += 1

mostPopular = [(gameCount[x], x) for x in gameCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count += ic
    return1.add(i)
    if count > totalPlayed/2: break


# In[12]:


# binary list according to the new validation set
actualCorrect = []
for _, _, binary in newValidSet:
    if binary == 1:
        actualCorrect.append(1)
    else:
        actualCorrect.append(0)
# list accordingly but of the baseline model
baselinePredictions = []
for _, game, _ in newValidSet:
    if game in return1: # top ranked games
        baselinePredictions.append(1)
    else:
        baselinePredictions.append(0)


# In[13]:


def calculateAccuracy(actualLabels, predictLabels):
    correctPredictions = sum(1 for actual, predict in zip(actualLabels, predictLabels) if actual == predict)
    total = len(actualLabels)
    accuracy = correctPredictions / total
    return accuracy


# In[14]:


calculateAccuracy(actualCorrect, baselinePredictions)


# In[15]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[16]:


def maxSimilarity(user, game):
    similarities = []
    userHasPlayed = gamesPerUser[user]
    for gprime in userHasPlayed:
        if game == gprime: continue
        sim = Jaccard(usersPerGame[game], usersPerGame[gprime])
        similarities.append(sim)
    if not similarities:
        return 0
    else:
        maxSim = max(similarities)
    return maxSim


# In[19]:


maxSimDict = {} 
for u, g, _ in hoursValid:
    maxSim = maxSimilarity(u, g)
    maxSimDict[(u, g)] = maxSim


# In[17]:


userGames = {}
for user, game in gamesPerUser.items():
    allGames = set()
    for g in game:
        allGames = allGames.union(usersPerGame[g])
    userGames[user] = allGames


# In[21]:


### THESE STEPS ARE FOR FINDING THE BEST THRESHOLD
# start = 0.01
# end = 0.02
# step = 0.0001
# thresholds = [i / 10000.0 for i in range(int(start * 10000), int((end + step) * 10000), int(step * 10000))]
# thresholds = [i / 10000.0 for i in range(1, 120001)]

threshold = 0.0113
predictions = []
correctPredictions = 0

for user, game, data in newValidSet:
    if game in usersPerGame:
        similarity = Jaccard(usersPerGame[game], userGames[user])
    else:
        similarity = 0
        
    if similarity > threshold:
        if data == 1:
            correctPredictions += 1
    else:
        if data == 0:
            correctPredictions += 1

accuracy = correctPredictions / len(newValidSet)
accuracy


# In[125]:


# thresholdUse = 0.015625


# In[ ]:





# In[24]:


predictions = open("predictions_Played.csv", 'w')
for l in open("pairs_Played.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u, g = l.strip().split(',')
    # Logics are here
    if u in userGames and g in usersPerGame:
        similarity = Jaccard(usersPerGame[g], userGames[u])
    else:
        similarity = 0
    
    if similarity > threshold:
        pred = 1
    else:
        pred = 0
    predictions.write(u + ',' + g + ',' + str(pred) + '\n')

predictions.close()


# In[40]:


##################################################
# Hours played prediction                        #
##################################################


# In[25]:


trainHours = [r[2]['hours_transformed'] for r in hoursTrain]
globalAverage = sum(trainHours) * 1.0 / len(trainHours)


# In[26]:


hoursTrain[0]


# In[27]:


def MSE(predictions, labels):
    differences = [(x-y)**2 for x,y in zip(predictions,labels)]
    return sum(differences) / len(differences)


# In[28]:


betaU = {}
betaI = {}
for u, g, d in hoursTrain:
    betaU[u] = 0.0
    betaI[g] = 0.0


# In[29]:


alpha = globalAverage # Could initialize anywhere, this is a guess


# In[30]:


userItemHour = {(user, game): data['hours_transformed'] for user, game, data in hoursTrain}


# In[46]:


def iterate(lamb):
    alpha = globalAverage
    for u, i, _ in hoursTrain:
        # Get hours_transfromed of (u, i)
        R_ui = userItemHour.get((u, i))
        # Calculate alpha
        alpha += R_ui - (betaU[u] + betaI[i])
    alpha = alpha / len(hoursTrain)
        
    # for each game that user u has played, compute betaU of u
    for user, _, _ in hoursTrain:
        I_u = len(gamesPerUser[user])
        temBetaU = 0.0
        for game in gamesPerUser[user]:
            R_ui = userItemHour.get((user, game))
            temBetaU += R_ui - (alpha + betaI[game])
        betaU[user] = temBetaU / (lamb + I_u)
        
    # for each user that has played game i, compute betaI of i
    for _, game, _ in hoursTrain:
        U_i = len(usersPerGame[game])
        temBetaI = 0.0
        for user in usersPerGame[game]:
            R_ui = userItemHour.get((user, game))
            temBetaI += R_ui - (alpha + betaU[user])
        betaI[game] = temBetaI / (lamb + U_i)
        
    return alpha, betaU, betaI


# In[47]:


alpha, betaU, betaI = iterate(2)


# In[48]:


predictions = [alpha + betaU[u] + betaI[g] for u, g, _ in hoursValid]
labels = [r[2]['hours_transformed'] for r in hoursValid]


# In[49]:


validMSE = MSE(predictions, labels)
validMSE


# In[50]:


betaUs = [(betaU[u], u) for u in betaU]
betaIs = [(betaI[i], i) for i in betaI]
betaUs.sort()
betaIs.sort()

print("Maximum betaU = " + str(betaUs[-1][1]) + ' (' + str(betaUs[-1][0]) + ')')
print("Maximum betaI = " + str(betaIs[-1][1]) + ' (' + str(betaIs[-1][0]) + ')')
print("Minimum betaU = " + str(betaUs[0][1]) + ' (' + str(betaUs[0][0]) + ')')
print("Minimum betaI = " + str(betaIs[0][1]) + ' (' + str(betaIs[0][0]) + ')')


# In[51]:


[betaUs[-1][0], betaUs[0][0], betaIs[-1][0], betaIs[0][0]]


# In[53]:


predictions = open("predictions_Hours.csv", 'w')
for l in open("pairs_Hours.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    # Logic...
    bu = 0
    bi = 0
    if u in betaU:
        bu = betaU[u]
    if g in betaI:
        bi = betaI[g]
    
    _ = predictions.write(u + ',' + g + ',' + str(alpha + bu + bi) + '\n')

predictions.close()

