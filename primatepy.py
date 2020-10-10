## Null model:

# Description
'''
Agents are male or female; they mate at random; the model is not spatiotemporally explicit
They do not produce offspring, and do not have any characteristics that make them more or less likely to mate
Female agents do not have 'fertility' or anything that would signal fertility
'''

# Pseudo-code

'''
Create numberAgentsagents; randomly assign male or female sex (alternatively, assign m agents male sex and numberAgents- m agents female sex)
Each tick:
	One sex is selected randomly;
	Agents of that sex are sampled some random # of times (w/o replacement; the # of agents selected is drawn from uniform distribution)
	Each agent drawn in the sample mates with a randomly selected (without replacement) member of the opposite sex
'''

# Python code

import pandas as pd
from pandas.core.common import flatten
import numpy as np
#import plotly
import plotly.express as px
#%matplotlib inline
import matplotlib.pyplot as pl

def experiment():

    ticks = int(input("For how long should each model run (number of ticks)?"))
    iterations = int(input("How many iterations of each model run?"))
    includeNumberAgents = input("Is 'number of agents' a variable of interest (y/n)?")
    includeSexRatio = input("Is 'sex ratio' a variable of interest (y/n)?")
    includeIBI = input("Is 'interbirth interval' a variable of interest (y/n)?")
    includeMatingRegime = input("Is 'mating regime' a variable of interest (y/n)?")
    includeMaleReproductiveSkew = input("Is 'male reproductive skew' a variable of interest (y/n)?")
    includeFemaleReproductiveSkew = input("Is 'female reproductive skew' a variable of interest (y/n)?")
    
    ticks = np.arange(0,ticks)
    iterations = np.arange(0,iterations)
    includeNumberAgents = "y" if includeNumberAgents == "y" else "n"
    includeSexRatio = "y" if includeSexRatio == "y" else "n"
    includeIBI = "y" if includeIBI == "y" else "n"
    includeMatingRegime = "y" if includeMatingRegime == "y" else "n"
    includeMaleReproductiveSkew = "y" if includeMaleReproductiveSkew == "y" else "n"
    includeFemaleReproductiveSkew = "y" if includeFemaleReproductiveSkew == "y" else "n"
    
    if includeNumberAgents == "y":
        numberAgentsLow = int(input("Minimum number of agents:"))
        numberAgentsHigh = int(input("Maximum number of agents:"))
        numberAgentsInc = int(input("Increment to increase number of agents per run:"))
        numberAgentsRange = np.arange(numberAgentsLow, numberAgentsHigh + numberAgentsInc, numberAgentsInc)
    else:
        numberAgentsRange = [int(input('Number of agents:'))]
        
    if includeSexRatio == "y":
        sexRatioLow = float(input("Minimum ratio of males to females:"))
        sexRatioIncNum = round(float(input("Number of sex ratio increments:")) / 2)
        sexRatioInc = (1 - sexRatioLow) / (sexRatioIncNum + 1)
        sexRatioRange = np.arange(sexRatioLow, (1 + sexRatioInc), sexRatioInc)
        for i in sexRatioRange[0:-1]:
            sexRatioRange = np.append(sexRatioRange, 1 / i)
    else:
        sexRatioRange = [float(input('Ratio of males to females:'))]
        
    if includeIBI == "y":
        ibiLow = int(input("Minimum interbirth interval:"))
        ibiHigh = int(input("Maximum interbirth interval:"))
        ibiInc = int(input("Increment to increase interbirth interval:"))
        ibiRange = np.arange(ibiLow, ibiHigh + ibiInc, ibiInc)
    else:
        ibiRange = [int(input('Interbirth interval (in ticks):'))]
        
    if includeMatingRegime == "y":
        matingRegime1 = input("Include mating regime 1 (One mating pair per tick; y/n)?")
        matingRegime2 = input("Include mating regime 2 (Random number of mating pairs per tick; y/n)?")
        matingRegime3 = input("Include mating regime 3 (Maximum number of mating pairs per tick; y/n)?")
        
        matingRegime1 = 1 if matingRegime1 == "y" else "n"
        matingRegime2 = 2 if matingRegime2 == "y" else "n"
        matingRegime3 = 3 if matingRegime3 == "y" else "n"
        
        matingRegimeRange = [matingRegime1, matingRegime2, matingRegime3]
        for n in matingRegimeRange:
            if (n == "n"):
                matingRegimeRange.remove(n)
        
    else:
        matingRegimeRange = [int(input('Mating type (1 = One pair per run; 2 = Random number of pairs per run; 3 = Max pairs per run):'))]
    
    if includeMaleReproductiveSkew == "y":
        maleReproductiveSkewLow = int(input("Minimum male reproductive skew (0-100):"))
        maleReproductiveSkewHigh = int(input("Maximum male reproductive skew (0-100):"))
        maleReproductiveSkewIncNum = int(input("Number of male reproductive skew increments:"))
        maleReproductiveSkewInc = (maleReproductiveSkewHigh - maleReproductiveSkewLow ) / maleReproductiveSkewIncNum
        maleReproductiveSkewRange = np.arange(maleReproductiveSkewLow, maleReproductiveSkewHigh +
                                              maleReproductiveSkewInc, maleReproductiveSkewInc)
    else:
        maleReproductiveSkewRange = [int(input('Male reproductive skew (1-100):'))]
            
    if includeFemaleReproductiveSkew == "y":
        femaleReproductiveSkewLow = int(input("Minimum female reproductive skew (1-100):"))
        femaleReproductiveSkewHigh = int(input("Maximum female reproductive skew (1-100):"))
        femaleReproductiveSkewIncNum = int(input("Number of female reproductive skew increments:"))
        femaleReproductiveSkewInc = (femaleReproductiveSkewHigh - femaleReproductiveSkewLow) / femaleReproductiveSkewIncNum
        femaleReproductiveSkewRange = np.arange(femaleReproductiveSkewLow, femaleReproductiveSkewHigh +
                                                femaleReproductiveSkewInc, femaleReproductiveSkewInc)
    else:
        femaleReproductiveSkewRange = [int(input('Female reproductive skew (1-100):'))]
    
        
    experimentDF = pd.DataFrame(columns=['Iteration', 'numberAgents', 'sexRatio', 'interbirthInterval', 'matingRegime',
                                         'maleReproductiveSkew', 'femaleReproductiveSkew', 'sdMaleRS', 'sdFemaleRS'])
    experimentDF = setupExperiment(ticks, iterations, numberAgentsRange, sexRatioRange, ibiRange, matingRegimeRange,
                                   maleReproductiveSkewRange, femaleReproductiveSkewRange, experimentDF)
    return(experimentDF)
    


def setupExperiment(ticks, iterations, numberAgentsRange, sexRatioRange, ibiRange, matingRegimeRange,
                    maleReproductiveSkewRange, femaleReproductiveSkewRange, experimentDF):
    
    for num in numberAgentsRange:
        for ratio in sexRatioRange:
            for ibi in ibiRange:
                for mating in matingRegimeRange:
                    for mSkew in maleReproductiveSkewRange:
                        for fSkew in femaleReproductiveSkewRange:
                                experimentDF = goExperiment(ticks = ticks, iterations = iterations, numberAgents = num,
                                                            matingRegime = mating, IBI = ibi, ratio = ratio,
                                                            mSkew = mSkew, fSkew = fSkew, experimentDF = experimentDF)
    return(experimentDF)                
    

def goExperiment(ticks, iterations, numberAgents, matingRegime, IBI, ratio, mSkew, fSkew, experimentDF):
    
    for iteration in iterations:
        m = int(round((numberAgents * ratio / (1 + ratio))))
        f = numberAgents - m
        RSList = setupSkew(m, f, mSkew, fSkew)
        agentsDF = pd.DataFrame({"agentID":np.arange(1, numberAgents + 1), "agentSex":["m"] * m + ["f"] * f,
                                 "numberMates":[0] * numberAgents, "IBI": [0] * numberAgents, "RS": RSList})

        if matingRegime == 1:
            for tick in ticks:
                 agentsDF.IBI = [i - 1 for i in agentsDF.IBI]
                 if len(agentsDF.loc[(agentsDF['agentSex'] == 'f') & (agentsDF['IBI'] <= 0)]) > 0:
                     mates = list(flatten([np.random.choice(agentsDF['agentID'][agentsDF['agentSex'] == 'm'], 1),
                                           np.random.choice(agentsDF.loc[(agentsDF['agentSex'] == 'f') & (agentsDF['IBI'] <= 0)]['agentID'], 1)]))
    
                     for mate in mates:
                         agentsDF.loc[agentsDF.agentID == mate, 'numberMates'] += 1
                         agentsDF.loc[(agentsDF.agentID == mate) & (agentsDF.agentSex == "f"), 'IBI'] = IBI
    
    
        if matingRegime == 2:
            for tick in ticks:
                agentsDF.IBI = [i - 1 for i in agentsDF.IBI]
                limitingSex = min([len(agentsDF.loc[(agentsDF['agentSex'] == 'f' & agentsDF['RS'] != 0) & (agentsDF['IBI'] <= 0)]), sum(agentsDF['agentSex'] == 'm' & agentsDF['RS'] != 0)])
                numberMates = np.random.randint(0, limitingSex + 1)
                mates = list(flatten([np.random.choice(agentsDF['agentID'][agentsDF['agentSex'] == 'm'], numberMates, False, p = agentsDF.RS[agentsDF.agentSex == "m"]),
                                      np.random.choice(agentsDF.loc[(agentsDF['agentSex'] == 'f') & (agentsDF['IBI'] <= 0)]['agentID'], numberMates, False, p = agentsDF.RS[agentsDF.agentSex == "f"])]))
    
                for mate in mates:
                    agentsDF.loc[agentsDF.agentID == mate, 'numberMates'] += 1
                    agentsDF.loc[(agentsDF.agentID == mate) & (agentsDF.agentSex == "f"), 'IBI'] = IBI
        
    
        if matingRegime == 3:
            for tick in ticks:
                agentsDF.IBI = [i - 1 for i in agentsDF.IBI]
                limitingSex = min([len(agentsDF.loc[(agentsDF['agentSex'] == 'f' & agentsDF['RS'] != 0) & (agentsDF['IBI'] <= 0)]), sum(agentsDF['agentSex'] == 'm' & agentsDF['RS'] != 0)])
                mates = list(flatten([np.random.choice(agentsDF['agentID'][agentsDF['agentSex'] == 'm'], limitingSex, False, p = agentsDF.RS[agentsDF.agentSex == "m"]),
                                      np.random.choice(agentsDF.loc[(agentsDF['agentSex'] == 'f') & (agentsDF['IBI'] <= 0)]['agentID'], limitingSex, False, p = agentsDF.RS[agentsDF.agentSex == "f"])]))
    
                for mate in mates:
                    agentsDF.loc[agentsDF.agentID == mate, 'numberMates'] += 1
                    agentsDF.loc[(agentsDF.agentID == mate) & (agentsDF.agentSex == "f"), 'IBI'] = IBI
                    
        maleSD = np.std(agentsDF.numberMates[agentsDF.agentSex == 'm'])
        femaleSD = np.std(agentsDF.numberMates[agentsDF.agentSex == 'f'])
        experimentDF = experimentDF.append({'Iteration':iteration, 'numberAgents':numberAgents, 'sexRatio':ratio, 'interbirthInterval':IBI,
                                            'matingRegime':matingRegime, 'maleReproductiveSkew':mSkew, 'femaleReproductiveSkew':fSkew,
                                            'sdMaleRS':maleSD, 'sdFemaleRS':femaleSD}, ignore_index=True)

    return(experimentDF)                   


def setupSkew(m, f, mSkew, fSkew): # the measure of skew utilized here is Morisita's Iδ (see Tsuji et al. 2001 Am Nat)
    mSkew = ( ( ( mSkew * 0.9 ) + 10 ) / 100 ) * m # users input numbers between 1 and 100 for skew; this is only to make input intuitive
    fSkew = ( ( ( fSkew * 0.9 ) + 10 ) / 100 ) * f # Morisita's Iδ ranges from 1 to the # of individuals in the sex in question, a conversion carried out here
    mRSList  = [1 / m] * m # a list of equal probabilities of reproducing is initiated to start; this corresponds to a Morisita's Iδ of 1
    fRSList  = [1 / f] * f # the equation to find Morisita's Iδ is: Iδ = n * (∑πi^2)
                             # n is the number of individuals; πi is the probability of a mating being attribtued to the ith individual (RS)
    solveForMaleSkew = mSkew / m # since we know Iδ and n, we can divide both sides by n, leaving us with Iδ / n = (∑πi^2)
    solveForFemaleSkew = fSkew / f # the above has an infinite number of solutions, corresponding to various distributions of RS (πi)
                             # the following procedure raises and lowers entries in the RS lists until they correspond with the desired Iδ
                             # the same procedure will produce different RS distributions each time it is run even if Iδ remains the same
    solveForMaleSkewVar = 1 / m # Iδ starts at 1, so each RS starts at 1/N
    rsInc = solveForMaleSkew / 2 # the increment by which RS will be raised among some individuals and lowered among others
    while abs(solveForMaleSkew - solveForMaleSkewVar) > solveForMaleSkew / 20:
        if solveForMaleSkew > solveForMaleSkewVar:
            increaser = increaser =list(np.random.multinomial(1, mRSList)).index(1)
            increaseBy = np.random.uniform(0, rsInc)
            increaseBy = 1 - mRSList[increaser] if mRSList[increaser] + increaseBy > 1 else increaseBy
            mRSList[increaser] += increaseBy
            decreaseRemaining = increaseBy
            while decreaseRemaining > 0:
                decreaser = int(np.random.choice(np.arange(0,m), 1))
                decreaser = int(np.random.choice(np.arange(0,m), 1)) if decreaser == increaser else decreaser
                decreaseBy = decreaseRemaining
                decreaseBy = mRSList[decreaser] if decreaseBy > mRSList[decreaser] else decreaseBy
                mRSList[decreaser] -= decreaseBy
                decreaseRemaining -= decreaseBy
            #mRSList = [round(x,3) for x in mRSList]
            solveForMaleSkewVar = sum([x ** 2 for x  in mRSList])
            if solveForMaleSkew < solveForMaleSkewVar:
                rsInc /= 2
        if solveForMaleSkewVar > solveForMaleSkew:
            decreaser =list(np.random.multinomial(1, mRSList)).index(1)
            decreaseBy = np.random.uniform(0, rsInc)
            decreaseBy = mRSList[decreaser] if mRSList[decreaser] + decreaseBy > 1 else decreaseBy
            mRSList[decreaser] -= decreaseBy
            increaseRemaining = decreaseBy
            while increaseRemaining > 0:
                increaser = int(np.random.choice(np.arange(0,m), 1))
                increaser = int(np.random.choice(np.arange(0,m), 1)) if increaser == decreaser else increaser
                increaseBy = increaseRemaining
                increaseBy = 1 - mRSList[increaser] if mRSList[increaser] + increaseBy > 1 else increaseBy
                mRSList[increaser] += increaseBy
                increaseRemaining -= increaseBy
            #mRSList = [round(x,3) for x in mRSList]
            solveForMaleSkewVar = sum([x ** 2 for x  in mRSList])
            if solveForMaleSkew < solveForMaleSkewVar:
                rsInc /= 2
                
    solveForFemaleSkewVar = 1 / m # Iδ starts at 1, so each RS starts at 1/N
    rsInc = solveForFemaleSkew / 2 # the increment by which RS will be raised among some individuals and lowered among others
    while abs(solveForFemaleSkew - solveForFemaleSkewVar) > solveForFemaleSkew / 20:
        if solveForFemaleSkew > solveForFemaleSkewVar:
            increaser = increaser =list(np.random.multinomial(1, fRSList)).index(1)
            increaseBy = np.random.uniform(0, rsInc)
            increaseBy = 1 - fRSList[increaser] if fRSList[increaser] + increaseBy > 1 else increaseBy
            fRSList[increaser] += increaseBy
            decreaseRemaining = increaseBy
            while decreaseRemaining > 0:
                decreaser = int(np.random.choice(np.arange(0,f), 1))
                decreaser = int(np.random.choice(np.arange(0,f), 1)) if decreaser == increaser else decreaser
                decreaseBy = decreaseRemaining
                decreaseBy = fRSList[decreaser] if decreaseBy > fRSList[decreaser] else decreaseBy
                fRSList[decreaser] -= decreaseBy
                decreaseRemaining -= decreaseBy
            #fRSList = [round(x,3) for x in fRSList]
            solveForFemaleSkewVar = sum([x ** 2 for x  in fRSList])
            if solveForFemaleSkew < solveForFemaleSkewVar:
                rsInc /= 2
        if solveForFemaleSkewVar > solveForFemaleSkew:
            decreaser =list(np.random.multinomial(1, fRSList)).index(1)
            decreaseBy = np.random.uniform(0, rsInc)
            decreaseBy = fRSList[decreaser] if fRSList[decreaser] + decreaseBy > 1 else decreaseBy
            fRSList[decreaser] -= decreaseBy
            increaseRemaining = decreaseBy
            while increaseRemaining > 0:
                increaser = int(np.random.choice(np.arange(0,f), 1))
                increaser = int(np.random.choice(np.arange(0,f), 1)) if increaser == decreaser else increaser
                increaseBy = increaseRemaining
                increaseBy = 1 - fRSList[increaser] if fRSList[increaser] + increaseBy > 1 else increaseBy
                fRSList[increaser] += increaseBy
                increaseRemaining -= increaseBy
            #fRSList = [round(x,3) for x in fRSList]
            solveForFemaleSkewVar = sum([x ** 2 for x  in fRSList])
            if solveForFemaleSkew < solveForFemaleSkewVar:
                rsInc /= 2
    
    return(mRSList + fRSList)                
                
experimentDF = experiment()    
experimentDF.to_csv("experiment.csv",index = False)

fig = px.scatter(
    experimentDF, x='sexRatio', y='sdMaleRS', opacity=0.65,
    trendline='ols', trendline_color_override='darkblue'
)

fig.write_html("experimentPlot.html")

#fig.show()

#print(agentsDF)

#print("Variance of male RS: " + str(round(np.std(agentsDF.numberMates[agentsDF.agentSex == 'm']),3)))
#print("Variance of female RS: " + str(round(np.std(agentsDF.numberMates[agentsDF.agentSex == 'f']),3)))

       
