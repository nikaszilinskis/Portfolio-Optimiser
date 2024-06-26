"""
Implementation of Modern Portfolio Theory (MPT) using Python.
"""

import yfinance as yf 
import pandas as pd 
from pandas_datareader import data as pdr
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


def getInformation(tickers, start_date, end_date):
    """
    Import required infromation from Yahoo Finance
    """
    stockInformation = yf.download(tickers, start=start_date, end=end_date)
    stockInformation = stockInformation["Adj Close"]

    total_days = len(stockInformation)
    midPoint = total_days//2

    trainStockInformation = stockInformation[:midPoint]
    testStockInformation = stockInformation[midPoint:]

    trainReturns = trainStockInformation.pct_change()
    trainMeanReturns = trainReturns.mean() 
    trainCovMatrix = trainReturns.cov()
    testReturns = testStockInformation.pct_change()
    testMeanReturns = testReturns.mean()
    testCovMatrix = testReturns.cov()

    return trainMeanReturns, trainCovMatrix, testMeanReturns, testCovMatrix

def portfolioMetrics(weights, meanReturns, covMatrix):
    """
    Portfolio performence metrics
    """
    returns = np.sum(meanReturns*weights)*252 #Annualised returns
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights)))*np.sqrt(252)
    return returns, std

def negativeSharpeRatio(weights, meanReturns, covMatrix, riskFreeRate = 0.042):
    """
    Function to calculate the negative Sharpe Ratio
    """
    pReturns, pStd = portfolioMetrics(weights, meanReturns, covMatrix)
    return -((pReturns-riskFreeRate)/pStd)

def minimizeNegativeSR(meanReturns, covMatrix, riskFreeRate = 0.042, constraintSet = (0,0.5)):
    """
    Function to  minimise the negative Sharpe Ratio
    """
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights)-1})
    bounds = tuple(constraintSet for _ in range(len(tickers)))
    result = minimize(negativeSharpeRatio, np.array(numAssets*[1./numAssets,]), args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolioStd(weights, meanReturns, covMatrix):
    """
    Function to calculate the portfolio standard deviation
    """
    return portfolioMetrics(weights, meanReturns, covMatrix)[1]

def portfolioRetrun(weights, meanReturns, covMatrix):
    return portfolioMetrics(weights, meanReturns, covMatrix)[0]

def minimizeVariance(meanReturns, covMatrix, constraintSet = (0,0.5)):
    """
    Function to minimise the portfolio variance
    """
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights)-1})
    bounds = tuple(constraintSet for _ in range(len(tickers)))
    result = minimize(portfolioStd, np.array(numAssets*[1./numAssets,]), args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def efficientFrontier(meanReturns, covMatrix, returnTarget, constraintset=(0,0.5)):
    """
    Optimise the portfolios minimum variance for a given return target
    """
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)

    constraints = ({'type': 'eq', 'fun': lambda weights: portfolioRetrun(weights, meanReturns, covMatrix) - returnTarget},
                   {'type': 'eq', 'fun': lambda weights: np.sum(weights)-1})
    bounds = tuple(constraintset for _ in range(numAssets))
    efficientOptimiser = minimize(portfolioStd, np.array(numAssets*[1./numAssets,]), args=args, method='SLSQP', bounds=bounds, constraints=constraints)

    return efficientOptimiser
    

def calculatedResults(trainMeanReturns, trainCovMatrix, testMeanReturns, testCovMatrix, riskFreerate=0.042, constraintset=(0,0.5)):
    """
    Read in mean returns, covariance matrix, risk free rate and constraint set and
    output the maximised Sharpe Ratio, minimised variance and optimal weights
    """
    maxSRresult = minimizeNegativeSR(trainMeanReturns, trainCovMatrix, riskFreerate, constraintset)
    trainMaxSRreturns, trainMaxSRstd = portfolioMetrics(maxSRresult['x'], trainMeanReturns, trainCovMatrix)
    testMaxSRreturns, testMaxSRstd = portfolioMetrics(maxSRresult['x'], testMeanReturns, testCovMatrix)

    maxSRAllocation = pd.DataFrame(maxSRresult['x'], index=trainMeanReturns.index, columns=['MaxSRAllocation (%)'])
    maxSRAllocation['MaxSRAllocation (%)'] = round(maxSRAllocation['MaxSRAllocation (%)']*100, 4)

    minVarResult = minimizeVariance(trainMeanReturns, trainCovMatrix, constraintset)
    trainMinVarReturns, trainMinVarstd = portfolioMetrics(minVarResult['x'], trainMeanReturns, trainCovMatrix)
    testMinVarReturns, testMinVarstd = portfolioMetrics(minVarResult['x'], testMeanReturns, testCovMatrix)
    
    
    minVarAllocation = pd.DataFrame(minVarResult['x'], index=trainMeanReturns.index, columns=['MinVarAllocation (%)'])
    minVarAllocation['MinVarAllocation (%)'] = round(minVarAllocation['MinVarAllocation (%)']*100, 4)

    trainEfficientFrontierResults = []
    trainWeightsForTargets = []  #To store weights for each target return
    targetReturns1 = np.linspace(trainMinVarReturns, trainMaxSRreturns, 50)
    for target in targetReturns1:
        result = efficientFrontier(trainMeanReturns, trainCovMatrix, target, constraintset)
        trainEfficientFrontierResults.append(result.fun)
        trainWeightsForTargets.append(result.x)
    
    testEfficientFrontierResults = []
    targetReturns2 = np.linspace(testMinVarReturns, testMaxSRreturns, 50)
    for i, target2 in enumerate(targetReturns2):
        if i < len(trainWeightsForTargets):  #Ensure there is a corresponding set of weights
            weights = trainWeightsForTargets[i]
            std = portfolioMetrics(weights, testMeanReturns, testCovMatrix)[1]
            testEfficientFrontierResults.append(std)
        else:
            print("No corresponding weights found for target", target2)

    trainEqualWeightsReturns, trainEqualWeightsstd = portfolioMetrics(np.array(len(trainMeanReturns)*[1./len(trainMeanReturns),]), trainMeanReturns, trainCovMatrix)
    trainEqualWeightsReturns, trainEqualWeightsstd = round(trainEqualWeightsReturns, 4)*100, round(trainEqualWeightsstd, 4)*100

    testEqualWeightsReturns, testEqualWeightsstd = portfolioMetrics(np.array(len(testMeanReturns)*[1./len(testMeanReturns),]), testMeanReturns, testCovMatrix)
    testEqualWeightsReturns, testEqualWeightsstd = round(testEqualWeightsReturns, 4)*100, round(testEqualWeightsstd, 4)*100

    trainMaxSRreturns, trainMaxSRstd = round(trainMaxSRreturns, 4)*100, round(trainMaxSRstd, 4)*100
    trainMinVarReturns, trainMinVarstd = round(trainMinVarReturns, 4)*100, round(trainMinVarstd, 4)*100

    testMaxSRreturns, testMaxSRstd = round(testMaxSRreturns, 4)*100, round(testMaxSRstd, 4)*100
    testMinVarReturns, testMinVarstd = round(testMinVarReturns, 4)*100, round(testMinVarstd, 4)*100
    return trainMaxSRreturns, trainMaxSRstd, testMaxSRreturns, testMaxSRstd,  trainMinVarReturns, trainMinVarstd, testMinVarReturns, testMinVarstd, maxSRAllocation,  minVarAllocation, targetReturns1, targetReturns2, trainEfficientFrontierResults, testEfficientFrontierResults, trainEqualWeightsReturns, trainEqualWeightsstd, testEqualWeightsReturns, testEqualWeightsstd

def printResults(trainMeanReturns, trainCovMatrix, testMeanReturns, testCovMatrix, riskFreerate=0.042, constraintset=(0,0.5)):
    """
    Print the results of the calculated results
    """
    trainMaxSRreturns, trainMaxSRstd, testMaxSRreturns, testMaxSRstd,  trainMinVarReturns, trainMinVarstd, testMinVarReturns, testMinVarstd, maxSRAllocation,  minVarAllocation, targetReturns1, targetReturns2, trainEfficientFrontierResults, testEfficientFrontierResults, trainEqualWeightsReturns, trainEqualWeightsstd, testEqualWeightsReturns, testEqualWeightsstd = calculatedResults(trainMeanReturns, trainCovMatrix, testMeanReturns, testCovMatrix)
    # Print results for Maximized Sharpe Ratio
    print("Maximised Sharpe Ratio Optimal Weights:")
    print(maxSRAllocation)
    print("\nTraining Data:")
    print("Maximised Sharpe Ratio Portfolio Returns: ", round(trainMaxSRreturns, 4), "%")
    print("Maximised Sharpe Ratio Portfolio Standard Deviation: ", round(trainMaxSRstd, 4), "%")
    print("\nTesting Data:")
    print("Maximised Sharpe Ratio Portfolio Returns: ", round(testMaxSRreturns, 4), "%")
    print("Maximised Sharpe Ratio Portfolio Standard Deviation: ", round(testMaxSRstd, 4), "%")

    # Print results for Minimized Variance
    print("\nMinimised Variance Optimal Weights:")
    print(minVarAllocation)
    print("\nTraining Data:")
    print("Minimised Variance Portfolio Returns: ", round(trainMinVarReturns, 4), "%")
    print("Minimised Variance Portfolio Standard Deviation: ", round(trainMinVarstd, 4), "%")
    print("\nTesting Data:")
    print("Minimised Variance Portfolio Returns: ", round(testMinVarReturns, 4), "%")
    print("Minimised Variance Portfolio Standard Deviation: ", round(testMinVarstd, 4), "%")

    # Print results for equal weights portfolio
    print("\nEqual Weights Portfolio:")
    print("\nTraining Data:")
    print("Equal Weights Portfolio Returns: ", trainEqualWeightsReturns, "%")
    print("Equal Weights Portfolio Standard Deviation: ", trainEqualWeightsstd, "%")
    print("\nTesting Data:")
    print("Equal Weights Portfolio Returns: ", testEqualWeightsReturns, "%")
    print("Equal Weights Portfolio Standard Deviation: ", testEqualWeightsstd, "%")


def plotEfficientFrontier(trainMeanReturns, trainCovMatrix, testMeanReturns, testCovMatrix, constraintset=(0,0.5), plotStatus=True):
    if plotStatus:
        # Extracting results from your calculatedResults function
        trainMaxSRreturns, trainMaxSRstd, testMaxSRreturns, testMaxSRstd,  trainMinVarReturns, trainMinVarstd, testMinVarReturns, testMinVarstd, maxSRAllocation,  minVarAllocation, targetReturns1, targetReturns2, trainEfficientFrontierResults, testEfficientFrontierResults, trainEqualWeightsReturns, trainEqualWeightsStd, testEqualWeightsReturns, testEqualWeightsStd = calculatedResults(trainMeanReturns, trainCovMatrix, testMeanReturns, testCovMatrix)

        # Combined Data Plot
        data = [
            go.Scatter(name="Max SR (Train)", mode='markers', x=[trainMaxSRstd], y=[trainMaxSRreturns], marker=dict(size=15, color='red')),
            go.Scatter(name="Min Vol (Train)", mode='markers', x=[trainMinVarstd], y=[trainMinVarReturns], marker=dict(size=15, color='blue')),
            go.Scatter(name="Frontier (Train)", mode='lines', x=[round(sd * 100, 4) for sd in trainEfficientFrontierResults], y=[round(tr * 100, 4) for tr in targetReturns1], line=dict(color='black', width=3)),
            go.Scatter(name="Max SR (Test)", mode='markers', x=[testMaxSRstd], y=[testMaxSRreturns], marker=dict(size=15, color='red', symbol='x')),
            go.Scatter(name="Min Vol (Test)", mode='markers', x=[testMinVarstd], y=[testMinVarReturns], marker=dict(size=15, color='blue', symbol='x')),
            go.Scatter(name="Frontier (Test)", mode='lines', x=[round(sd * 100, 4) for sd in testEfficientFrontierResults], y=[round(tr * 100, 4) for tr in targetReturns2], line=dict(color='grey', width=3, dash='dashdot')),
            go.Scatter(name="Equal Weight (Train)", mode='markers', x=[trainEqualWeightsStd], y=[trainEqualWeightsReturns], marker=dict(size=15, color='purple')),
            go.Scatter(name="Equal Weight (Test)", mode='markers', x=[testEqualWeightsStd], y=[testEqualWeightsReturns], marker=dict(size=15, color='purple', symbol='x'))
        ]

        layout = go.Layout(
            title="Efficient Frontier: Training vs Testing",
            xaxis=dict(
                title='Annualised Volatility (%)',
                showgrid=False,  # Turn off the grid lines
                showline=True,  # Show the axis line
                showticklabels=True,  # Ensure tick labels are shown
                linewidth=2,  # Width of the axis line for visibility
                linecolor='black'  # Color of the axis line
            ),
            yaxis=dict(
                title='Annualised Return (%)',
                showgrid=False,  # Turn off the grid lines
                showline=True,  # Show the axis line
                showticklabels=True,  # Ensure tick labels are shown
                linewidth=2,  # Width of the axis line for visibility
                linecolor='black'  # Color of the axis line
            ),
            showlegend=True,
            legend=dict(x=0.01, y=1, orientation="h"),
            plot_bgcolor='white',
            width=800,
            height=700
        )

        fig = go.Figure(data=data, layout=layout)
        fig.show()
         
        maxSRAllocation = maxSRAllocation.squeeze()  #Squeeze DataFrame to Series
        minVarAllocation = minVarAllocation.squeeze()  #Squeeze DataFrame to Series
        
        # Define a color for the bars
        bar_color = 'skyblue'

        # Visualization of Maximized Sharpe Ratio Portfolio Weights
        non_zero_maxSRAllocation = maxSRAllocation[maxSRAllocation.values > 0]
        plt.figure(figsize=(10, 6))
        bars1 = plt.bar(non_zero_maxSRAllocation.index, non_zero_maxSRAllocation.values, color=bar_color)
        plt.xlabel('Stocks')
        plt.ylabel('Allocation (%)')
        plt.title('Maximised Sharpe Ratio Portfolio Weights')
        plt.xticks(rotation=45)

        # Adding the percentage allocation within the bar chart
        for bar in bars1:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.1f}%', ha='center', va='bottom')

        plt.show()

        # Visualization of Minimized Variance Portfolio Weights
        non_zero_minVarAllocation = minVarAllocation[minVarAllocation.values > 0]
        plt.figure(figsize=(8, 6))
        bars2 = plt.bar(non_zero_minVarAllocation.index, non_zero_minVarAllocation.values, color=bar_color)
        plt.xlabel('Stocks')
        plt.ylabel('Allocation (%)')
        plt.title('Minimised Variance Portfolio Weights')
        plt.xticks(rotation=45)

        # Adding the percentage allocation within the bar chart
        for bar in bars2:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.1f}%', ha='center', va='bottom')

        plt.show()

def plotCovMatrix(covMatrix, plotStatus=True):
    """
    Function to plot the covariance matrix if True
    """
    if plotStatus:
        plt.figure(figsize=(10, 10))
        sns.heatmap(covMatrix, annot=True, fmt='.4f', cmap='coolwarm')
        plt.show()
    else:
        pass


tickers = ["LLOY.L","BARC.L","BT-A.L","EZJ.L","JD.L","SBRY.L","TSCO.L","SHEL.L","VOD.L","NWG.L"]

end_date = datetime.now()
start_date = end_date - timedelta(days=10*365)
print("Start Date:", start_date)
print("End Date:", end_date)

trainMeanReturns, trainCovMatrix, testMeanReturns, testCovMatrix = getInformation(tickers, start_date, end_date)
trainMeanReturnsResult = trainMeanReturns
trainCovMatrixResult = trainCovMatrix
print("Train Mean Returns:\n", round(trainMeanReturnsResult*252,5)*100)
print(round(trainCovMatrixResult*252,5))
plotCovMatrix(trainCovMatrixResult*252, plotStatus=True)#Annualised Covariance Matrix

printResults(trainMeanReturns, trainCovMatrix, testMeanReturns, testCovMatrix)
plotEfficientFrontier(trainMeanReturns, trainCovMatrix, testMeanReturns, testCovMatrix)