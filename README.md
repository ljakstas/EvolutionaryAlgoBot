# EvolutionaryAlgoBot
A trading strategy is a set of rules to follow for desicions regarding buying and selling in markets. Strategies can range from involving a couple of relatively simple indicators to a plethora of intricate mathamatics. Some strategies involve parameters whose values greatly impact trading decision in the market. This project uses evolutionary algorithms (a type of deep learning) to optimize a trading strategy's parameters to maximize returns.

#### *Packages and Frameworks*
Three main packages and frameworks were used: QuantConnect, LEAN Engine, and DEAP. 
1. Quantconnect is a browser/cloud-based, open-source, algorithmic trading platform where engineers can develop, execute, and backtest their strategies for a multitude of equities. QuantConnect provides their own trading API for market access and trading manipulation. 
2. LEAN Engine: QuantConnect’s CLI backtesting framework. This allowed us to script the backtests and incorporate the Genetic algorithm alongside it, which was done through DEAP. 
3. Distributed Evolutionary Algorithms in Python (DEAP) is an evolutionary computation framework that allows for accelerated testing/“prototyping” of ideas. I used genetic algorithms from this framework. With these packages I was able to successfully develop, backtest, and evolve our trading strategy.


### Background

##### **S&P 500 (SPY) Market Data History (1993-2022)**

<img src="/Images/SPY_plain.PNG" width=75% height=75%>

Throughout the market's history, we have encountured a number of crashes and recoveries

### Strategy

<img src="/Images/SPY_Dark.JPG" width=75% height=75%>

<img src="/Images/MOVING_AVG_EX.PNG" width=70% height=70%>

### Methodology

<img src="/Images/EvolutionCycle.PNG" width=75% height=75%>


### Results
<img src="/Images/FitnessGraph_Dark.PNG" width=55% height=55%>

Iteration | Best Fitness | Average Fitness
:---: | :---: | :---:
**0** | **74** | **28**
**1** | **73** | **42**
**2** | **78** | **49**
**3** | **79** | **61**
**4** | **86** | **66**
**5** | **86** | **69**
**6** | **87** | **75**
**7** | **88** | **84**
**8** | **89** | **82**
**9** | **89** | **83**



<img src="/Images/120-20-50_Dark.JPG" >
<img src="/Images/120-20-50_Dark_TESTED.jpg" >
<img src="/Images/120-20-50_Light.JPG" >

<img src="/Images/60-35-33_Dark.JPG" >
<img src="/Images/60-35-33_Dark_TESTED.jpg" >
<img src="/Images/60-35-33_Light.JPG" >

<img src="/Images/66-42-9_Dark.JPG" >
<img src="/Images/66-42-9_Dark_TESTED.jpg" >
<img src="/Images/66-42-9_Light.JPG" >

### Conclusion and Future Work

Genetic algorithms can indeed converge algorithmic trading parameters to optimize returns in the market (as seen through the training results). Fitness of the agents was equated to the percentage of net return. Over 10 generations fitness increased 200% (55 percentage points, see Table I.) However, due to using a single date range to train the model, the algorithm tended to overfit the data. After testing on a new date range, we find that the parameters were not optimized. To make the model more robust, training would need to consist of mutliple date ranges. To incorporate this into the model, we could add a new (quantitative) parameter for the market stage - accumulation(0), uptrend(1), distribution(2), and downtrend(3). After training, depending on the market phase, we could expect the model to produce more robust and effective parameters for testing on different date ranges.



<img src="/Images/Research Poster.png" width=55% height=55%>


