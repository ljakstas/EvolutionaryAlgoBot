#Driver program
import subprocess
import datetime
import random
import time
import backtrader as bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#from alpha_vantage.timeseries import TimeSeries
from deap import base
from deap import creator
from deap import tools
from tqdm import trange
from pathlib import Path
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def changeParams(file, slow, fast, rsi):
    dir = file + "\\" +  'config.json'
    with open(dir) as f:
        data = json.load(f)
    #for i in data['parameters']:
    #    print(i[1])
    data['parameters']["movingAvgSlow"] = str(slow)
    data['parameters']["movingAvgFast"] = str(fast)
    data['parameters']["rSIselling"] = str(rsi)
    #change params here
    print(data['parameters']["movingAvgSlow"])
    print(data['parameters']["movingAvgFast"])
    print(data['parameters']["rSIselling"])
    
    with open(dir, 'w') as x:
        json.dump(data, x)

    f.close()


#Strategy func
STRATEGY_PARAMS = dict(fast_period=15, slow_period=50, rsi_sell=40)
def RunBackTest(file,params):

    changeParams(file, params["fast_period"], params["slow_period"], params["rsi_sell"])
    command = "lean backtest \"testp\""
    subprocess.run(command, shell =True)
    a = subprocess.run(["py", "Scrape.py"], capture_output=True, text=True).stdout.strip("\n")
    
    print(float(a))
    return float(a) #net gain percent


#######RunBackTest('filePath', STRATEGY_PARAMS)

def csvAppend(csvFile, params, fit):
    values = []
    values.append(params["fast_period"])
    values.append(params["slow_period"])
    values.append(params["rsi_sell"])
    values.append(fit)
    arr = []
    arr.append(values)

    df =pd.DataFrame(arr)
    df.to_csv(csvFile, mode='a', index=False, header=False)

# fix the seed so that we will get the same results
# feel free to change it or comment out the line
random.seed(1)

# GA parameters
PARAM_NAMES = ["fast_period", "slow_period", "rsi_sell"]
NGEN = 10 #was 20
NPOP = 100
CXPB = 0.5
MUTPB = 0.3

def evaluate(individual, plot=False, log=False):

    # convert list of parameter values into dictionary of kwargs
    strategy_params = {k: v for k, v in zip(PARAM_NAMES, individual)}

    # fast moving average by definition cannot be slower than the slow one
    if strategy_params["fast_period"] >= strategy_params["slow_period"]:
        return [-np.inf]

    #set params here
    changeParams("testp", strategy_params["fast_period"], strategy_params["slow_period"], strategy_params["rsi_sell"])

    # by setting stdstats to False, backtrader will not store the changes in
    # statistics like number of trades, buys & sells, etc.
    #cerebro = bt.Cerebro(stdstats=False)
    #cerebro.adddata(data)

    # Remember to set it high enough or the strategy may not
    # be able to trade because of short of cash
    #initial_capital = 10_000.0
    #cerebro.broker.setcash(initial_capital)

    # Pass in the genes of the individual as kwargs
    #cerebro.addstrategy(CrossoverStrategy, **strategy_params)

    # This is needed for calculating our fitness score
    #cerebro.addanalyzer(bt.analyzers.DrawDown)

    # Let's say that we have 0.25% slippage and commission per trade,
    # that is 0.5% in total for a round trip.
    #cerebro.broker.setcommission(commission=0.0025, margin=False)

    # Run over everything
    #strats = cerebro.run()

    #profit = cerebro.broker.getvalue() - initial_capital
    #max_dd = strats[0].analyzers.drawdown.get_analysis()["max"]["moneydown"]
    fitness = RunBackTest("testp",strategy_params)#profit / (max_dd if max_dd > 0 else 1)
    csvAppend("Outputs.csv", strategy_params, fitness)
    #if log:
    #    print(f"Starting Portfolio Value: {initial_capital:,.2f}")
    #    print(f"Final Portfolio Value:    {cerebro.broker.getvalue():,.2f}")
    #    print(f"Total Profit:             {profit:,.2f}")
    #    print(f"Maximum Drawdown:         {max_dd:,.2f}")
    #    print(f"Profit / Max DD:          {fitness}")

    #if plot:
    #    cerebro.plot()

    return [fitness]


# our fitness score is supposed to be maximised and there is only 1 objective
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# our individual is a list of genes, with the fitness score the higher the better
creator.create("Individual", list, fitness=creator.FitnessMax)

# register some handy functions for calling
toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(NPOP), NPOP)
# crossover strategy
toolbox.register("mate", tools.cxUniform, indpb=CXPB)
# mutation strategy
toolbox.register("mutate", tools.mutUniformInt, low=1, up=151, indpb=0.2)
# selection strategy
toolbox.register("select", tools.selTournament, tournsize=3)
# fitness function
toolbox.register("evaluate", evaluate)

# definition of an individual & a population
toolbox.register("attr_fast_period", random.randint, 1, 51)
toolbox.register("attr_slow_period", random.randint, 10, 151)
toolbox.register("attr_rsi_sell", random.randint, 1, 101)
toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    (
        toolbox.attr_fast_period,
        toolbox.attr_slow_period,
        toolbox.attr_rsi_sell,
    ),
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

mean = np.ndarray(NGEN)
best = np.ndarray(NGEN)
hall_of_fame = tools.HallOfFame(maxsize=3)

t = time.perf_counter()
pop = toolbox.population(n=NPOP)
for g in trange(NGEN):
    #arr = []
    #arr.append
    #df =pd.DataFrame([g]])
    #df.to_csv("Outputs.csv", mode='a', index=False, header=False)
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Apply mutation on the offspring
    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # The population is entirely replaced by the offspring
    pop[:] = offspring
    hall_of_fame.update(pop)
    print(
        "HALL OF FAME:\n"
        + "\n".join(
            [
                f"    {_}: {ind}, Fitness: {ind.fitness.values[0]}"
                for _, ind in enumerate(hall_of_fame)
            ]
        )
    )

    fitnesses = [
        ind.fitness.values[0] for ind in pop if not np.isinf(ind.fitness.values[0])
    ]
    mean[g] = np.mean(fitnesses)
    best[g] = np.max(fitnesses)

end_t = time.perf_counter()
print(f"Time Elapsed: {end_t - t:,.2f}")

fig, ax = plt.subplots(sharex=True, figsize=(16, 9))

sns.lineplot(x=range(NGEN), y=mean, ax=ax, label="Average Fitness Score")
sns.lineplot(x=range(NGEN), y=best, ax=ax, label="Best Fitness Score")
ax.set_title("Fitness Score")
ax.set_xticks(range(NGEN))
ax.set_xlabel("Iteration")

plt.tight_layout()
plt.show()

OPTIMISED_STRATEGY_PARAMS = {k: v for k, v in zip(PARAM_NAMES, hall_of_fame[0])}
RunBackTest("projecctName",**OPTIMISED_STRATEGY_PARAMS)

#if __name__ == "__main__":
#    changeParams("testp", 5, 10, 30)