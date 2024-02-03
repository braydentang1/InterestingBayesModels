## BG-NBD model

- Mostly used for customer lifetime value stuff.
    - i.e. frequency = number of repeat transactions at times T1, ...Tx
    - recency = time between first and most recent transaction within study period
    - age = time between first transaction and end date of study period
- Transactions occur via. a Poisson process, i.e. arrival times are exponentially distributed and memoryless
- Probability of "dying"/being inactive is p, assumed to happen after every transaction (shifted geometric distribution)
- At the end date of the study period, user is either alive with probability (1-p) and we have yet to observe their next transaction arrival (i.e. they are censored), or they are dead/inactive.
