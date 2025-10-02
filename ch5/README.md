# Tabular Learning and the Bellman Equation

## Value, state and optimality

- Value: the expected total reward (optionally discounted) that is obtainable from the state

Value is always calculated in terms of some policy that our agent follows.

The `reward` is an instantaneous value received by the agent upon taking an action.

`Value` is essentially the sum of the rewards that can be achieved from that state weighted by the probability that those actions will occur until the end of an episode. In other words is the the sum of the proability weighted rewards for all paths that can be taken from the current state.

Ex: Let's say we are in state A and can move to state B or state C with 50% likelihood. The value of state A becomes:

$$
V_A = 0.5(r_{A->B}) + 0.5(r_{A->C})
$$

## Bellman Equation of Optimality

Bellman showed that if we consider not just the immediate reward of getting to the next state but also the long-term value of that new state then we can always choose the best state (optimal)

In the `deterministic` case we only consider the immediate reward + the value of the next state we can choose the optimal state.

In the `stochastic` case we simply introduce probability (p_i) that an action (a_i) will be taken to receive the immediate reward (r_i) and get to a state with value (V_i)

<img src='./Stochastic_Bellman.png' height="200"/>
