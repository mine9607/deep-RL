# Tabular Learning and the Bellman Equation

## Value, state and optimality

- Value: the expected total reward (optionally discounted) that is obtainable from the state

Value is always calculated in terms of some policy that our agent follows.

> NOTE: The `reward` is an instantaneous value received by the agent upon taking an action. `Value` is essentially the sum of the rewards that can be achieved from that state weighted by the probability that those actions will occur.

Ex: Let's say we are in state A and can move to state B or state C with 50% likelihood. The value of state A becomes:

$$
V_A = 0.5 \cdot r_{A->B} + 0.5 \cdot r_{A->C}
$$

- State:
- Optimality:
