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

<img src='./Stochastic_Bellman.png' height="300"/>

> NOTE: The long-term value of a state can be discounted. Similar to discounting EV of a project by discounting back to NPV. Essentially, this could be thought of similarly to the opportunity to implement 3 different oil and gas projects which have a probability of success an an expected cashflow which could be compared by discounting back to the current state.

**Summary**: At every state the agent ends up in, it needs to select the action with the **_maximum expected reward_**--which is a sum of the immediate reward and the one-step discounted long-term reward.

## Value of the Action

The `value of the action` Q(s,a) -- is the total reward **V(s)** we can get by executing action _a_ in state _s_

Since we always want to maximize this value we can define **V(s)** as the maximum available action value:

$$
V(s) = {max}Q(s,a)
$$

The value of some state equals the value of the maximum action we can execute from this state.

We can recursively (think walking a tree-structure) express Q(s,a) such as:

$$
Q(s,a) = r(s,a) + \gamma \, \text{max} \, Q(s', a')
$$

Note: the value of terminal states (leaf nodes) is equal to their immediate reward (r_i) since they have no further actions that can move them to a new state.

Example: Simple grid-like environment

<img src='./simple_grid.png' height="200"/>

Transition Diagram:

<img src='./transition_diagram.png' height="300"/>

For the diagram above the `action values Q(s,a)` are given by:

$$
Q(s_0, \text{up}) = 0.33 \cdot V_1 + 0.33 \cdot V_2 + 0.33 \cdot V_4 = 0.33 \cdot 1 + 0.33 \cdot 2 + 0.33 \cdot 4 = 2.31
$$

$$
Q(s_0, \text{left}) = 0.33 \cdot V_1 + 0.33 \cdot V_2 + 0.33 \cdot V_3 = 0.33 \cdot 1 + 0.33 \cdot 2 + 0.33 \cdot 3 = 1.98
$$

$$
Q(s_0, \text{right}) = 0.33 \cdot V_4 + 0.33 \cdot V_1 + 0.33 \cdot V_3 = 0.33 \cdot 4 + 0.33 \cdot 1 + 0.33 \cdot 3 = 2.64
$$

$$
Q(s_0, \text{down}) = 0.33 \cdot V_3 + 0.33 \cdot V_2 + 0.33 \cdot V_4 = 0.33 \cdot 3 + 0.33 \cdot 2 + 0.33 \cdot 4 = 2.97
$$

The final value for state **s_0** is the maximum of the possible action values which is **2.97**

> Q-values are more convenient in practice--it's simpler for the agent to make decisions about actions based on Q than on V

- In the case of Q--to choose the action based on state, the agent just needs to calculate Q for all available actions using the current state and choose the action with the largest value of Q.

- To do the same using values of the states--the agent needs to know not only the values, but also the probabilities for transitions (which are rarely known in advance in practice)--which must be estimated for every action and state pair.

# ??? THIS STATEMENT ABOVE IS CONFUSING ME ????

- If the model is trained via **state-value based learning V(s)** then the agent will need to perform a calculation for each action at each step using known transition probabilities

  - Agent estimates how good each **state** is on average
  - To pick an action it must calculate Q(s,a) = P(r + gammaV)
  - Then it can choose an action based on max[Q(s,a)]

- If the model is trained via **action-value based learning Q(s,a)** then the agent simply needs to choose
  - Agent estimates how good each action in a state is (expected return)
  - Agent simply chooses action based on max[Q(s,a)] -- no calculation of action values needed

## Value Iteration Method

`value iteration` - allows us to numerically calculate the values of the states and values of the actions of `Markov decision processes (MDPs)` with known transition probabilities and rewards.

1. Initialize the values of all states, V_i, to some initial value (usually zero)
2. For every state, s, in the MDP, perform the Bellman update:

$$
V_s \leftarrow \text{max} \sum{p_{a,s \rightarrow s'} \, (r_{s,a,s'} + \gamma \,V_{s'})}
$$

## Q-iteration for Frozen Lake
