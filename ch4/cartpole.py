import typing as tt
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

HIDDEN_SIZE = 128  # number of neurons in hidden layer
BATCH_SIZE = 16  # number of episodes per batch
PERCENTILE = 70  # acceptance cutoff for episode reward for self-improvement


class Net(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


@dataclass
class EpisodeStep:
    observation: np.ndarray
    action: int


@dataclass
class Episode:
    reward: float
    steps: tt.List[EpisodeStep]


def iterate_batches(
    env: gym.Env, net: Net, batch_size: int
) -> tt.Generator[tt.List[Episode], None, None]:
    """
    Generate batches of completed episodes by interacting with the environment
    using the given policy network.

    The function continuously samples actions from the network’s softmax
    probability distribution, steps the environment, accumulates rewards,
    and groups finished episodes into batches of size `batch_size`.

    Parameters
    ----------
    env : gym.Env
        The Gymnasium environment to interact with.
    net : Net
        The neural network policy model producing action probabilities.
    batch_size : int
        Number of completed episodes per yielded batch.

    Yields
    ------
    List[Episode]
        A list of `Episode` objects, each containing the total reward
        and sequence of steps (observations and actions) for that episode.
    """
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs, _ = env.reset()
    sm = nn.Softmax(dim=1)

    while True:
        obs_v = torch.tensor(obs, dtype=torch.float32)
        act_probs_v = sm(net(obs_v.unsqueeze(0)))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, is_trunc, _ = env.step(action)
        episode_reward += float(reward)
        step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(step)
        if is_done or is_trunc:
            e = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(e)
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(
    batch: tt.List[Episode], percentile: float
) -> tt.Tuple[torch.FloatTensor, torch.LongTensor, float, float]:
    """
    Select training data from a batch of episodes based on a reward percentile cutoff.

    Episodes with total reward below the cutoff are discarded. The remaining
    episodes are decomposed into observation-action pairs for supervised
    policy training. Additionally, the function computes the reward boundary
    (percentile threshold) and the mean reward across the batch.

    Parameters
    ----------
    batch : List[Episode]
        A list of completed episodes.
    percentile : float
        The reward percentile to use as the cutoff. Episodes with rewards
        greater than or equal to this value are kept.

    Returns
    -------
    observations : torch.FloatTensor
        A tensor of stacked observations from the selected episodes.
    actions : torch.LongTensor
        A tensor of corresponding actions taken at each observation.
    reward_bound : float
        The cutoff reward value at the given percentile.
    reward_mean : float
        The mean reward across all episodes in the batch.
    """
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = float(np.percentile(rewards, percentile))
    reward_mean = float(np.mean(rewards))

    train_obs: tt.List[np.ndarray] = []
    train_act: tt.List[int] = []
    for episode in batch:
        if episode.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, episode.steps))
        train_act.extend(map(lambda step: step.action, episode.steps))

    train_obs_v = torch.FloatTensor(np.vstack(train_obs))
    train_act_v = torch.LongTensor(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    assert env.observation_space.shape is not None
    obs_size = env.observation_space.shape[0]
    assert isinstance(env.action_space, gym.spaces.Discrete)
    n_actions = int(env.action_space.n)

    # 1 - Instantiate the neural net model
    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    print(net)

    # 2 - Define the objective function
    objective = nn.CrossEntropyLoss()

    # 3 - Define the optimization algorithm
    optimizer = optim.Adam(params=net.parameters(), lr=0.01)

    # Instantiate the Tensorboard writer
    writer = SummaryWriter(comment="-cartpole")

    # MAIN LOGIC
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):

        # Take each batch and filter it to only the completed episodes which meet the threshold criteria
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)

        # Clear the optimizer vector gradients (weights and bias)
        optimizer.zero_grad()

        # 1 - Get the NN output of environment action scores (foward pass)
        # Forward pass: compute action logits (unnormalized scores) from observations
        action_scores_v = net(obs_v)

        # 2 - Compute the cross-entropy loss between predicted action logits and true actions
        loss_v = objective(action_scores_v, acts_v)

        # 3 - Compute the gradient (backprop)
        loss_v.backward()

        # 4 - Update the network parameters using the computed gradients (weights and bias)
        optimizer.step()

        # Log Epoch data to console
        print(
            "%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f"
            % (iter_no, loss_v.item(), reward_m, reward_b)
        )

        # Write scalar output values to Tensorboard
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_b, iter_no)
        writer.add_scalar("reward_mean", reward_m, iter_no)
        if reward_m > 475:
            print("Solved!")
            break
        writer.close()
