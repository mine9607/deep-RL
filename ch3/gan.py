import argparse
import logging
import random
import time
import typing as tt

import ale_py
import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from gymnasium import spaces
from torch.utils.tensorboard.writer import SummaryWriter

# Register gymnasium environments from secondary pip installs
gym.register_envs(ale_py)

# Setup Python root logger
# NOTE--Log Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
"""
%(asctime)s -> timestamp of the log record
%(levelname)s -> the level string, e.g. INFO, ERROR, etc
%(message)s -> the actual log message
"""
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Get a logger named after the current module (__name__) e.g. ("my_module")
# Avoids every module writing to the root logger directly
log = logging.getLogger(__name__)

# DEFINE GLOBAL VARIABLES
LATENT_VECTOR_SIZE = 100
DISCR_FILTERS = 64  # discriminator class
GENER_FILTERS = 64  # generator class
BATCH_SIZE = 16

# dimension input image will be rescaled
IMAGE_SIZE = 64

LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000
MAX_ITERATIONS = 30000

if torch.cuda.is_available():
    device = torch.device("cuda")


class InputWrapper(gym.ObservationWrapper):
    """
    Preprocessing of input numpy array
    1. resize image into predefined size
    2. move color channel axis to first place
    """

    def __init__(self, *args):
        super().__init__(*args)
        old_space = self.observation_space
        assert isinstance(old_space, spaces.Box)
        self.observation_space = spaces.Box(
            self.observation(old_space.low),
            self.observation(old_space.high),
            dtype=np.float32,
        )

    def observation(self, observation: gym.core.ObsType) -> gym.core.ObsType:
        # resize image
        new_obs = cv2.resize(observation, (IMAGE_SIZE, IMAGE_SIZE))
        # transform (w, h, c) -> (c, w, h)
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32)  # cast image from bytes to float


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        # this pipe converges image into the single number output

        self.conv_pipe = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape[0],
                out_channels=DISCR_FILTERS,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=DISCR_FILTERS,
                out_channels=DISCR_FILTERS * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(DISCR_FILTERS * 2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=DISCR_FILTERS * 2,
                out_channels=DISCR_FILTERS * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(DISCR_FILTERS * 4),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=DISCR_FILTERS * 4,
                out_channels=DISCR_FILTERS * 8,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(DISCR_FILTERS * 8),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=DISCR_FILTERS * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        conv_out = self.conv_pipe(x)
        # -1 : automatically calculated dimension (based on total number of elements)
        #  1 : second dimension of size 1
        # .squeeze() : removes that dimension of size 1
        return conv_out.flatten()


class Generator(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        # pipe deconvolves input vector into (3, 64, 64) image
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=LATENT_VECTOR_SIZE,
                out_channels=GENER_FILTERS * 8,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(GENER_FILTERS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=GENER_FILTERS * 8,
                out_channels=GENER_FILTERS * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=GENER_FILTERS * 4,
                out_channels=GENER_FILTERS * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=GENER_FILTERS * 2,
                out_channels=GENER_FILTERS,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=GENER_FILTERS,
                out_channels=output_shape[0],
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.pipe(x)


def iterate_batches(
    envs: tt.List[gym.Env], batch_size: int = BATCH_SIZE
) -> tt.Generator[torch.Tensor, None, None]:
    batch = [e.reset()[0] for e in envs]
    env_gen = iter(lambda: random.choice(envs), None)

    while True:
        e = next(env_gen)
        action = e.action_space.sample()
        obs, reward, is_done, is_trunc, _ = e.step(action)
        if np.mean(obs) > 0.01:
            batch.append(obs)
        if len(batch) == batch_size:
            batch_np = np.array(batch, dtype=np.float32)
            # normalizing input to [-1, 1] and convert to tensor
            yield torch.tensor(batch_np * 2.0 / 255.0 - 1.0)
            batch.clear()
        if is_done or is_trunc:
            e.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cuda", help="Device name, default=cuda")
    args = parser.parse_args()

    device = torch.device(args.dev)

    envs = [
        InputWrapper(gym.make(name))
        for name in (
            "Breakout-v4",
            "AirRaid-v4",
            "Pong-v4",
        )
    ]
    shape = envs[0].observation_space.shape

    net_discr = Discriminator(input_shape=shape).to(device)
    net_gener = Generator(output_shape=shape).to(device)

    objective = nn.BCELoss()
    gen_optimizer = optim.Adam(
        params=net_gener.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
    )
    dis_optimizer = optim.Adam(
        params=net_discr.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999)
    )

    writer = SummaryWriter()

    gen_losses = []
    dis_losses = []
    iter_no = 0

    true_labels_v = torch.ones(BATCH_SIZE, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, device=device)
    ts_start = time.time()

    for batch_v in iterate_batches(envs):
        # fake samples, input is 4D: batch, filters, x, y
        gen_input_v = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
        gen_input_v.normal_(0, 1)
        gen_input_v = gen_input_v.to(device)
        batch_v = batch_v.to(device)
        gen_output_v = net_gener(gen_input_v)

        # train discriminator
        dis_optimizer.zero_grad()
        dis_output_true_v = net_discr(batch_v)
        dis_output_fake_v = net_discr(gen_output_v.detach())
        dis_loss = objective(dis_output_true_v, true_labels_v) + objective(
            dis_output_fake_v, fake_labels_v
        )
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())

        # train generator
        gen_optimizer.zero_grad()
        dis_output_v = net_discr(gen_output_v)
        gen_loss_v = objective(dis_output_v, true_labels_v)
        gen_loss_v.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss_v.item())

        iter_no += 1

        # Check for max iterations
        if iter_no >= MAX_ITERATIONS:
            log.info(
                "Reached maximum iterations (%d), stopping training", MAX_ITERATIONS
            )
            break

        if iter_no % REPORT_EVERY_ITER == 0:
            dt = time.time() - ts_start
            log.info(
                "Iter %d in %.2fs: gen_loss=%.3e, dis_loss=%.3e",
                iter_no,
                dt,
                np.mean(gen_losses),
                np.mean(dis_losses),
            )
            ts_start = time.time()
            writer.add_scalar("gen_loss", np.mean(gen_losses), iter_no)
            writer.add_scalar("dis_loss", np.mean(dis_losses), iter_no)
            gen_losses = []
            dis_losses = []
        if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
            img = vutils.make_grid(gen_output_v.data[:64], normalize=True)
            writer.add_image("fake", img, iter_no)
            img = vutils.make_grid(batch_v.data[:64], normalize=True)
            writer.add_image("real", img, iter_no)
