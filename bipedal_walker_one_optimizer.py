# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from typing import NamedTuple
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


class Transition(NamedTuple):
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool
    complementary_info: dict[str, torch.Tensor] = None


class BatchTransition(NamedTuple):
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    done: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int, device: str = "cuda:0"):
        self.capacity = capacity
        self.device = device
        self.memory: list[Transition] = []
        self.position = 0

    def add(self, *args, **kwargs):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args, **kwargs)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> BatchTransition:
        list_of_transitions = random.sample(self.memory, batch_size)
        batch_obs = torch.cat([t.state for t in list_of_transitions]).to(self.device)
        batch_actions = torch.stack([t.action for t in list_of_transitions]).to(
            self.device
        )
        batch_rewards = torch.tensor(
            [t.reward for t in list_of_transitions], dtype=torch.float32
        ).to(self.device)
        batch_next_obs = torch.cat([t.next_state for t in list_of_transitions]).to(
            self.device
        )
        batch_dones = torch.tensor(
            [t.done for t in list_of_transitions], dtype=torch.float32
        ).to(self.device)
        return BatchTransition(
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones
        )


def make_env(seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(
                "BipedalWalker-v3",
                render_mode="rgb_array",
            )
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(
                "BipedalWalker-v3",
            )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.observation_space.shape).prod()
            + np.prod(env.action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"bipedal_walker__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.seed, 0, args.capture_video, run_name)]
    # )
    env = make_env(args.seed, 0, args.capture_video, run_name)()
    assert isinstance(
        env.action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_action = float(env.action_space.high[0])

    actor = Actor(env).to(device)
    qf1 = SoftQNetwork(env).to(device)
    qf2 = SoftQNetwork(env).to(device)
    qf1_target = SoftQNetwork(env).to(device)
    qf2_target = SoftQNetwork(env).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    # Automatic entropy tuning
    optimizers_parameters = [
        {"params": list(actor.parameters()), "lr": args.policy_lr},
        {"params": list(qf1.parameters()) + list(qf2.parameters()), "lr": args.q_lr},
    ]
    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(env.action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        optimizers_parameters.append({"params": [log_alpha], "lr": args.q_lr})
    else:
        alpha = args.alpha

    optimizer = optim.Adam(params=optimizers_parameters)
    env.observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        device,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = env.reset(seed=args.seed)
    sum_episode_return = 0
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = env.action_space.sample()
        else:
            actions, _, _ = actor.get_action(
                torch.Tensor(obs).unsqueeze(dim=0).to(device)
            )
            actions = actions.squeeze(dim=0).detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        sum_episode_return += rewards

        obs = torch.Tensor(obs).unsqueeze(dim=0).to(device)
        real_next_obs = torch.Tensor(next_obs).unsqueeze(dim=0).to(device)
        actions = torch.Tensor(actions).to(device)
        rb.add(
            state=obs,
            next_state=real_next_obs,
            action=actions,
            reward=rewards,
            done=terminations,
        )

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        if terminations or truncations:
            print(f"global_step={global_step}, episodic_return={sum_episode_return} ")
            writer.add_scalar("charts/episodic_return", sum_episode_return, global_step)
            sum_episode_return = 0
            obs, infos = env.reset(seed=args.seed)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        # for idx, trunc in enumerate(truncations):
        #     if trunc:
        #         real_next_obs[idx] = infos["final_observation"][idx]

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data.next_state
                )
                qf1_next_target = qf1_target(data.next_state, next_state_actions)
                qf2_next_target = qf2_target(data.next_state, next_state_actions)
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - alpha * next_state_log_pi
                )
                next_q_value = data.reward.flatten() + (
                    1 - data.done.flatten()
                ) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.state, data.action).view(-1)
            qf2_a_values = qf2(data.state, data.action).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value.detach())
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value.detach())
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            optimizer.zero_grad()
            # qf_loss.backward()
            # q_optimizer.step()
            final_loss = qf_loss

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.state)
                    qf1_pi = qf1(data.state, pi)
                    qf2_pi = qf2(data.state, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi.detach()).mean()

                    # actor_optimizer.zero_grad()
                    # actor_loss.backward()
                    # actor_optimizer.step()
                    final_loss += actor_loss

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.state)
                        alpha_loss = (
                            -log_alpha.exp() * (log_pi + target_entropy).detach()
                        ).mean()

                        # a_optimizer.zero_grad()
                        # alpha_loss.backward()
                        # a_optimizer.step()
                        final_loss += alpha_loss

            final_loss.backward()
            optimizer.step()

            alpha = log_alpha.exp().item()
            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if global_step % 100 == 0:
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                writer.add_scalar(
                    "losses/qf2_values", qf2_a_values.mean().item(), global_step
                )
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                writer.add_scalar(
                    "losses/sum_all_losses",
                    scalar_value=qf_loss.item() + actor_loss.item() + alpha_loss.item(),
                    global_step=global_step,
                )
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )

    env.close()
    writer.close()
