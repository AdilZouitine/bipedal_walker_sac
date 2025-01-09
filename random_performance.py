from statistics import mean, stdev
import gymnasium as gym

env = gym.make(
    "BipedalWalker-v3",
    render_mode="rgb_array",
)


nb_episodes = 1000
episode_sum_rewards = []
for episode in range(nb_episodes):
    sum_rewards = 0
    obs, info = env.reset()
    done = False
    truncated = False
    while not done or truncated:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        sum_rewards += reward
        if done or truncated:
            break
    episode_sum_rewards.append(sum_rewards)

print(f"Mean reward over {nb_episodes} episodes: {mean(episode_sum_rewards)}")
print(f"Standard deviation of rewards: {stdev(episode_sum_rewards)}")

# Mean reward over 1000 episodes: -99.21397399902344
# Standard deviation of rewards: 13.913405594052428
