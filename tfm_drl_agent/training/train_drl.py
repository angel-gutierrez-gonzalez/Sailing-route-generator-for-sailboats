import os
import csv
import numpy as np

def train_ddpg(agent, env, episodes=300, batch_size=64, buffer_capacity=100000, save_interval=50):
    replay_buffer = []
    rewards_history = []

    save_path = "models/ddpg_checkpoint"
    latest_path = "models/ddpg_latest"
    log_path = "models/ddpg_latest/reward_log.csv"

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(latest_path, exist_ok=True)

    with open(log_path, mode='w', newline='') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["Episode", "Reward"])

        for ep in range(episodes):
            state = env.reset()
            episodic_reward = 0

            while True:
                action = agent.policy(state)
                action += agent.noise()
                action = np.clip(action, env.action_space.low, env.action_space.high)

                next_state, reward, done, _ = env.step(action)
                replay_buffer.append((state, action, reward, next_state, float(done)))

                if len(replay_buffer) > buffer_capacity:
                    replay_buffer.pop(0)

                state = next_state
                episodic_reward += reward

                if len(replay_buffer) >= batch_size:
                    agent.learn(replay_buffer, batch_size)

                if done:
                    break

            rewards_history.append(episodic_reward)
            print(f"Episode {ep+1}: Reward = {episodic_reward:.2f}, Buffer = {len(replay_buffer)}")
            writer.writerow([ep + 1, episodic_reward])

            if (ep + 1) % save_interval == 0:
                actor_path = os.path.join(save_path, f"actor_ep{ep+1}.h5")
                critic_path = os.path.join(save_path, f"critic_ep{ep+1}.h5")
                agent.actor_model.save(actor_path)
                agent.critic_model.save(critic_path)

        final_actor_path = os.path.join(latest_path, f"actor_final_ep{episodes}.h5")
        final_critic_path = os.path.join(latest_path, f"critic_final_ep{episodes}.h5")
        agent.actor_model.save(final_actor_path)
        agent.critic_model.save(final_critic_path)

    return rewards_history
