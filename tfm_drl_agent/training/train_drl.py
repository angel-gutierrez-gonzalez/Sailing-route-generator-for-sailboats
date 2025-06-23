import numpy as np
import tensorflow as tf
from collections import deque
import random
import os

def train_ddpg(agent, env, episodes=500, batch_size=64, buffer_capacity=100_000, save_path="models/ddpg_checkpoint", save_interval=30):
    os.makedirs(save_path, exist_ok=True)
    replay_buffer = deque(maxlen=buffer_capacity)
    rewards_history = []

    for ep in range(episodes):
        state = env.reset()
        episodic_reward = 0

        for step in range(env.config['max_steps']):
            tf_state = tf.convert_to_tensor([state], dtype=tf.float32)
            action = agent.actor_model(tf_state)[0].numpy()
            action += agent.noise()
            action = np.clip(action, env.action_space.low, env.action_space.high)

            next_state, reward, done, _ = env.step(action)
            replay_buffer.append((state, action, reward, next_state, float(done)))

            state = next_state
            episodic_reward += reward

            if done:
                break

            if len(replay_buffer) > batch_size:
                minibatch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

                states = tf.convert_to_tensor(states, dtype=tf.float32)
                actions = tf.convert_to_tensor(actions, dtype=tf.float32)
                rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
                next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
                dones = tf.convert_to_tensor(dones, dtype=tf.float32)

                # Critic update
                target_actions = agent.target_actor(next_states)
                y = rewards + agent.gamma * (1 - dones) * agent.target_critic([next_states, target_actions])[:, 0]

                with tf.GradientTape() as tape:
                    q_vals = agent.critic_model([states, actions])[:, 0]
                    critic_loss = tf.reduce_mean(tf.square(y - q_vals))
                grads = tape.gradient(critic_loss, agent.critic_model.trainable_variables)
                agent.critic_optimizer.apply_gradients(zip(grads, agent.critic_model.trainable_variables))

                # Actor update
                with tf.GradientTape() as tape:
                    actions_pred = agent.actor_model(states)
                    critic_val = agent.critic_model([states, actions_pred])
                    actor_loss = -tf.reduce_mean(critic_val)
                grads = tape.gradient(actor_loss, agent.actor_model.trainable_variables)
                agent.actor_optimizer.apply_gradients(zip(grads, agent.actor_model.trainable_variables))

                # Update target networks
                agent.update_target()

        rewards_history.append(episodic_reward)
        print(f"Episode {ep+1}: Reward = {episodic_reward:.2f}, Buffer = {len(replay_buffer)}")

        # Guardar modelo cada save_interval episodios
        if (ep + 1) % save_interval == 0:
            actor_path = os.path.join(save_path, f"actor_ep{ep+1}.h5")
            critic_path = os.path.join(save_path, f"critic_ep{ep+1}.h5")
            agent.actor_model.save(actor_path)
            agent.critic_model.save(critic_path)
            print(f"[INFO] Modelos guardados en episodio {ep+1}")

    return rewards_history
