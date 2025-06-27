import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class DDPGAgent:
    def __init__(self, env, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.action_bounds = env.action_space.high

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.shape[0]

        self.actor_model = get_actor(num_states, self.action_bounds)
        self.critic_model = get_critic(num_states, num_actions)

        self.target_actor = get_actor(num_states, self.action_bounds)
        self.target_critic = get_critic(num_states, num_actions)

        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

        self.noise_std = 0.1

    def policy(self, state):
        state = np.expand_dims(state, axis=0)
        action = self.actor_model(state, training=False).numpy()[0]
        return action

    def noise(self):
        return np.random.normal(0, self.noise_std, size=self.env.action_space.shape)

    def learn(self, replay_buffer, batch_size):
        indices = np.random.choice(len(replay_buffer), size=batch_size)
        batch = [replay_buffer[i] for i in indices]

        states = np.array([sample[0] for sample in batch])
        actions = np.array([sample[1] for sample in batch])
        rewards = np.array([sample[2] for sample in batch])
        next_states = np.array([sample[3] for sample in batch])
        dones = np.array([sample[4] for sample in batch])

        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        next_actions = self.target_actor(next_states, training=False)
        next_qs = self.target_critic([next_states, next_actions], training=False)
        targets = rewards + self.gamma * (1 - dones) * tf.squeeze(next_qs)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)

        with tf.GradientTape() as tape:
            current_qs = tf.squeeze(self.critic_model([states, actions], training=True))
            critic_loss = tf.keras.losses.MeanSquaredError()(targets, current_qs)
        critic_grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions_pred = self.actor_model(states, training=True)
            critic_value = self.critic_model([states, actions_pred], training=True)
            actor_loss = -tf.reduce_mean(critic_value)
        actor_grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

        self.update_target(self.target_actor.variables, self.actor_model.variables)
        self.update_target(self.target_critic.variables, self.critic_model.variables)

    def update_target(self, target_weights, source_weights):
        for (target, source) in zip(target_weights, source_weights):
            target.assign(self.tau * source + (1 - self.tau) * target)

def get_actor(input_shape, action_bounds):
    inputs = layers.Input(shape=(input_shape,))
    out = layers.Dense(400, activation="relu")(inputs)
    out = layers.Dense(300, activation="relu")(out)
    outputs = layers.Dense(len(action_bounds), activation="tanh")(out)
    model = tf.keras.Model(inputs, outputs)
    return model

def get_critic(state_shape, action_shape):
    state_input = layers.Input(shape=(state_shape,))
    action_input = layers.Input(shape=(action_shape,))
    concat = layers.Concatenate()([state_input, action_input])

    out = layers.Dense(400, activation="relu")(concat)
    out = layers.Dense(300, activation="relu")(out)
    outputs = layers.Dense(1)(out)
    model = tf.keras.Model([state_input, action_input], outputs)
    return model

