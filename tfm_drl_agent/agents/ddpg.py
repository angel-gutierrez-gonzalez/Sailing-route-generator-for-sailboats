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


def get_actor(input_shape, action_bounds):
    inputs = layers.Input(shape=input_shape)
    out = layers.Dense(400, activation="relu")(inputs)
    out = layers.Dense(300, activation="relu")(out)
    outputs = layers.Dense(len(action_bounds), activation="tanh")(out)
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(input_shape, action_dim):
    state_input = layers.Input(shape=input_shape)
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    action_input = layers.Input(shape=(action_dim,))
    action_out = layers.Dense(32, activation="relu")(action_input)

    concat = layers.Concatenate()([state_out, action_out])
    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    model = tf.keras.Model([state_input, action_input], outputs)
    return model


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

        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

        self.update_target(tau=1.0)  # Hard copy at init

        self.noise = OUActionNoise(mean=np.zeros(num_actions), std_deviation=0.2 * np.ones(num_actions))

    def update_target(self, tau=None):
        tau = tau or self.tau
        weights = zip(self.target_actor.weights, self.actor_model.weights)
        for (target_w, w) in weights:
            target_w.assign(tau * w + (1 - tau) * target_w)

        weights = zip(self.target_critic.weights, self.critic_model.weights)
        for (target_w, w) in weights:
            target_w.assign(tau * w + (1 - tau) * target_w)
