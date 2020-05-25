"""
Actor Critic with a custom trading environment
Based on https://github.com/keras-team/keras-io

"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from environment import TradeEnv
import api_helpers as helpers
from sklearn import preprocessing
import matplotlib.pyplot as plt
import vis_utils as vis
import wandb

# Hyperparameters
num_episodes = 201  # Number of trading days to train on
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 390  # Number of minutes market is open
eps = np.finfo(
    np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()

# AC params
num_actions = 3

# Logging
log = True
run_name = "PnL-2"
group = "1H"


def train_episode(env, verbose, ep):
    action_probs_history = []
    critic_value_history = []
    rewards_history = []

    state = env.get_state()

    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities (from actor) and
            # Future expected reward (from critic) based on current state
            action_probs, critic_value = model(state)

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))

            # Keep track of log action prob and critic estimate
            action_probs_history.append(tf.math.log(action_probs[0, action]))
            critic_value_history.append(critic_value[0, 0])

            # Apply the sampled action in our environment
            # Obtain reward and new state from environment
            state, reward, done = env.step(action)

            # Keep track of actual reward and total ep reward
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        qvals = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            qvals.insert(0, discounted_sum)

        # Normalize
        qvals = np.array(qvals)
        qvals = (qvals - np.mean(qvals)) / (np.std(qvals) + eps)
        qvals = qvals.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, qvals)
        actor_losses = []
        critic_losses = []
        for log_prob, value, q in history:
            # value = predicted value by critic at this point in history
            # Think of this as the value of taking the average action at each state
            # log_prob = log probability of the action we took
            # q = actual total discounted reward we ended up receiving aka the Q value

            # How much better was this action compared to the average
            advantage = q - value

            # high prob action + low advantage = small loss
            # high prob action + high advantage = higher loss
            # low prob action + low advantage = higher loss
            # low prob action + high advantage = highest loss
            # Negative loss when critic's estimate > actual return

            # Actor continuously trying to learn high prob actions
            # that result in larger than expected rewards
            actor_loss = -log_prob * advantage
            actor_losses.append(actor_loss)  # actor loss

            # Critic is continusouly trying to learn the state-value
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(q, 0)))

            # print("q="+str(q))
            # print("v"+str(value))
            # print("logprob=" + str(log_prob))
            # print("actor loss="+str(actor_loss))
            # print("critic loss=" + str(huber_loss(tf.expand_dims(value, 0), tf.expand_dims(q, 0))))

        # Backpropagation
        actor_loss = sum(actor_losses)  #/ len(actor_losses)
        critic_loss = sum(critic_losses)  #/ len(critic_losses)
        loss_value = actor_loss + critic_loss
        # Compute the gradient of loss wrt model weights
        grads = tape.gradient(loss_value, model.trainable_variables)
        # Gradient descent
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Metrics
        if log:
            wandb.log({'Reward': episode_reward}, step=ep)
            wandb.log({'Actor loss': actor_loss}, step=ep)
            wandb.log({'Critic loss': critic_loss}, step=ep)
            wandb.log({'Overall loss': loss_value}, step=ep)

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()


def train(model, data):
    env = TradeEnv(1)
    verbose = False
    for x in range(1, num_episodes):
        print("Training on sine wave")
        if x == num_episodes:
            verbose = True
        env.reset(data)
        train_episode(env, verbose, x)
        if (x % 100 == 0) or (x == 1):
            # vis.plot_trades(env.data, env.buys, env.sells)
            plot = vis.get_trade_plot(env.data, env.buys, env.sells)
            if log:
                wandb.log({'Trades': plot}, step=x)
        template = "PnL: {:.6f} at episode {}"
        pnl = env.get_pnl()
        if log:
            wandb.log({'PnL': pnl}, step=x)
        print(template.format(pnl, x))

    # Just for testing, should sample ticker-date combos randomly for num_episodes
    # for x in range(1, num_episodes):
    #     tickers = data.ticker.unique()
    #     for ticker in tickers:
    #         dates = data[data.ticker == ticker].date.unique()
    #         for date in dates:
    #             print("Training on " + ticker + " on " + date)
    #             ep_data = data.loc[(data.ticker == ticker) & (data.date == date)]
    #             ep_data.iloc[0] = 0
    #             env.reset(ep_data.iloc[:, :-3])  # remove data/ticker cols
    #             train_episode(env)
    #             episode_count += 1
    #             template = "PnL: {:.6f} at episode {}"
    #             print(template.format(env.pnl, episode_count))


# Start with simple NN, try LSTM at some point
def create_model(num_inputs):
    num_actions = 3
    num_hidden = 128

    # Use one NN for both actor and critic (2 outputs)
    inputs = layers.Input(shape=(num_inputs, ))
    common = layers.Dense(num_hidden, activation="relu")(inputs)
    action = layers.Dense(num_actions, activation="softmax")(common)
    critic = layers.Dense(1)(common)

    return keras.Model(inputs=inputs, outputs=[action, critic])


if __name__ == '__main__':
    if log:
        wandb.init(project="rl-trader", name=run_name, group=group)
    # date = pd.to_datetime("05/21/2020")
    # data = helpers.get_min_data("KDMN", date)
    # data['ticker'] = "KDMN"
    # data['date'] = "05/21/2020"
    # data['open'] = data.open.pct_change(1)
    # data['high'] = data.high.pct_change(1)
    # data['low'] = data.low.pct_change(1)
    # data['close'] = data.close.pct_change(1)
    # data['volume'] = data.volume.pct_change(1)
    model = create_model(3)
    time = np.arange(0, 100, 1)
    wave = np.sin(time)
    wave = wave + 1
    # plt.plot(time, wave)
    # plt.show()
    data = pd.DataFrame({'close': wave, 'open': wave})
    # print(data)
    train(model, data)
