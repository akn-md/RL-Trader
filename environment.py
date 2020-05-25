"""
Trading Environment

State = portfolio data, OHLCV + other TA
Actions = buy, sell, hold
Reward = change in portoflio value after each action

TODO: Make action space continuous to allow for buying/selling different amounts of shares
TODO: Allow for scaling in and out of trades

"""

import api_helpers as helpers
import pandas as pd
import numpy as np


class TradeEnv:
    def __init__(self, memory):
        self.memory = memory
        self.init_cash = 100000
        self.cash = self.init_cash  # liquid equity
        self.value = self.init_cash  # total equity
        self.pnl = 0.0  # PnL
        self.vol = 1000  # constant for now

    def get_state(self):
        # Assumes each row in self.data contains all input features to NN
        row = self.data.iloc[self.t].values.flatten().tolist()
        # state = [self.position, self.value, self.pnl] + row
        state = [self.pnl] + row
        return state

    def reset(self, data):
        self.data = data  # new stock data
        self.t = 0  # start at 09:30 EST
        self.done = False
        self.position = 0  # entry price of current position
        self.buys = []
        self.sells = []
        # do we need this?
        self.history = [0 for _ in range(self.memory)]  # history of previous states

        # for now, reset balances
        self.init_cash = 100000
        self.cash = self.init_cash  # liquid equity
        self.value = self.init_cash  # total equity
        self.pnl = 0.0  # PnL

        return self.get_state()

    def get_pnl(self):
        return (self.cash - self.init_cash) / self.cash

    # Old
    # def calc_pnl(self):
    #     gain = 0
    #     if self.position > 0:
    #         # Calculate realized/unrealized gain based on close
    #         # This keeps pnl most uptodate
    #         gain = self.vol * self.data.iloc[self.t, :]['close']

    #     self.value = self.cash + gain
    #     pnl = (self.value - self.init_cash) / self.init_cash

    #     return pnl

    # Current trade PnL, returns 0 between trades
    # def calc_reward(self, action):
    #     pnl = 0.0
    #     if self.position > 0:
    #         gain = 0.0
    #         if action == 0:
    #             gain = self.data.iloc[self.t, :]['open'] - self.position
    #         else:
    #             gain = self.data.iloc[self.t, :]['close'] - self.position
    #         pnl = gain / self.position

    #     # Capping PnL
    #     if pnl > 1:
    #         pnl = 1
    #     elif pnl < -1:
    #         pnl = -1
    #     return pnl

    # Overall PnL
    def calc_reward(self, action):
        old_pnl = self.pnl
        gain = 0
        if self.position > 0:
            # Calculate realized/unrealized gain based on close
            # This keeps pnl most uptodate
            gain = 0.0
            if action == 0:
                gain = self.vol * self.data.iloc[self.t, :]['open']
            else:
                gain = self.vol * self.data.iloc[self.t, :]['close']
        self.value = self.cash + gain
        self.pnl = (self.value - self.init_cash) / self.init_cash

        # Just PnL
        pnl = self.pnl

        # Diff
        # pnl = self.pnl - old_pnl

        # 2nd derivative
        # pnl = 0.0
        # if old_pnl == 0.0:
        #     if self.pnl > 0:
        #         pnl = 1.0
        #     else:
        #         pnl = -1.0
        # else:
        #     pnl = (self.pnl - old_pnl)/old_pnl

        # Capping PnL
        # if pnl < -1:
        #     pnl = -1
        # elif pnl > 1:
        #     pnl = 1

        return pnl

    def step(self, action):
        # Agent used t-1 to make a decision
        # Agent now performs action in t
        self.t += 1
        reward = 0.0

        # 0 = sell, 1 = hold, 2 = buy
        if action == 2:
            op = self.data.iloc[self.t, :]['open']
            if self.position > 0:
                reward = self.calc_reward(action)  # punish if buying more
            else:
                # Buy at open
                # print("Position = " + str(self.position))
                # print("Buying at " + str(self.t))
                self.position = op
                self.cash -= self.vol * self.position
                reward = self.calc_reward(action)
                self.buys.append((self.t, op))
        elif action == 0:  # sell
            op = self.data.iloc[self.t, :]['open']
            if self.position == 0:
                reward = self.calc_reward(action)  # punish if selling without a position
            else:
                # sell at open
                # print("Selling at " + str(self.t))
                reward = self.calc_reward(action)
                self.cash += self.vol * op
                self.position = 0
                self.sells.append((self.t, op))
        else:  # hold
            reward = self.calc_reward(action)
        if (self.t == len(self.data) - 1):
            if self.position > 0:
                reward = -1  # punish if holding through close
            self.done = True

        state = self.get_state()

        return state, reward, self.done


# Test sim of env
if __name__ == '__main__':
    date = pd.to_datetime("05/21/2020")
    data = helpers.get_min_data("KDMN", date)
    env = TradeEnv(1)
    print(data)
    print(env.reset(data))
    for _ in range(100):
        pact = np.random.randint(3)
        print(pact)
        print(env.step(pact))
