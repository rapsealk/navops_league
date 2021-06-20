#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import pathlib
from time import time
from datetime import datetime

import seaborn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

seaborn.set_theme(style="darkgrid")


class ReportingBoard:

    def __init__(self, dirname=os.path.join(os.path.dirname(__file__), 'reportings')):
        self._id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self._dirname = os.path.abspath(os.path.join(dirname, self._id))
        if not os.path.exists(self.dirname):
            pathlib.Path(self.dirname).mkdir(parents=True, exist_ok=True)

    def plot(self, rewards):
        rewards_ = [rewards[0]]
        for i in range(1, len(rewards)):
            rewards_.append(rewards_[i-1] * 0.9 + rewards[i] * 0.1)
        df = pd.DataFrame({
            "Episode": range(1, len(rewards)+1),
            "Reward": rewards_
        })
        seaborn.lineplot(x='Episode', y='Reward', color='#FF6F00',
                         ci=95,
                         #hue='AverageValue',
                         #style='event',
                         data=df)
        plt.savefig(os.path.join(os.path.dirname(__file__), 'seaborn_rewards.png'))

    def plot_winning_rate(self, wins, draws, losses,
                          colors=['#00AB84', '#F6D258', '#F2552C']):
        assert len(wins) == len(draws) == len(losses)
        results = [(w, w+d, w+d+l) for w, d, l in zip(wins, draws, losses)]

        plt.figure(figsize=(16, 9))

        for i in reversed(range(3)):
            x = [i*100 for i in range(1, len(results)+1)]
            df = pd.DataFrame({
                "Episode": x,
                # "Episode": range(1, len(results)+1),
                "Win Rates (per 100 eps)": [r[i] for r in results]
            })
            seaborn.barplot(x='Episode', y='Win Rates (per 100 eps)', data=df, color=colors[i])

        plt.legend(
            bbox_to_anchor=(1.01, 1),
            loc='upper left',
            borderaxespad=0,
            handles=[
                mpatches.Patch(color=colors[2], label='Lose'),
                mpatches.Patch(color=colors[1], label='Draw'),
                mpatches.Patch(color=colors[0], label='Win'),
            ])

        # plt.show()
        plt.savefig(os.path.join(self.dirname, f'{int(time() * 1000)}.png'))

    @property
    def dirname(self):
        return self._dirname


def main():
    pass


if __name__ == "__main__":
    main()
