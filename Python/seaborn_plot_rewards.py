#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import time

import seaborn
import matplotlib.pyplot as plt
import pandas as pd

seaborn.set_theme(style='darkgrid')


def main():
    # path = os.path.join(os.path.dirname(__file__), 'run-2021-04-02_18-04-32-NavOpsMultiDiscrete-v1-tag-logging_hitpoint.csv')
    dpath = os.path.join(os.path.dirname(__file__), 'run-2021-04-02_18-04-32-NavOpsMultiDiscrete-v1-tag-logging_hitpoint.csv')
    fpath = os.path.join(os.path.dirname(__file__), 'run-2021-04-02_18-04-32-NavOpsMultiDiscrete-v1-tag-logging_fuel_usage.csv')
    df1 = pd.read_csv(dpath)
    df2 = pd.read_csv(fpath)

    df = pd.DataFrame({
        "Durability": list(df1.Value),
        "Fuel consumption": list(df2.Value),
        "Episode": list(df1.Step)
    })
    averages = list(df.Durability)
    for i, hp in enumerate(list(df.Durability)[1:]):
        averages[i+1] = averages[i+1] * 0.1 + averages[i] * 0.9
    df["Durability"] = averages
    # df["Fuel comsumption"] = list(df.Value)
    # df["Episode"] = list(df.Step)
    seaborn.lineplot(x='Episode',
                     y='Durability',
                     style='Fuel consumption',
                     color='#FF6F00',
                     ci=95,
                     data=df)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'seaborn_hitpoint.png'))

    """
    averages = list(df.Value)
    for i, reward in enumerate(list(df.Value)[1:]):
        averages[i+1] = averages[i+1] * 0.1 + averages[i] * 0.9

    df['Reward'] = averages

    seaborn.lineplot(x='Episode', y='Reward', color='#FF6F00',
                     ci=95,
                     #hue='AverageValue',
                     #style='event',
                     data=df)
    plt.savefig(os.path.join(os.path.dirname(__file__), 'seaborn_rewards.png'))
    """


if __name__ == "__main__":
    main()
