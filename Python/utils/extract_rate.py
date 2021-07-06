#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import shutil
import time

import seaborn
import matplotlib
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image

seaborn.set_theme(style="darkgrid")

GREEN = [21, 150, 120, 255]
RED = [241, 241, 246, 255]


def extract_plot_rate(path):
    image = Image.open(path)
    imgnp = np.array(image)[140:800, 210:-170]
    for i in range(10):
        g = 0
        for j in range(imgnp.shape[0]):
            g += np.all(imgnp[j, 124*i+30] == GREEN)
        yield min(100, max(0, int(np.round(g / imgnp.shape[0] * 100))))


def move_files(path, per=10, filters=None):
    # files = os.listdir(path)
    for filter_ in filters:
        path_ = os.path.join(path, filter_)
        dst_path = os.path.join(path, f'p{filter_}')
        try:
            os.mkdir(dst_path)
        except FileExistsError:
            pass
        i, reports = 18, os.listdir(path_)
        while i < len(reports):
            src = os.path.join(path_, reports[i])
            dst = os.path.join(dst_path, reports[i])
            shutil.copyfile(src, dst)
            i += 20


def plot_winning_rate(wins, colors=['#00AB84']):
    fig = plt.figure(figsize=(16, 9))
    colors = ['#000000']

    x = [i for i in range(1, len(wins)+1)]
    # x = [i*100 for i in range(1, len(wins)+1)]
    df = pd.DataFrame({
        "Episode": x,
        # "Episode": range(1, len(results)+1),
        "Win Rates (per 100 eps)": wins
    })
    print(wins, len(wins))
    # seaborn.barplot(x='Episode', y='Win Rates (per 100 eps)', data=df, color=colors[0])
    fig = seaborn.barplot(x='Episode', y='Win Rates (per 100 eps)', data=df, color=colors[0])

    xticks = ['' for _ in range(len(x))]
    for i in range(len(xticks) // 10):
        try:
            xticks[(i+1)*10] = str((i+1)*1000)
        except IndexError:
            pass
    xticks[-1] = str(len(xticks)*100)
    fig.set(xticklabels=xticks)
    plt.ylim(0, 100)
    font = {'family' : 'normal',
            # 'weight' : 'bold',
            'size'   : 48}
    matplotlib.rc('font', **font)
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{int(time.time() * 1000)}.png'))


"""
def plot_winning_rate(wins, draws, losses,
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

    # plt.legend(
    #     bbox_to_anchor=(1.01, 1),
    #     loc='upper left',
    #     borderaxespad=0,
    #     handles=[
    #         mpatches.Patch(color=colors[2], label='Lose'),
    #         mpatches.Patch(color=colors[1], label='Draw'),
    #         mpatches.Patch(color=colors[0], label='Win'),
    #     ])

    plt.savefig(os.path.join(os.path.join(os.path.dirname(__file__), f'{int(time() * 1000)}.png')))
"""


def main():
    path = os.path.join(os.path.dirname(__file__), 'reportings')
    # filters = ['2021-06-23_12-25-51', '2021-06-23_15-26-49']
    # filters = ['2021-06-23_15-26-49']
    filters = ['2021-07-04_21-54-45']
    move_files(path, filters=filters)

    for filter_ in filters:
        rates = []
        with open(os.path.join(path, f'{filter_}.csv'), 'w') as f:
            for file_ in os.listdir(os.path.join(path, f'p{filter_}')):
                rates.extend(tuple(map(str, extract_plot_rate(os.path.join(path, f'p{filter_}', file_)))))
            f.write(','.join(rates))
            rates = list(map(int, rates))
            # plot_winning_rate(rates, [0 for i in range(len(rates))], [0 for i in range(len(rates))])
            #plot_winning_rate(rates[:-1])
            plot_winning_rate(rates)
            # rates = [0, 0, 0, 0] + rates
            # rates = [int(np.mean(rates[i-5:i])) for i in range(5, len(rates))]
            # rates = [int(np.max(rates[i-5:i])) for i in range(5, len(rates))]
            # x = 10
            # rates = [int(np.mean(rates[i:i+x])) for i in range(len(rates)-x)]
            rates_ = [np.round(np.mean(rates[:i+1])) for i in range(len(rates))]
            plot_winning_rate(rates_)
            # rates = [int(np.min(rates[i-5:i])) for i in range(5, len(rates))]
            rates_ = [int(np.min(rates[i:i+5])) for i in range(len(rates))]
            plot_winning_rate(rates_)


if __name__ == "__main__":
    main()
    """
    wins = [
        21, 17, 14, 23, 14, 6, 10, 10, 37, 29,
        40, 22, 37, 37, 40, 48, 40, 34, 46, 43,
        39, 50, 41, 17, 29, 31, 34, 42, 51, 63,
        64, 46, 55, 77, 68, 58, 60, 58, 67, 71,
        69, 69
    ]
    plot_winning_rate(wins)
    """
