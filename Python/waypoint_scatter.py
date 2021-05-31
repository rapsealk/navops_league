#!/usr/bin/python3
# -*- coding: utf-8 -*-
import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from database import MongoDatabase

seaborn.set_theme(style='darkgrid')


def main():
    database = MongoDatabase()
    # ref = database.ref("experiences")
    result_db = database.ref("result")
    results = result_db.get()
    session_id = results[-1]["session"]

    for episode in range(1, len(result_db.get())):
        episode_id = ('0' * 10 + str(episode))[-10:]
        doc_id = session_id + '.' + episode_id
        documents = database.ref(doc_id).get()

        positions = np.array([document["position"] for document in documents])
        for n in range(positions.shape[1]):
            positions_ = positions[:, n].squeeze()
            indices_4 = list(filter(lambda x: x % 4 == 0, range(len(positions_))))
            positions_ = np.take(positions_, indices_4, axis=0)
            df = pd.DataFrame({
                "x": -positions_[:, 0],
                "y": -positions_[:, 1],
                "timestamp": [i+1 for i in range(len(positions_))]
            })
            # colors = ["flare", "crest"]
            color_palette = seaborn.color_palette("Blues", as_cmap=True)
            ax = seaborn.scatterplot(x='x', y='y', hue='timestamp', data=df, legend=False, palette=color_palette)
            ax.set(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))

        positions = np.array([document["opponent"]["position"] for document in documents])
        for n in range(positions.shape[1]):
            positions_ = positions[:, n].squeeze()
            indices_4 = list(filter(lambda x: x % 4 == 0, range(len(positions_))))
            positions_ = np.take(positions_, indices_4, axis=0)
            df = pd.DataFrame({
                "x": -positions_[:, 0],
                "y": -positions_[:, 1],
                "timestamp": [i+1 for i in range(len(positions_))]
            })
            # colors = ["flare", "crest"]
            color_palette = seaborn.color_palette("YlOrBr", as_cmap=True)
            ax = seaborn.scatterplot(x='x', y='y', hue='timestamp', data=df, legend=False, palette=color_palette)
            ax.set(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))

        # Custom legend
        custom = [Line2D([], [], marker='.', color='b', linestyle='None'),
                  Line2D([], [], marker='.', color='r', linestyle='None')]

        # fig = plt.figure(figsize=(8,5))

        plt.legend(custom, ['RL-CGF', 'RB-CGF'], loc='upper right')

        plt.savefig(f'scatter_{session_id}_{episode_id}.png')
        # plt.show()
        plt.clf()

    """
    positions = np.array([document["opponent"]["position"][0] for document in documents])
    df = pd.DataFrame({
        "x": -positions[:, 0],
        "y": -positions[:, 1],
        "timestamp": [i+1 for i in range(len(positions))]
    })
    ax = seaborn.scatterplot(x='x', y='y', hue='timestamp', data=df)
    ax = ax.set(xlim=(-1.0, 1.0), ylim=(-1.0, 1.0))
    # ax.set_xlim(-1.0, 1.0)
    # ax.set_ylim(-1.0, 1.0)
    # seaborn.pairplot(df, hue='timestamp')
    """


if __name__ == "__main__":
    main()
