#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import pathlib
from time import time
from datetime import datetime

import plotly.graph_objects as go


class WinRateBoard:
    """
    https://plotly.com/python-api-reference/generated/plotly.graph_objects.Bar.html#plotly-graph-objs-bar
    """
    def __init__(self, dirpath=os.path.join(os.path.dirname(__file__), 'plots')):
        self._id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self._dirpath = os.path.join(dirpath, self._id)
        if not os.path.exists(self._dirpath):
            pathlib.Path(os.path.abspath(self._dirpath)).mkdir(parents=True, exist_ok=True)

    def plot(self, x, y, title=['Win', 'Draw', 'Lose'], colors=['#00AB84', '#F6D258', '#F2552C']):
        fig = go.Figure(data=[
            go.Bar(name=name, x=x, y=data, marker={"color": color}) for name, data, color in zip(title, y, colors)
        ])
        """
        fig = go.Figure(data=[
            go.Bar(name='Win', x=['100'], y=[0.2]),
            go.Bar(name='Lose', x=['100'], y=[0.6])
        ])
        """
        fig.update_xaxes(title_text='Episodes')
        fig.update_yaxes(title_text='Rate')
        fig.update_layout(barmode='stack')
        fig.write_image(os.path.join(self.dirpath, f'{int(time() * 1000)}.png'))

    @property
    def dirpath(self):
        return self._dirpath


def main():
    animals = ['giraffes', 'orangutans', 'monkeys']
    fig = go.Figure(data=[
        go.Bar(name='Win', x=['100'], y=[0.2]),
        go.Bar(name='Draw', x=['100'], y=[0.3]),
        go.Bar(name='Lose', x=['100'], y=[0.5]),
        # go.Bar(name='SF Zoo', x=animals, y=[0.1, 0.2, 0.7]),
        # go.Bar(name='LA Zoo', x=animals, y=[12, 18, 29])
    ])
    fig.update_layout(barmode='stack')
    # fig.show()
    fig.write_image(os.path.join(os.path.dirname(__file__), 'plot.png'))
    # image = fig.to_image(format='png')
    # print(type(image))



if __name__ == "__main__":
    main()
