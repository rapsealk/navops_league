#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os

import plotly.graph_objects as go


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
