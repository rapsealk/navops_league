#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Application.targetFrameRate = 30;
class Model(nn.Module):

    def __init__(
        self,
        inputs_size,
        hidden_size,
        outputs_size,
        batch_size=256,
        num_layers=256
    ):
        super(Model, self).__init__()

        self.n_frame = 4    # 256
        # Encoder
        self.lstm = nn.LSTM(inputs_size, hidden_size, num_layers=num_layers)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, outputs_size)

        self.hidden = (torch.randn((num_layers, batch_size, hidden_size)),
                       torch.randn((num_layers, batch_size, hidden_size)))

        """
        Supported actions
        - Engine(forward/backward)
        - Steer(left/right)
        - Turret (x6)
            - Fire
            - Offset X
            - Offset Y
        - Torpedo
        """

    def forward(self, x):
        x, self.hidden = self.lstm(x, self.hidden)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)


if __name__ == "__main__":
    model = Model(inputs_size=60, hidden_size=256, outputs_size=21)
    # output = model(np.random.uniform(-1, 1, (1, 1, 60)))
    inputs = torch.Tensor(np.random.uniform(-1, 1, (1, 256, 60)))
    output = model(inputs)
    print(output.squeeze()[-1], output.shape)
