#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size=32):
        super(Encoder, self).__init__()

        self.rnn = nn.GRUCell(input_size, hidden_size)

        self._hidden_size = hidden_size

    def forward(self, x, h_in):
        x = x.view(-1, x.shape[-1])
        print(f'Encoder.forward(x={x.shape}, h_in={h_in.shape})')
        # x, h_out = self.rnn(x, h_in)
        h_out = self.rnn(x, h_in)
        print(f'Encoder.forward(x={x.shape}, h_out={h_out.shape})')
        # return x, h_out
        return h_out

    def reset_hidden_states(self):
        return torch.zeros(1, self.hidden_size)

    @property
    def hidden_size(self):
        return self._hidden_size


class Decoder(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=32):
        super(Decoder, self).__init__()

        self.attention = nn.Linear(input_size + hidden_size, hidden_size)#output_size)
        self.attention_combine = nn.Linear(input_size + hidden_size, hidden_size)
        self.rnn = nn.GRUCell(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self._hidden_size = hidden_size
        self._output_size = output_size

    def forward(self, x, h_in, encoder_outputs):
        x = x.unsqueeze(0)
        print(f'Decoder(x={x.shape}, h_in={h_in.shape}, encoder_outputs={encoder_outputs.shape})')
        attention_weights = F.softmax(
            self.attention(torch.cat([x, h_in], dim=1)),
            dim=1
        )
        print(f'Decoder.forward(attention_weights={attention_weights.shape}, x={x.shape})')
        aa = attention_weights.unsqueeze(0)
        bb = encoder_outputs.unsqueeze(0)
        print(f'Decoder(aa={aa.shape}, bb={bb.shape})')
        attention_applied = torch.bmm(attention_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat([x, attention_applied[0]], dim=1)
        output = self.attention_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, h_out = self.rnn(output, h_in)
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, h_out, attention_weights

    def reset_hidden_states(self):
        return torch.zeros(1, self.hidden_size)

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._output_size


def main():
    inputs = np.random.uniform(-1.0, 1.0, (8,))
    actions = 3
    hidden_size = 32

    encoder = Encoder(inputs.shape[0], hidden_size=hidden_size)
    # decoder = Decoder(inputs.shape[0], actions, hidden_size=hidden_size)
    decoder = Decoder(inputs.shape[0], actions, hidden_size=hidden_size)

    inputs = torch.from_numpy(inputs).float()
    encoder_h = encoder.reset_hidden_states()
    decoder_h = decoder.reset_hidden_states()

    # y, encoder_h = encoder(inputs, encoder_h)
    encoder_h = encoder(inputs, encoder_h)
    # output, decoder_h, attention = decoder(y, decoder_h)
    output, decoder_h, attention = decoder(inputs, decoder_h, encoder_h)

    print(f'attention: {attention.shape}')


if __name__ == "__main__":
    main()
