#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class MEnet(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout_rate, n_layers, activation, output_size):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.n_mid_layers = n_layers - 2
        if activation.lower() == 'relu':
            ac = nn.ReLU
        elif activation.lower() == 'tanh':
            ac = nn.Hardtanh
        elif activation.lower() == 'leakyrelu':
            ac = nn.LeakyReLU

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            ac(),
            nn.Dropout(dropout_rate)
        )

        if self.n_mid_layers > 0:
            middle = []
            for _ in range(self.n_mid_layers):
                middle.append(nn.Linear(hidden_dim, hidden_dim))
                middle.append(nn.Dropout(dropout_rate))
                middle.append(nn.BatchNorm1d(hidden_dim))
                middle.append(ac())

            self.middle_layers = nn.Sequential(*middle)
        self.output_layer = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        if self.n_mid_layers > 0:
            x = self.middle_layers(x)
        x = self.output_layer(x)
        return x
