# This file is part of f2fcss.
#
# Copyright (C) 2020 Manuel Rebol <rebol at student dot tugraz dot at>
# Patrick Knoebelreiter <knoebelreiter at icg dot tugraz dot at>
# Institute for Computer Graphics and Vision, Graz University of Technology
# https://www.tugraz.at/institute/icg/teams/team-pock/
#
# f2fcss is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free Software
# Foundation, either version 3 of the License, or any later version.
#
# f2fcss is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell5(nn.Module):  # normal conv with peephole connections
    def __init__(self, input_channels, hidden_channels, kernel_size, dilation, activation_function):
        super(ConvLSTMCell5, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.num_gates = 4  # f i g o
        self.padding = int((kernel_size - 1) / 2)
        self.convolution = nn.Conv2d(self.input_channels + self.hidden_channels, 4 * self.hidden_channels,
                                     self.kernel_size, stride=1, padding=self.padding, dilation=dilation)
        self.activation_function = activation_function
        self.peephole_weights = nn.Parameter(torch.zeros(3, self.hidden_channels), requires_grad=True)

    def forward(self, x, h, c):  # batch, channel, height, width
        x_stack_h = torch.cat((x, h), dim=1)
        A = self.convolution((x_stack_h))
        split_size = int(A.shape[1] / self.num_gates)
        (ai, af, ao, ag) = torch.split(A, split_size, dim=1)
        f = torch.sigmoid(af + c * self.peephole_weights[1, :, None, None])
        i = torch.sigmoid(ai + c * self.peephole_weights[0, :, None, None])
        g = self.activation_function(ag)
        o = torch.sigmoid(ao + c * self.peephole_weights[2, :, None, None])
        new_c = f * c + i * g
        new_h = o * self.activation_function(new_c)
        return new_h, new_c


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, activation_function, device,
                 dtype, state_init, cell_type, batch_size, time_steps, overlap, dilation=1, init='default',
                 is_stateful=True, state_img_size=None):
        super(ConvLSTM, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        if activation_function == 'tanh':
            activation_function = torch.tanh
        elif activation_function == 'lrelu':
            activation_function = F.leaky_relu
        elif activation_function == 'prelu':
            activation_function = nn.PReLU()
        self.cell_type = cell_type
        if cell_type == 5:
            self.cell = ConvLSTMCell5(self.input_channels, self.hidden_channels, self.kernel_size, self.dilation,
                                      activation_function)
        self.is_stateful = is_stateful
        self.dtype = dtype
        self.device = device
        self.state_init = state_init
        self.state_img_size = state_img_size

        self.update_parameters(batch_size, time_steps, overlap)
        self.init_states(state_img_size, state_init)

        # initialization
        if init == 'default':
            self.cell.convolution.bias.data.fill_(0)  # init all biases with 0
            nn.init.xavier_normal_(self.cell.convolution.weight.data[
                                   0 * self.cell.hidden_channels: 1 * self.cell.hidden_channels])  # sigmoid, i
            nn.init.xavier_normal_(self.cell.convolution.weight.data[
                                   1 * self.cell.hidden_channels: 2 * self.cell.hidden_channels])  # sigmoid, f
            self.cell.convolution.bias.data[1 * self.cell.hidden_channels: 2 * self.cell.hidden_channels].fill_(
                0.1)  # f bias
            nn.init.xavier_normal_(self.cell.convolution.weight.data[
                                   2 * self.cell.hidden_channels: 3 * self.cell.hidden_channels])  # sigmoid, o
            if cell_type == 5:
                nn.init.constant_(self.cell.peephole_weights, 0.1)
        if activation_function == 'tanh':
            nn.init.xavier_normal_(self.cell.convolution.weight.data[
                                   3 * self.cell.hidden_channels: 4 * self.cell.hidden_channels])  # tanh, g
        elif activation_function in ['lrelu', 'prelu']:
            nn.init.kaiming_normal_(
                self.cell.convolution.weight.data[3 * self.cell.hidden_channels: 4 * self.cell.hidden_channels],
                nonlinearity='leaky_relu')  # lrelu, g

    def forward(self, inputs, states):  # inputs shape: time_step, batch_size, channels, height, width
        new_states = None
        time_steps = inputs.shape[0]
        outputs = torch.empty(time_steps, self.batch_size, self.hidden_channels, inputs.shape[3],
                              inputs.shape[4], dtype=self.dtype, device=self.device)
        if self.is_stateful == 0 or states is None:
            h = nn.functional.interpolate(self.h0.expand(self.batch_size, -1, -1, -1),
                                          size=(inputs.shape[3], inputs.shape[4]), mode='bilinear', align_corners=True)
            c = nn.functional.interpolate(self.c0.expand(self.batch_size, -1, -1, -1),
                                          size=(inputs.shape[3], inputs.shape[4]), mode='bilinear', align_corners=True)
            print("Init LSTM")
        else:
            c = states[0]
            h = states[1]

        for time_step in range(time_steps):
            x = inputs[time_step]
            h, c = self.cell(x, h, c)  # to run hooks (pre, post) and .forward()

            if self.cell_type == 4:
                outputs[time_step] = h[:, :, 0]
            else:
                outputs[time_step] = h
            if self.is_stateful and time_step == time_steps - (self.overlap + 1):
                new_states = torch.stack((c.data, h.data))

        return outputs, new_states

    def init_states(self, state_size, state_init):
        if state_init == 'zero':
            self.h0 = nn.Parameter(torch.zeros(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype),
                                   requires_grad=False)
            self.c0 = nn.Parameter(torch.zeros(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype),
                                   requires_grad=False)
        elif state_init == 'rand':  # cell_state rand [0,1) init
            self.h0 = nn.Parameter(torch.rand(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype),
                                   requires_grad=False)
            self.c0 = nn.Parameter(torch.rand(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype),
                                   requires_grad=False)
        elif state_init == 'learn':
            if self.cell_type == 4:
                self.h0 = nn.Parameter(torch.zeros(19, 4, state_size[0], state_size[1], dtype=self.dtype),
                                       requires_grad=True)
                self.c0 = nn.Parameter(torch.zeros(19, 4, state_size[0], state_size[1], dtype=self.dtype),
                                       requires_grad=True)
            else:
                self.h0 = nn.Parameter(
                    torch.zeros(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype),
                    requires_grad=True)
                self.c0 = nn.Parameter(
                    torch.zeros(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype),
                    requires_grad=True)

    def update_parameters(self, batch_size, time_steps, overlap):
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.overlap = overlap
