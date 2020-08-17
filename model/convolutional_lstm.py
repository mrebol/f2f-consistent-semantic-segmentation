# copied from https://github.com/shahabty/Convolution_LSTM_PyTorch/blob/master/convolution_lstm.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, dilation, activation_function, bias=True):
        super(ConvLSTMCell, self).__init__()

        # assert hidden_channels % 2 == 0  # Why do we need this assert?

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels  # hidden_state_channels = cell_state_channels
        #self.bias = bias
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.num_gates = 4  # f i g o

        self.padding = int((kernel_size - 1) / 2)
        self.convolution = nn.Conv2d(self.input_channels + self.hidden_channels, 4 * self.hidden_channels,
                              self.kernel_size, stride=1, padding=self.padding,
                              dilation=dilation)  # vssnet2: input_channels=64, hidden_channels=19
        self.activation_function = activation_function

    def forward(self, x, h, c):
        # batch, channel, height, width
        x_and_h = torch.cat((x, h), dim=1)  # now we have (input+hidden) channels
        A = self.convolution((x_and_h))  # reflection padding on x and h
        split_size = int(A.shape[1] / self.num_gates) # must not be a tensor
        (ai, af, ao, ag) = torch.split(A, split_size , dim=1)  # Split A into 4 chunks of size (256/4=64) along the channel-axis # 76 / 4 = 19
        # it works: i-gate has both, an input part and a hidden part
        f = torch.sigmoid(af)
        i = torch.sigmoid(ai)  # change to F.sigmoid
        g = self.activation_function(ag)
        o = torch.sigmoid(ao)
        new_c = f * c + i * g # elementwise multiplication
        new_h = o * self.activation_function(new_c)
        return new_h, new_c

class ConvLSTMCell2(nn.Module):  # zero-padding, 2-channel conv, # Parallel ConvLSTM with weight sharing
    def __init__(self, input_channels, hidden_channels, kernel_size, dilation, activation_function, bias=True):
        super(ConvLSTMCell2, self).__init__()

        # assert hidden_channels % 2 == 0  # Why do we need this assert?

        self.input_channels = 1
        self.hidden_channels = 1  # hidden_state_channels = cell_state_channels
        #self.bias = bias
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.num_gates = 4  # f i g o

        self.padding = int((kernel_size - 1) / 2)
        self.convolution = nn.Conv2d(self.input_channels + self.hidden_channels, 4 * self.hidden_channels, self.kernel_size, stride=1, padding=self.padding,
                              dilation=dilation)  # input_channels=1, hidden_channels=1
        self.activation_function = activation_function

    def forward(self, x, h, c):
        h = h.transpose(0,1)
        c = c.transpose(0,1)
        x = x.transpose(0,1)  # bs=19
        x_and_h = torch.cat((x, h), dim=1)  # now we have (input+hidden) channels
        A = self.convolution(x_and_h)  # reflection padding on x and h
        split_size = int(A.shape[1] / self.num_gates) # must not be a tensor
        (ai, af, ao, ag) = torch.split(A, split_size , dim=1)  # Split A into 4 chunks of size (256/4=64) along the channel-axis # 76 / 4 = 19
        f = torch.sigmoid(af)
        i = torch.sigmoid(ai)
        g = self.activation_function(ag)
        o = torch.sigmoid(ao)
        new_c = f * c + i * g
        new_h = o * self.activation_function(new_c)
        return new_h.transpose(0,1), new_c.transpose(0,1)

class ConvLSTMCell3(nn.Module):  # zero-padding, 2-channel conv, Parallel ConvLSTM
    def __init__(self, input_channels, hidden_channels, kernel_size, dilation, activation_function, bias=True):
        super(ConvLSTMCell3, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.num_gates = 4  # f i g o

        self.padding = int((kernel_size - 1) / 2)
        self.convolution = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, self.kernel_size, stride=1, padding=self.padding,
                              dilation=dilation, groups=19)  # input_channels=19, hidden_channels=19
        self.activation_function = activation_function

    def forward(self, x, h, c):
        x_and_h = torch.cat((x, h), dim=2).view(x.shape[0], 19*2, x.shape[2], x.shape[3])  # merge into: x0...x18, h0...h18 --> x0, h0, x1, h1, ..., x18, h18
        A = self.convolution(x_and_h)  # self.convolution.weight[0:4,:] corresponds to x_h[0:2] and produces output A[0:4]
        ai = A[:,0::4]
        af = A[:,1::4]
        ao = A[:,2::4]
        ag = A[:,3::4]
        f = torch.sigmoid(af)
        i = torch.sigmoid(ai)
        g = self.activation_function(ag)
        o = torch.sigmoid(ao)
        new_c = f * c + i * g  # elementwise multiplication
        new_h = o * self.activation_function(new_c)
        return new_h, new_c

class ConvLSTMCell4(nn.Module):  # zero-padding, 2-channel conv
    def __init__(self, input_channels, hidden_channels, kernel_size, dilation, activation_function, bias=True):
        super(ConvLSTMCell4, self).__init__()

        # assert hidden_channels % 2 == 0  # Why do we need this assert?

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels  # hidden_state_channels = cell_state_channels
        #self.bias = bias
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.num_gates = 4  # f i g o

        self.padding = int((kernel_size - 1) / 2)
        self.convolution = nn.Conv2d(input_channels + hidden_channels, 4 * hidden_channels, self.kernel_size, stride=1, padding=self.padding,
                              dilation=dilation, groups=19)  # input_channels=19, hidden_channels=19*4
        self.activation_function = activation_function

    def forward(self, x, h, c):
        # x: batch, channel, height, width    h/c: batch, channel, time, height, width
        x_and_h = torch.empty(x.shape[0], 19*5, x.shape[2], x.shape[3], device=x.device)
        """ Proof by example
        x_and_h = torch.empty(x.shape[0], 19*5, 1,1, device=x.device) 
        self.convolution.weight.data.fill_(1)
        self.convolution.bias.data.fill_(0)
        x = torch.zeros(1,19,1,1, device=x.device)
        h = torch.zeros(1,19,4,1,1, device=x.device)
        x[:,0] = 1
        h[:,0] = 1   
        c0 = A[:,0].sum()
        co = A[:,1:].sum()
        """
        x_and_h[:, 0::5] = x
        x_and_h[:, 1::5] = h[:,:,0]  # t-1
        x_and_h[:, 2::5] = h[:,:,1]  # t-2
        x_and_h[:, 3::5] = h[:,:,2]  # t-3
        x_and_h[:, 4::5] = h[:,:,3]  # t-4
        A = self.convolution(x_and_h)
        A = A.view(A.shape[0], 19, 4, 4, A.shape[2], A.shape[3])  # bs, class!!!, gate, time,...  class dim tested! gate and time dimensions are interchangeable
        ai = A[:,:,0]
        af = A[:,:,1]
        ao = A[:,:,2]
        ag = A[:,:,3]
        # it works: i-gate has both, an input part and a hidden part
        f = torch.sigmoid(af)
        i = torch.sigmoid(ai)  # change to F.sigmoid
        g = self.activation_function(ag)
        o = torch.sigmoid(ao)
        new_c = f * c + i * g # elementwise multiplication
        new_h = o * self.activation_function(new_c)
        return new_h, new_c

class ConvLSTMCell5(nn.Module):  # normal conv with peephole connections
    def __init__(self, input_channels, hidden_channels, kernel_size, dilation, activation_function, bias=True):
        super(ConvLSTMCell5, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.num_gates = 4  # f i g o

        self.padding = int((kernel_size - 1) / 2)
        self.convolution = nn.Conv2d(self.input_channels + self.hidden_channels, 4 * self.hidden_channels,
                              self.kernel_size, stride=1, padding=self.padding,
                              dilation=dilation)  # vssnet2: input_channels=64, hidden_channels=19
        self.activation_function = activation_function
        #self.peephole_weights = nn.Parameter(torch.zeros(3, self.hidden_channels, 512//8, 1024//8 ), requires_grad=True)
        self.peephole_weights = nn.Parameter(torch.zeros(3, self.hidden_channels), requires_grad=True)  # TODO

    def forward(self, x, h, c):
        # batch, channel, height, width
        x_and_h = torch.cat((x, h), dim=1)  # now we have (input+hidden) channels
        A = self.convolution((x_and_h))  # reflection padding on x and h
        split_size = int(A.shape[1] / self.num_gates) # must not be a tensor
        (ai, af, ao, ag) = torch.split(A, split_size, dim=1)  # Split A into 4 chunks of size (256/4=64) along the channel-axis # 76 / 4 = 19
        # it works: i-gate has both, an input part and a hidden part
        f = torch.sigmoid(af + c * self.peephole_weights[1, :, None, None])
        #f = torch.sigmoid(af + c * self.peephole_weights[1]) # TODO
        i = torch.sigmoid(ai + c * self.peephole_weights[0, :, None, None])  # change to F.sigmoid
        #i = torch.sigmoid(ai + c * self.peephole_weights[0])  # change to F.sigmoid
        g = self.activation_function(ag)
        o = torch.sigmoid(ao + c * self.peephole_weights[2, :, None, None])
        #o = torch.sigmoid(ao + c * self.peephole_weights[2])
        new_c = f * c + i * g  # elementwise multiplication
        new_h = o * self.activation_function(new_c)
        return new_h, new_c


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, activation_function, device,
                 dtype, state_init, cell_type, batch_size, time_steps, overlap, dilation=1, init='default', is_stateful=True, state_img_size=None):
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
        if cell_type == 1:
            self.cell = ConvLSTMCell(self.input_channels, self.hidden_channels, self.kernel_size, self.dilation, activation_function)  # self.bias)
        elif cell_type == 2:
            self.cell = ConvLSTMCell2(self.input_channels, self.hidden_channels, self.kernel_size, self.dilation, activation_function)
        elif cell_type == 3:
            self.cell = ConvLSTMCell3(self.input_channels, self.hidden_channels, self.kernel_size, self.dilation, activation_function)
        elif cell_type == 4:
            self.cell = ConvLSTMCell4(19, 19 * 4, self.kernel_size, self.dilation, activation_function)
        elif cell_type == 5:
            self.cell = ConvLSTMCell5(self.input_channels, self.hidden_channels, self.kernel_size, self.dilation, activation_function)
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
            nn.init.xavier_normal_(self.cell.convolution.weight.data[0 * self.cell.hidden_channels: 1 * self.cell.hidden_channels])  # sigmoid, i
            nn.init.xavier_normal_(self.cell.convolution.weight.data[1 * self.cell.hidden_channels: 2 * self.cell.hidden_channels])  # sigmoid, f
            self.cell.convolution.bias.data[1 * self.cell.hidden_channels: 2 * self.cell.hidden_channels].fill_(0.1)  # f bias
            nn.init.xavier_normal_(self.cell.convolution.weight.data[2 * self.cell.hidden_channels: 3 * self.cell.hidden_channels])  # sigmoid, o
            if cell_type == 5:
                nn.init.constant_(self.cell.peephole_weights, 0.1)
        if activation_function == 'tanh':
            nn.init.xavier_normal_(self.cell.convolution.weight.data[3 * self.cell.hidden_channels: 4 * self.cell.hidden_channels])  # tanh, g
        elif activation_function in ['lrelu', 'prelu']:
            nn.init.kaiming_normal_(self.cell.convolution.weight.data[3 * self.cell.hidden_channels: 4 * self.cell.hidden_channels], nonlinearity='leaky_relu')  # lrelu, g


    def forward(self, inputs, states):  # inputs shape: time_step, batch_size, channels, height, width
        new_states = None
        time_steps = inputs.shape[0]
        outputs = torch.empty(time_steps, self.batch_size, self.hidden_channels, inputs.shape[3],
                              inputs.shape[4], dtype=self.dtype ,device=self.device)
        if self.is_stateful == 0 or states is None:
            h = nn.functional.interpolate(self.h0.expand(self.batch_size, -1, -1, -1), size=(inputs.shape[3],inputs.shape[4]), mode='bilinear', align_corners=True)
            c = nn.functional.interpolate(self.c0.expand(self.batch_size, -1, -1, -1), size=(inputs.shape[3],inputs.shape[4]), mode='bilinear', align_corners=True)
            print("Init LSTM")
        else:
            c = states[0]
            h = states[1]

        for time_step in range(time_steps):
            x = inputs[time_step]
            h, c = self.cell(x, h, c)  # to run hooks (pre, post) and .forward()

            if self.cell_type == 4:
                outputs[time_step] = h[:,:,0]
            else:
                outputs[time_step] = h
            if self.is_stateful and time_step == time_steps - (self.overlap + 1):
                new_states = torch.stack((c.data, h.data))

        return outputs, new_states

    def init_states(self, state_size, state_init):
        if state_init == 'zero':
            self.h0 = nn.Parameter(torch.zeros(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype), requires_grad=False)
            self.c0 = nn.Parameter(torch.zeros(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype), requires_grad=False)
        elif state_init == 'rand':  # cell_state rand [0,1) init
            self.h0 = nn.Parameter(torch.rand(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype), requires_grad=False)
            self.c0 = nn.Parameter(torch.rand(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype), requires_grad=False)
        elif state_init == 'learn':
            if self.cell_type == 4:
                self.h0 = nn.Parameter(torch.zeros(19, 4, state_size[0], state_size[1], dtype=self.dtype), requires_grad=True)
                self.c0 = nn.Parameter(torch.zeros(19, 4, state_size[0], state_size[1], dtype=self.dtype), requires_grad=True)
            else:
                self.h0 = nn.Parameter(torch.zeros(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype), requires_grad=True)
                self.c0 = nn.Parameter(torch.zeros(self.hidden_channels, state_size[0], state_size[1], dtype=self.dtype), requires_grad=True)

    def update_parameters(self, batch_size, time_steps,  overlap):
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.overlap = overlap
