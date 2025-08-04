# resnet1d.py
"""
Shenda Hong, Oct 2019, with GRU and with Attention
"""

import numpy as np
from collections import Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
#from sklearn.metrics import classification_report 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))

    def __len__(self):
        return len(self.data)
    
class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net
    
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out += identity

        return out
    
class ResNet1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Parameters:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        
    """

    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, 
                 downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, verbose=False,
                 with_attention=False, n_attention_heads=8, with_gru=False, n_gru_layers=1, gru_hidden_size=None):
        super(ResNet1D, self).__init__()
        
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.with_attention = with_attention
        self.with_gru = with_gru

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # Determine the input dimension for GRU/Attention and the final dense layer
        # This is the 'out_channels' of the last residual block
        feature_sequence_dim = out_channels 
        
        # Optional GRU layers
        self.gru_output_dim = feature_sequence_dim # Default, if gru_hidden_size not provided
        if self.with_gru:
            # If gru_hidden_size is not specified, use the feature_sequence_dim
            self.gru_output_dim = gru_hidden_size if gru_hidden_size is not None else feature_sequence_dim
            self.gru = nn.GRU(
                input_size=feature_sequence_dim,
                hidden_size=self.gru_output_dim,
                num_layers=n_gru_layers,
                batch_first=False, # GRU expects (seq_len, batch, features) by default
                bidirectional=False # Unidirectional by default
            )

        # Multi-head Attention will operate on GRU output if with_gru is True, otherwise on ResNet features
        attention_input_dim = self.gru_output_dim if self.with_gru else feature_sequence_dim

        # Multi-head Attention
        if self.with_attention:
            self.attention = nn.MultiheadAttention(embed_dim=attention_input_dim, num_heads=n_attention_heads, batch_first=False)
            self.attention_norm = nn.LayerNorm(attention_input_dim) # Add a layer norm for stability after attention

        # final prediction: input to dense layer will be the output of GRU/Attention or ResNet features after pooling
        final_dense_input_dim = self.gru_output_dim if self.with_gru else feature_sequence_dim
        
        self.final_bn = nn.BatchNorm1d(feature_sequence_dim) # BN before GRU/Attention operates on ResNet output
        self.final_relu = nn.ReLU(inplace=True)
        self.dense = nn.Linear(final_dense_input_dim, n_classes)
        
    def forward(self, x):
        
        out = x
        
        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # Apply final BN and ReLU for CNN features before RNN/Attention
        if self.use_bn:
            out = self.final_bn(out) # This BN applies to the output of the last conv block
        out = self.final_relu(out)

        # Permute for GRU/Attention: (batch_size, channels, sequence_length) -> (sequence_length, batch_size, channels)
        out = out.permute(2, 0, 1) # (L, N, C) where L is sequence_length, N is batch_size, C is channels (embedding_dim)
        if self.verbose:
            print('before GRU/Attention permute shape', out.shape)

        # Optional GRU layer
        if self.with_gru:
            if self.verbose:
                print('before GRU input shape', out.shape)
            # GRU outputs (output, h_n) where output is (L, N, H_out)
            out, _ = self.gru(out)
            if self.verbose:
                print('after GRU output shape', out.shape)

        # Optional Multi-head Attention
        # Attention should now operate on the GRU output (if with_gru) or the ResNet features
        if self.with_attention:
            if self.verbose:
                print('before attention input shape', out.shape)
            # MultiheadAttention expects (query, key, value)
            attn_output, _ = self.attention(out, out, out)
            # Add residual connection and layer norm for attention
            out = self.attention_norm(attn_output + out) 
            if self.verbose:
                print('after attention output shape', out.shape)

        # Permute back to (batch_size, channels, sequence_length) for Global Average Pooling
        out = out.permute(1, 2, 0) # (N, C, L)
        if self.verbose:
            print('after GRU/Attention permute back shape for pooling', out.shape)

        out = out.mean(-1) # Global Average Pooling over the sequence length dimension
        if self.verbose:
            print('final pooling shape', out.shape)
        
        out = self.dense(out)
        if self.verbose:
            print('dense output shape', out.shape)
        
        return out