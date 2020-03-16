#!/usr/bin/env python

import torch

import ChannelNorm

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

##########################################################


net = ChannelNorm.ModuleChannelNorm()
net = net.cuda()

for i in range(1):

	input1 = torch.rand(2, 1, 8, 8).cuda()

	input1 = input1.requires_grad_()


	output = net(input1)

	print(torch.autograd.gradcheck(net, tuple([ input1 ]), 1e-3), '<-- should be true')

# end

print('switching to DataParallel mode')

net = torch.nn.DataParallel(ChannelNorm.ModuleChannelNorm()).cuda()
for i in range(1):
	input1 = torch.rand(2, 1, 8, 8).cuda()

	input1 = input1.requires_grad_()

	output = net(input1)

	print(torch.autograd.gradcheck(net, tuple([ input1 ]), 1e-3), '<-- should be true')
# end