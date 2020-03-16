#!/usr/bin/env python

import torch

from channelnorm import ChannelNorm

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

##########################################################


net = ChannelNorm()
net = net.cuda()

for i in range(1):

	input1 = torch.rand(2, 1, 8, 8).cuda()

	input1 = input1.requires_grad_()


	output = net(input1)

	print(torch.autograd.gradcheck(net, tuple([ input1 ]), 1e-2), '<-- should be true')

# end

# print('switching to DataParallel mode')

# net = torch.nn.DataParallel(Network()).cuda()
# for i in range(1):
# 	input1 = torch.rand(2, 3, 8, 8).cuda()
# 	input2 = torch.rand(2, 3, 8, 8).cuda()

# 	input1 = input1.requires_grad_()
# 	input2 = input2.requires_grad_()

# 	output = net(input1, input2)
# 	expected = torch.mul(input1, input2)

# 	print((output.data - expected.data).abs().sum(), '<-- should be 0.0')
# 	print(torch.autograd.gradcheck(net, tuple([ input1, input2 ]), 0.001), '<-- should be true')
# # end