#!/usr/bin/env python

import torch

import cupy
import re


kernel_ChannelNorm_updateOutput = '''
    extern "C" __global__ void kernel_ChannelNorm_updateOutput(
        const int n,
        const float* input1,

        const long input1_size_b,
        const long input1_size_c,
        const long input1_size_h,
        const long input1_size_w,

        const long input1_stride_b,
        const long input1_stride_c,
        const long input1_stride_h,
        const long input1_stride_w,

        float* output,

        const long output_size_b,
        const long output_size_c,
        const long output_size_h,
        const long output_size_w,

        const long output_stride_b,
        const long output_stride_c,
        const long output_stride_h,
        const long output_stride_w) {

        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index >= n) {
            return;
        }

        int dim_b = output_size_b;
        int dim_c = output_size_c;
        int dim_h = output_size_h;
        int dim_w = output_size_w;

        int dim_chw = dim_c * dim_h * dim_w;

        int b = ( index / dim_chw ) % dim_b;
        int y = ( index / dim_w )   % dim_h;
        int x = ( index          )  % dim_w;

        int i1dim_c = input1_size_c;
        int i1dim_h = input1_size_h;
        int i1dim_w = input1_size_w;
        int i1dim_chw = i1dim_c * i1dim_h * i1dim_w;
        int i1dim_hw  = i1dim_h * i1dim_w;

        float result = 0.0;

        for (int c = 0; c < i1dim_c; ++c) {
            int i1Index = b * i1dim_chw + c * i1dim_hw + y * i1dim_w + x;
            float val = input1[i1Index];
            result += static_cast<float>(val * val);
        }
        result = sqrt(result);
        output[index] = static_cast<float>(result);
    }
'''


kernel_ChannelNorm_backward_input1 = '''
    extern "C" __global__ void kernel_ChannelNorm_backward_input1(
        const int n,
        const float* input1,

        const long input1_size_b,
        const long input1_size_c,
        const long input1_size_h,
        const long input1_size_w,

        const long input1_stride_b,
        const long input1_stride_c,
        const long input1_stride_h,
        const long input1_stride_w,

        const float* output,

        const long output_size_b,
        const long output_size_c,
        const long output_size_h,
        const long output_size_w,

        const long output_stride_b,
        const long output_stride_c,
        const long output_stride_h,
        const long output_stride_w,

        const float* gradOutput,

        const long gradOutput_size_b,
        const long gradOutput_size_c,
        const long gradOutput_size_h,
        const long gradOutput_size_w,

        const long gradOutput_stride_b,
        const long gradOutput_stride_c,
        const long gradOutput_stride_h,
        const long gradOutput_stride_w,

        float* gradInput,

        const long gradInput_size_b,
        const long gradInput_size_c,
        const long gradInput_size_h,
        const long gradInput_size_w,

        const long gradInput_stride_b,
        const long gradInput_stride_c,
        const long gradInput_stride_h,
        const long gradInput_stride_w) {

        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index >= n) {
            return;
        }

        float val = 0.0;

        int dim_b = gradInput_size_b;
        int dim_c = gradInput_size_c;
        int dim_h = gradInput_size_h;
        int dim_w = gradInput_size_w;
        int dim_chw = dim_c * dim_h * dim_w;
        int dim_hw  = dim_h * dim_w;

        int b = ( index / dim_chw ) % dim_b;
        int y = ( index / dim_w )   % dim_h;
        int x = ( index          )  % dim_w;


        int outIndex = b * dim_hw + y * dim_w + x;
        val = static_cast<float>(gradOutput[outIndex]) * static_cast<float>(input1[index]) / (static_cast<float>(output[outIndex])+1e-9);
        gradInput[index] = static_cast<float>(val);
    }
'''


def cupy_kernel(strFunction, objectVariables):
    
    strKernel = globals()[strFunction]

    # replce the C code with real numbers
    # SIZE_0 Batch
    # SIZE_1 Channel
    # SIZE_2 H
    # SIZE_3 W
    # SIZE_x(vector), get the size of vector of dim-x
    while True:
        objectMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

        if objectMatch is None:
            break
        # end

        intArg = int(objectMatch.group(2))

        strTensor = objectMatch.group(4)
        intSizes = objectVariables[strTensor].size()

        strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))
    # end

    # get the stride
    # VALUE_0 Batch_Stride
    # VALUE_1 Channel_Stride
    # VALUE_2 H_Stride
    # VALUE_3 W_Stride
    # VALUE_x( vector, b, c, h, w) get the value  of vector withshape BCHW
    # it return vector[ b * B_S + c * C_S + h * H_S + w * W_S ]

    while True:

        objectMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objectMatch is None:
            break
        # end

        intArgs = int(objectMatch.group(2))
        strArgs = objectMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()

        # strIndex = [B*C*H*W, C*H*W, H*W, 1]
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip(
        ) + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objectMatch.group(
            0), strTensor + '[' + str.join('+', strIndex) + ']')
    # end

    return strKernel
# end


@cupy.util.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)
# end

@cupy.util.memoize(for_each_device=True)
def cunnex(strFunction):
	return cupy.cuda.compile_with_cache(globals()[strFunction]).get_function(strFunction)
# end


class _ChannelNorm(torch.autograd.Function):
    @staticmethod
    def forward(self, input1):
        B = input1.shape[0]  # batch
        C = input1.shape[1]  # channel
        H = input1.shape[2]  # height
        W = input1.shape[3]  # width
        # stride()
        assert(input1.is_contiguous() == True)
        input1_size_b, input1_size_c, input1_size_h, input1_size_w = B, C, H, W

        output_size_b, output_size_c, output_size_h, output_size_w = B, 1, H, W

        input1_stride_b, input1_stride_c, input1_stride_h, input1_stride_w = \
             C * H * W, H * W, W, 1

        C_out = 1
        output_stride_b, output_stride_c, output_stride_h, output_stride_w = \
             C_out * H * W, H * W, W, 1

        output = input1.new_zeros([B, C_out, H, W])

        if input1.is_cuda == True:
            n = output.nelement()
            cunnex('kernel_ChannelNorm_updateOutput')(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n, input1.data_ptr(), input1_size_b, input1_size_c, input1_size_h, input1_size_w,
                        input1_stride_b, input1_stride_c, input1_stride_h,  input1_stride_w,
                        output.data_ptr(), output_size_b, output_size_c, output_size_h, output_size_w,
                        output_stride_b, output_stride_c, output_stride_h, output_stride_w,
                        ])
        elif input1.is_cuda == False:
            raise NotImplementedError()
        # end

        self.save_for_backward(input1, output)

        return output
        # end

    @staticmethod
    def backward(self, gradOutput):

        input1, output = self.saved_tensors


        assert(gradOutput.is_contiguous() == True)

        B = input1.shape[0]
        C = input1.shape[1]
        H = input1.shape[2]
        W = input1.shape[3]

        assert(gradOutput.is_contiguous() == True)

        gradInput1 = input1.new_zeros(
            [B, C, H, W]) if self.needs_input_grad[0] == True else None

        input1_size_b = output_size_b = gradOutput_size_b = gradInput_size_b = B
        input1_size_c = output_size_c = gradOutput_size_c = gradInput_size_c = C
        input1_size_h = output_size_h = gradOutput_size_h = gradInput_size_h = H
        input1_size_w = output_size_w = gradOutput_size_w = gradInput_size_w = W

        input1_stride_b = output_stride_b = gradOutput_stride_b = gradInput_stride_b = C * H * W
        input1_stride_c = output_stride_c = gradOutput_stride_c = gradInput_stride_c = H * W
        input1_stride_h = output_stride_h = gradOutput_stride_h = gradInput_stride_h = W
        input1_stride_w = output_stride_w = gradOutput_stride_w = gradInput_stride_w = 1

        if input1.is_cuda == True:
            if gradInput1 is not None:
                n = output.nelement()
                cunnex('kernel_ChannelNorm_backward_input1')(
                    grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                    block=tuple([512, 1, 1]),
                    args=[n, input1.data_ptr(), input1_size_b, input1_size_c, input1_size_h, input1_size_w,
                        input1_stride_b, input1_stride_c, input1_stride_h, input1_stride_w,
                        output.data_ptr(), output_size_b, output_size_c, output_size_h, output_size_w,
                        output_stride_b, output_stride_c, output_stride_h, output_stride_w,
                        gradOutput.data_ptr(), gradOutput_size_b, gradOutput_size_c, gradOutput_size_h, gradOutput_size_w,
                        gradOutput_stride_b, gradOutput_stride_c, gradOutput_stride_h, gradOutput_stride_w,
                        gradInput1.data_ptr(), gradInput_size_b, gradInput_size_c, gradInput_size_h, gradInput_size_w,
                        gradInput_stride_b, gradInput_stride_c, gradInput_stride_h, gradInput_stride_w
                        ])
            elif input1.is_cuda == False:
                raise NotImplementedError()
        # end

        return gradInput1
	# end
# end


def FunctionChannelNorm(input1):
    return _ChannelNorm.apply(input1)
# end

class ModuleChannelNorm(torch.nn.Module):
    def __init__(self):
        super(ModuleChannelNorm, self).__init__()
    # end

    def forward(self, input1):
        return _ChannelNorm.apply(input1)
    # end
# end
