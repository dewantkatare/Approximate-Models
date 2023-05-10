import torch
import torch.nn as nn
import numpy as np
from joblib import Parallel, delayed
import torch.nn.functional as F
from torchvision import datasets
import struct
import torchvision.transforms as transforms


def axcmult4x4(A,B):

    S = 0
    if (A == 0) or (B == 0):
        S = 0
    elif (A == 1):
        S = B
    elif (A % 2 == 0) & (A>1):
        if (B < 8): 
            S = 0
        else:
            S = 32 * (A/2)
    else:
        if (B < 8): 
            S = B
        else:
            S = B + 32 * (A-1)/2
    return S
    
#Approximate 8x8 array multiplier
def axc_mult8x8(a,b):

    a0b0 = axcmult4x4(int(a[4:8],2), int(b[4:8],2))
    a1b0 = axcmult4x4(int(a[0:4],2), int(b[4:8],2))
    a0b1 = axcmult4x4(int(a[4:8],2), int(b[0:4],2))
    a1b1 = axcmult4x4(int(a[0:4],2), int(b[0:4],2))
    S = (a0b0 + (a1b0 + a0b1)*16 + a1b1*256) 
    S = format(int(S), '016b')
    return S
    
#Approximate 24x24 array multiplier    
def axc_mult24x24(a,b):
    a0 = a[16:24]
    a1 = a[8:16]
    a2 = a[0:8]
    b0 = b[16:24]
    b1 = b[8:16]
    b2 = b[0:8]
    
    a0b0 = int(axc_mult8x8(a0,b0),2)
    a1b0 = int(axc_mult8x8(a1,b0),2)*256
    a2b0 = int(axc_mult8x8(a2,b0),2)*256*256
    a0b1 = int(axc_mult8x8(a0,b1),2)*256
    a1b1 = int(axc_mult8x8(a1,b1),2)*256*256
    a2b1 = int(axc_mult8x8(a2,b1),2)*256*256*256
    a0b2 = int(axc_mult8x8(a0,b2),2)*256*256
    a1b2 = int(axc_mult8x8(a1,b2),2)*256*256*256
    a2b2 = int(axc_mult8x8(a2,b2),2)*256*256*256*256
    
    S = a0b0 + a1b0 + a2b0 + a0b1 + a1b1 + a2b1 + a0b2 + a1b2 + a2b2
    S = format(S, '048b')
    #print(S)

    return S

#Convert Decimal number to Floating Point number
def dec2FP(num):
    s = ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack('!f', num))    
    return s
     

#Convert Floating Point number to Decimal number
def FP2dec(n):
    #add subnormal numbers, for NaNs, for +/- infinity
    s = struct.unpack('!f',struct.pack('!I', int(n, 2)))[0]
    return s
     

#Approximate Floating Point multiplier    
def FP_appx_mul(A,B):
    if (abs(A)<1e-36) or (abs(B)<1e-36) or (A == 0) or (B==0):
        s = 0
    else:
        S = ['0','00000000','00000000000000000000000']
        a = dec2FP(A)
        b = dec2FP(B)
        sign_ab = int(a[0])^int(b[0])
        exponent_a = a[1:9]
        if int(exponent_a,2)>255:
            exponent_a = 255
        exponent_b = b[1:9]
        if int(exponent_a,2)>255:
            exponent_b = 255
        exponent_ab = int(exponent_a,2) + int(exponent_b,2) - 127
        if exponent_ab>255:
            exponent_ab = 255
        mantissa_ab = axc_mult24x24('1'+ a[9:32],'1'+ b[9:32])
        if mantissa_ab[0] == '1':
            final_mantissa = mantissa_ab[1:24]
            exponent_ab = exponent_ab + 1
        else:
            final_mantissa = mantissa_ab[2:25]
        S = [str(sign_ab), format(exponent_ab,'08b'), final_mantissa]
        S = ''.join(S)  
        s = FP2dec(S)
    return s 


class AxC_Conv_Opr(torch.autograd.Function):

    '''source - deep learning self methods pytorch - https://discuss.pytorch.org/t/how-to-write-self-define-conv2d-in-an-efficient-way/164334'''
    @staticmethod
    def forward(ctx, input, weight, bias, padding, stride):
        #confs = torch.from_numpy(np.array([stride[0], padding[0]]))

        ctx.save_for_backward(input, weight, bias)        
        (m, n_C_prev, n_H_prev, n_W_prev) = input.shape
        (n_C, n_C_prev, f, f) = weight.shape

        n_H = ((n_H_prev - f + 2 * padding[0]) // stride[0]) + 1
        n_W = ((n_W_prev - f + 2 * padding[0]) // stride[0]) + 1

        def appx_mul(A,B):
            window = np.zeros((A.shape))
            for l in range(A.shape[0]):
              for j in range(A.shape[1]):
                for k in range(A.shape[2]):
                  window[l,j,k] = FP_appx_mul(A[l,j,k],B[l,j,k])  #A[l,j,k]*B[l,j,k]
            return np.sum(window)

        def mul_channel( weight,bias, x_pad, n_H, n_W,f):
              Z = np.zeros(( n_H, n_W ))
              for h in range(n_H):
                  for w in range(n_W):
                      vert_start = h
                      vert_end = vert_start + f
                      horiz_start = w
                      horiz_end = horiz_start + f
            
                      x_slice = x_pad[:, vert_start:vert_end, horiz_start:horiz_end]  
                      Z[ h, w] = appx_mul(x_slice,weight)  #torch.matmul(A,B)
                      Z[ h, w] += bias
              return Z
      
        input_pad = F.pad(input, (padding[0],padding[0],padding[0],padding[0]))
        weight = weight.data.numpy()
        bias = bias.data.numpy()
        input_pad = input_pad.data.numpy()

        Z = np.zeros((m, n_C, n_H, n_W ))     
         
        for i in range(m):
            Z[0] = Parallel(n_jobs=8)(delayed(mul_channel)( weight[c, :, :, :],bias[c], input_pad[0], n_H, n_W, f)  for c in  range(n_C) )
        #print("forward")    
        return torch.from_numpy(Z).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors 

        grad_input = grad_weight = grad_bias = None

        def convolutionBackward(dconv_prev, conv_in, weight, padding =1, stride=1):
            (m, n_C_prev, n_H_prev, n_W_prev) = conv_in.shape
            (n_C, n_C_prev, f, f) = weight.shape
            (m, n_C, n_H, n_W) = dconv_prev.shape

            dA_prev = torch.zeros((m, n_C_prev, n_H_prev, n_W_prev))
            dW = torch.zeros((n_C, n_C_prev, f, f))
            db = torch.zeros((n_C))
            X_pad = F.pad(conv_in, (padding,padding,padding,padding))
            dA_prev_pad = F.pad(dA_prev, (padding,padding,padding,padding))

            for i in range(m):
                x_pad = X_pad[i]
                da_prev_pad = dA_prev_pad[i]
              
                for c in range(n_C):
                    for h in range(n_H):
                        for w in range(n_W):
                            vert_start = h + h * (stride - 1)
                            vert_end = vert_start + f
                            horiz_start = w + w * (stride - 1)
                            horiz_end = horiz_start + f

                            x_slice = x_pad[:, vert_start:vert_end, horiz_start:horiz_end]

                            da_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end] += weight[c, :, :, :] * dconv_prev[i, c, h, w]
                        
                            dW[c,:,:,:] += x_slice * dconv_prev[i, c, h, w]
                            
                            db[c] += dconv_prev[i, c, h, w]  
                if padding == 0:
                  dA_prev[i, :, :, :] = da_prev_pad[:]
                else:
                  dA_prev[i, :, :, :] = da_prev_pad[:, padding:-padding, padding:-padding] 
          
            return dA_prev, dW, db
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        grad_input, grad_weight, grad_bias = convolutionBackward(grad_output, x, weight)
        grad_bias = grad_bias.squeeze()
        #print("Backward!")
        return grad_input, grad_weight, grad_bias, None,None   

class MyConv2d(nn.Module):
    def __init__(self, n_channels, out_channels, kernel_size , padding, stride, dilation=1):
        super(MyConv2d, self).__init__()

        self.kernel_size = (kernel_size, kernel_size)
        self.kernal_size_number = kernel_size * kernel_size
        self.out_channels = out_channels
        self.dilation = (dilation, dilation)
        self.padding = (padding, padding)
        self.stride = (stride, stride)
        self.n_channels = n_channels
        self.weight = nn.Parameter(torch.rand(self.out_channels, self.n_channels, self.kernel_size[0] , self.kernel_size[1] ))
        self.bias = nn.Parameter(torch.rand(self.out_channels))

    def forward(self, x):
        res = AxC_Conv_Opr.apply(x, self.weight, self.bias, self.padding, self.stride)

        return res
        
class linear_appx(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)

        input = input.data.numpy()
        weight = weight.data.numpy()
        bias = bias.data.numpy()
        def appx_mul(A,B):
          window = np.zeros((A.shape[0],B.shape[1] ))
          for k in range(A.shape[0]):
            for l in range(B.shape[1]):
              for j in range(A.shape[1]):
                  window[k,l] +=  FP_appx_mul(A[k,j],B[j,l])
          return window

        #output = input.mm(weight.t()) + bias
        output = appx_mul(input,np.transpose(weight)) + bias
        return torch.from_numpy(output).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        grad_input = grad_output.mm(weight.float())
        grad_weight = grad_output.t().mm(input.float())
        grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias


class MyLinear(nn.Module):
    def __init__(self,in_features, out_features ):
        super(MyLinear, self).__init__()
        self.fn = linear_appx.apply
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        x = self.fn(x, self.weight, self.bias)
        return x
        
batch_size = 1000
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=False)

for data, label in test_loader:
    break
data, label = data, label

class real_model(nn.Module):

    def __init__(self):
        super(real_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7 * 7 * 64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.linear1(x))
        x = self.linear2(x)
        #x = F.softmax(x, dim = 1)
        return x
    
model_exact = real_model()
filename = "trained_lenet5.pt" #nu-det-model.pt

model_exact.load_state_dict(torch.load(filename, map_location=torch.device('cpu') ))
model_exact.eval()

class appx_model(nn.Module):
    def __init__(self):
        super(appx_model, self).__init__()
        self.conv1 = MyConv2d(1, 32, 3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = MyConv2d(32, 64,3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = MyLinear(7 * 7 * 64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = MyLinear(200, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(self.relu1(x))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.linear1(x))
        x = self.linear2(x)
        #x = F.softmax(x, dim=1)
        return x
    
model_appx = appx_model()
model_appx.load_state_dict(torch.load(filename, map_location=torch.device('cpu') ))
model_appx.eval() 

i =  0
x = data[i].unsqueeze(0)
y = label[i].unsqueeze(0)

scores_exact = model_exact(x)
pred_exact = model_exact(x).data.max(1, keepdim=True)[1][0].item() 
print("Scores exact:", scores_exact)
print("Prediction exact:", pred_exact)

scores_appx = model_appx(x)
pred_appx = model_appx(x).data.max(1, keepdim=True)[1][0].item() 
print("Scores approximate:", scores_appx)
print("Prediction approximate:", pred_appx)