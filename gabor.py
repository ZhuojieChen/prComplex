import numpy as np
import torch
import torch.nn.functional as F

# 用于快速傅里叶变换之前
def for_fft(ker, shape):
#    print(np.shape(ker))   # (23,23)
    ker_mat = np.zeros(shape, dtype=np.float32)
#    print(np.shape(ker_mat))   # (256,256)
    ker_shape = np.asarray(np.shape(ker))
    circ = np.ndarray.astype(-np.floor((ker_shape) / 2), dtype=np.int)
    ker_mat[:ker_shape[0], :ker_shape[1]] = ker
    ker_mat = np.roll(ker_mat, circ, axis=(0, 1))
    return ker_mat

# f Convolution r for each f & r
def cconv_torch(x, ker):
    with torch.no_grad():
        x_h, x_v = x.size()   # 128, 128
        conv_ker = np.flip(np.flip(ker, 0), 1)   # 对角线翻转
        ker = torch.FloatTensor(conv_ker.copy())
        k_h, k_v = ker.size()   # 3, 3
        k2_h = k_h // 2   # 1
        k2_v = k_v // 2   # 1
		# padding(128*128~130*130)
        x = torch.cat((x[-k2_h:,:], x, x[0:k2_h,:]), dim = 0)   # 130*128
        x = torch.cat((x[:,-k2_v:], x, x[:,0:k2_v]), dim = 1)   # 130*130
        x = x.unsqueeze(0)   # 1*130*130
        x = x.unsqueeze(1)   # 1*1*130*130
        ker = ker.unsqueeze(0)   # 1*3*3
        ker = ker.unsqueeze(1)   # 1*1*3*3
        y1 = F.conv2d(x, ker)   # 1*1*128*128
        y1 = torch.squeeze(y1)   # 128*128
        y = y1[-x_h:, -x_v:]
    return y


def normalize(kernel):
        ppos = np.where(kernel > 0)
        pos = kernel[ppos].sum()

        pneg = np.where(kernel < 0)
        neg = kernel[pneg].sum()

        meansum = (pos - neg) / 2
        if meansum > 0:
            pos = pos / meansum
            neg = -neg / meansum

        kernel[pneg] = pos * kernel[pneg]
        kernel[ppos] = neg * kernel[ppos]

        return kernel


def generate_gabor(low=False):
        gabor_c = []
        gabor_r = []
        gabor_i = []
        
        y = np.array([[[0.235702260395516 + 0.00000000000000j,	0.471404520791032 + 0.00000000000000j,	0.235702260395516 + 0.00000000000000j]],
                      [[-0.176776695296637 + 0.204124145231932j,	0.353553390593274 + 0.00000000000000j,	-0.176776695296637 - 0.204124145231932j]],
					  [[-0.176776695296637 - 0.204124145231931j,	0.353553390593274 + 0.00000000000000j,	-0.176776695296637 + 0.204124145231931j]]])
		
        for i in range(3):
	        for j in range(3):
	            raw = y[i,:,:]
	            column = y[j,:,:].T
	            gabor = np.dot(column, raw)
	            gabor_c.append(gabor)
	            gabor_r.append(np.real(gabor))
	            gabor_i.append(np.imag(gabor))	
		 # 控制滤波器数量	
        if low == False:
            del gabor_c[0]
            del gabor_r[0]
            del gabor_i[0]  			 
		
        return gabor_c, gabor_r, gabor_i


# 用于算高通滤波器的L1范数   能量
def gabor_norm(gabor, dtype=np.float32):
    chan = len(gabor)
    WvNorm = np.zeros(chan,dtype=dtype)   # 8
    j = 0
    for d in range(len(gabor)):
        norm = np.sum(np.abs(gabor[d]))   # 每个矩阵元素的绝对值之和
        WvNorm[j] = norm
        j += 1
    return WvNorm

 
# f(gabor) Convolution r   1张被8个卷
def gabor_conv(x, gabor = None, weights = None, tag = 'r', low = False):
    img_num = x.size()[0]   # 1
    img_shape = x.size()[2:]   # 128*128
    if gabor is None:
        if tag == 'r':
            _, gabor, _ = generate_gabor(low=low)   # 8*3*3
        elif tag == 'i':
            _, _, gabor = generate_gabor(low=low)   # 8*3*3			

    w = torch.ones(len(gabor))   # 8 权重向量 默认为1
    if weights is not None:
        if isinstance(weights,list) == True:
            w = torch.FloatTensor(weights)
            w = w.view(len(gabor), 1, 1, 1)
        else:
            norm = torch.from_numpy(gabor_norm(gabor))
            w = torch.ones(len(gabor)) * weights / norm    # norm用于对weights进行归一化，norm越大，weights越小
            w = w.view(len(gabor), 1, 1, 1)

    chan = len(gabor)   # 8
    z = torch.zeros((img_num, chan, *img_shape))   # 1*8*128*128
    for i in range(img_num):   # 0
        x_s = x[i,]   # 1*256*256
        j = 0
        for idx in range(len(gabor)):   # 0:7  
            z[i,j,] = cconv_torch(x_s[0,], gabor[idx]) * w[j]
            j += 1
    return z

# 对gabor做FT
def Fgabor(gabor, shape):
    chan_num = len(gabor)   # 8
    W = np.zeros((chan_num, *shape), dtype = np.float32)   # 8*128*128
    i = 0
    for d in range(len(gabor)):
        W[i,] = for_fft(gabor[d], shape=shape)
        i += 1

    F = np.zeros((chan_num, *shape)).astype('complex64')   # 8*128*128
    for i in range(chan_num):   # 0:7
        F[i,] = np.fft.fft2(W[i,])
    return F
