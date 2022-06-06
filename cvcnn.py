import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F


class ComplexConv1d(nn.Module):
	def __init__(self, in_chan, out_chan, kernel_size, stride, padding, bias=False):
	    super(ComplexConv1d, self).__init__()		
	    self.in_chan = in_chan
	    self.conv_r_h = nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=(1, kernel_size), stride=[stride,1], padding=[0,padding], bias=bias)
	    self.conv_r_v = nn.Conv2d(in_channels=out_chan, out_channels=out_chan, kernel_size=(kernel_size, 1), stride=[1,stride], padding=[padding,0], bias=bias)
	    self.conv_i_h = nn.Conv2d(in_channels=in_chan, out_channels=out_chan, kernel_size=(1, kernel_size), stride=[stride,1], padding=[0,padding], bias=bias)
	    self.conv_i_v = nn.Conv2d(in_channels=out_chan, out_channels=out_chan, kernel_size=(kernel_size, 1), stride=[1,stride], padding=[padding,0], bias=bias)
		
	def forward(self, real_imag):
		rr = self.conv_r_h(real_imag[0])
		rr = self.conv_r_v(rr)
       
		ii = self.conv_i_h(real_imag[1])
		ii = self.conv_i_v(ii)
		
		ri = self.conv_i_h(real_imag[0])
		ri = self.conv_i_v(ri)
		
		ir = self.conv_r_h(real_imag[1])
		ir = self.conv_r_v(ir)

		real_conv = rr - ii
		imag_conv = ri + ir
	
		return [real_conv, imag_conv]

class ComplexDeconv1d(nn.Module):
	def __init__(self, in_chan, out_chan, kernel_size, stride, padding, bias=False):
	    super(ComplexDeconv1d, self).__init__()		
	    self.in_chan = in_chan
	    self.deconv_r_h = nn.ConvTranspose2d(in_channels=in_chan, out_channels=out_chan, kernel_size=(1, kernel_size), stride=[stride,1], padding=[0,padding], output_padding=[1,0], bias=bias)   # output_padding=1 当stride>1时，保证输出恰好是输入的两倍大
	    self.deconv_r_v = nn.ConvTranspose2d(in_channels=out_chan, out_channels=out_chan, kernel_size=(kernel_size, 1), stride=[1,stride], padding=[padding,0], output_padding=[0,1], bias=bias)
	    self.deconv_i_h = nn.ConvTranspose2d(in_channels=in_chan, out_channels=out_chan, kernel_size=(1, kernel_size), stride=[stride,1], padding=[0,padding], output_padding=[1,0], bias=bias)
	    self.deconv_i_v = nn.ConvTranspose2d(in_channels=out_chan, out_channels=out_chan, kernel_size=(kernel_size, 1), stride=[1,stride], padding=[padding,0], output_padding=[0,1], bias=bias)
	
	def forward(self, real_imag):
		rr_ = self.deconv_r_h(real_imag[0])       
		rr = self.deconv_r_v(rr_)
        
		ii_ = self.deconv_i_h(real_imag[1])
		ii = self.deconv_i_v(ii_)
		
		ri_ = self.deconv_i_h(real_imag[0])
		ri = self.deconv_i_v(ri_)
		
		ir_ = self.deconv_r_h(real_imag[1])
		ir = self.deconv_r_v(ir_)
	
		real_deconv = rr - ii
		imag_deconv = ri + ir
	
		return [real_deconv, imag_deconv]

class ComplexBN(nn.Module):
	def __init__(self, n_features):
		self.n_features = n_features
		super(ComplexBN, self).__init__()
		self.bn_r = nn.BatchNorm2d(n_features)
		self.bn_i = nn.BatchNorm2d(n_features)
		
	def forward(self, real_imag):
		real_bn = self.bn_r(real_imag[0].float())
		imag_bn = self.bn_i(real_imag[1].float())
		
		return [real_bn, imag_bn]

class ComplexAC(nn.Module):
# CReLU   
	def __init__(self):
		super(ComplexAC, self).__init__()
		self.relu = nn.ReLU6(inplace=True).cuda()    # 修改！
 
	def forward(self, real_imag): 
#CReLU
#		real_ac = self.relu(real_imag[0])
#		imag_ac = self.relu(real_imag[1])   

#amp&phase ReLU  （速度奇慢，出现nan）
#		real, imag = real_imag[0], real_imag[1]
#		del real_imag        
#		assert(real.size()==imag.size())    
#
#		phase = self.relu(torch.atan((imag.cpu()+0.01)/(real.cpu()+0.01)).cuda())
#
#		real_relu = torch.cos(phase) * torch.sqrt(real*real + imag*imag)
#		imag_relu = torch.sin(phase) * torch.sqrt(real*real + imag*imag)
#		del phase
        
# h-swish
		h_r = (self.relu(real_imag[0]+3)) / 6.
		h_i = (self.relu(real_imag[1]+3)) / 6.
		real_ac = (real_imag[0] * h_r) - (real_imag[1] * h_i)
		imag_ac = (real_imag[0] * h_i) + (real_imag[1] * h_r)

		return [real_ac, imag_ac]


class Residual_Block(nn.Module):
	def __init__(self, in_chan, out_chan, kernel_size, stride, padding):
		super(Residual_Block, self).__init__()
		self.in_chan = in_chan
		layers = []   # []
		layers.append(ComplexConv1d(in_chan=in_chan, out_chan=out_chan, kernel_size=kernel_size, stride=stride, padding=padding))
		layers.append(ComplexBN(n_features=out_chan))
		layers.append(ComplexAC())
		layers.append(ComplexConv1d(in_chan=in_chan, out_chan=out_chan, kernel_size=kernel_size, stride=stride, padding=padding))
		layers.append(ComplexBN(n_features=out_chan))
		self.resb = nn.Sequential(*layers)
	
	def forward(self, y):        
		x = self.resb(y)
		real_rb = x[0] + y[0]
		imag_rb = x[1] + y[1]
		
		return [real_rb, imag_rb]

class ResidualNet(nn.Module):
	def __init__(self, depth, in_chan, out_chan, kernel_size, stride, padding):
		super(ResidualNet, self).__init__()
		self.in_chan = in_chan		
		layers = []   # []
		for _ in range(depth-6):
			layers.append(Residual_Block(in_chan=in_chan, out_chan=out_chan, kernel_size=kernel_size, stride=stride, padding=padding))
			layers.append(ComplexAC())
			self.resnet = nn.Sequential(*layers)
			
	def forward(self, x_encode):               		
		x_decode = self.resnet(x_encode)
		x1 = torch.cat([x_decode[0], x_encode[0]], 1)
		x2 = torch.cat([x_decode[1], x_encode[1]], 1)
		
		return [x1, x2]


class CVCNN(nn.Module):
	def __init__(self, depth=10, in_chan=9, n_channels=32, out_chan=8): 
		super(CVCNN, self).__init__()
		self.out_chan = out_chan
		kernel_size = 3
		padding = 1 # (kernel_size - 1) // 2
		
		layers = []   #nn.ModuleList()
		layers.append(ComplexConv1d(in_chan=in_chan, out_chan=n_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=True))
		layers.append(ComplexAC())   # 0
		layers.append(ComplexConv1d(in_chan=n_channels, out_chan=n_channels, kernel_size=kernel_size, stride=1, padding=padding))
		layers.append(ComplexBN(n_features=n_channels))
		layers.append(ComplexAC())   # 1
		layers.append(ComplexConv1d(in_chan=n_channels, out_chan=n_channels, kernel_size=kernel_size, stride=2, padding=padding))
		layers.append(ComplexBN(n_features=n_channels))
		layers.append(ComplexAC())   # 2
		layers.append(ResidualNet(depth=depth, in_chan=n_channels, out_chan=n_channels, kernel_size=kernel_size, stride=1, padding=padding))   # 3_1~11_2
		layers.append(ComplexDeconv1d(in_chan=n_channels*2, out_chan=n_channels, kernel_size=kernel_size, stride=2, padding=padding))
		layers.append(ComplexBN(n_features=n_channels))
		layers.append(ComplexAC())   # 12		
		layers.append(ComplexConv1d(in_chan=n_channels, out_chan=n_channels, kernel_size=kernel_size, stride=1, padding=padding)) 
		layers.append(ComplexBN(n_features=n_channels))
		layers.append(ComplexAC())   #13
		layers.append(ComplexConv1d(in_chan=n_channels, out_chan=out_chan, kernel_size=kernel_size, stride=1, padding=padding))   # 14		
		self.cnnet = nn.Sequential(*layers)
		self._initialize_weights()
	
	def forward(self, real, imag):
		
		y = self.cnnet([real, imag])   # [bz*9*w*h, bz*9*w*h]
		out = torch.cat([y[0], y[1]], 1)   # [bz*8*w*h, bz*8*w*h]

		return out      # 5*16*w*h

	def _initialize_weights(self):
	    for layer in self.modules():	
		    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
		        init.xavier_uniform_(layer.weight)
#		        init.orthogonal_(layer.weight)
#		        layer.weight.data = layer.weight.data.half()
		        if layer.bias is not None:
		            init.constant_(layer.bias, 0)
#		            layer.bias.data = layer.bias.data.half()
		        layer = layer.cuda()
		    elif isinstance(layer, nn.BatchNorm2d):		
#		        init.constant_(layer.weight, 1)
#		        layer.weight.data = layer.weight.data.half()
#		        if layer.bias is not None:
#		            init.constant_(layer.bias, 0)
#		            layer.bias.data = layer.bias.data.half()
		        layer = layer.cuda()
#		        layer.eval().half()   # BN需要指定其输出值的类型，否则会报错RuntimeError: expected scalar type Half but found Float


class CVCNN_flops(nn.Module):
	def __init__(self, depth=10, in_chan=9, n_channels=32, out_chan=8):
		super(CVCNN_flops, self).__init__()
		self.out_chan = out_chan
		kernel_size = 3
		padding = 1  # (kernel_size - 1) // 2

		layers = []  # nn.ModuleList()
		layers.append(
			ComplexConv1d(in_chan=in_chan, out_chan=n_channels, kernel_size=kernel_size, stride=1, padding=padding,
						  bias=True))
		layers.append(ComplexAC())  # 0
		layers.append(
			ComplexConv1d(in_chan=n_channels, out_chan=n_channels, kernel_size=kernel_size, stride=1, padding=padding))
		layers.append(ComplexBN(n_features=n_channels))
		layers.append(ComplexAC())  # 1
		layers.append(
			ComplexConv1d(in_chan=n_channels, out_chan=n_channels, kernel_size=kernel_size, stride=2, padding=padding))
		layers.append(ComplexBN(n_features=n_channels))
		layers.append(ComplexAC())  # 2
		layers.append(
			ResidualNet(depth=depth, in_chan=n_channels, out_chan=n_channels, kernel_size=kernel_size, stride=1,
						padding=padding))  # 3_1~11_2
		layers.append(ComplexDeconv1d(in_chan=n_channels * 2, out_chan=n_channels, kernel_size=kernel_size, stride=2,
									  padding=padding))
		layers.append(ComplexBN(n_features=n_channels))
		layers.append(ComplexAC())  # 12
		layers.append(
			ComplexConv1d(in_chan=n_channels, out_chan=n_channels, kernel_size=kernel_size, stride=1, padding=padding))
		layers.append(ComplexBN(n_features=n_channels))
		layers.append(ComplexAC())  # 13
		layers.append(ComplexConv1d(in_chan=n_channels, out_chan=out_chan, kernel_size=kernel_size, stride=1,
									padding=padding))  # 14
		self.cnnet = nn.Sequential(*layers)
		self._initialize_weights()

	def forward(self, input):
		real = input[:,:9,:,:]
		imag = input[:,9:,:,:]
		y = self.cnnet([real, imag])  # [bz*9*w*h, bz*9*w*h]
		out = torch.cat([y[0], y[1]], 1)  # [bz*8*w*h, bz*8*w*h]

		return out  # 5*16*w*h

	def _initialize_weights(self):
		for layer in self.modules():
			if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
				init.xavier_uniform_(layer.weight)
				#		        init.orthogonal_(layer.weight)
				#		        layer.weight.data = layer.weight.data.half()
				if layer.bias is not None:
					init.constant_(layer.bias, 0)
				#		            layer.bias.data = layer.bias.data.half()
				layer = layer.cuda()
			elif isinstance(layer, nn.BatchNorm2d):
				#		        init.constant_(layer.weight, 1)
				#		        layer.weight.data = layer.weight.data.half()
				#		        if layer.bias is not None:
				#		            init.constant_(layer.bias, 0)
				#		            layer.bias.data = layer.bias.data.half()
				layer = layer.cuda()

class DnCNN_B(nn.Module):
    def __init__(self, depth=20, n_channels=64, image_channels=18, use_bnorm=True, kernel_size=3):
        super(DnCNN_B, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)