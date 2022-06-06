import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from gabor import *
from cvcnn import CVCNN
import fasta as fast


def solve(y, x_HIO, fasta_opts, prox_opts, SamplingRate, measurement_type, mask):
    
    net = CVCNN(depth=10, in_chan=9, n_channels=32, out_chan=8)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(prox_opts.logdir, 'net_%d.pth' % (prox_opts.sigma_hat))))
    model.eval()
        
    '''Fourier'''
    if measurement_type == 'Fourier': 
        def A(x):
            imsize2 = x.shape[0]   # 128
            imsize1 = imsize2*2   # 256
            pad_num = int((imsize1-imsize2)/2)   # 64
            x = np.pad(x, pad_num, "constant")   # 256*256
            Ax = np.fft.fft2(x)*(1/imsize1)*(imsize2/imsize1)   # 256*256_complex
            return Ax
        
        def At(Ax):
            imsize1 = Ax.shape[0]   # 256
            imsize2 = int(imsize1/2)   # 128
            crop_num = int((imsize1-imsize2)/2)   # 64
            Atx = np.real(np.fft.ifft2(Ax)*(imsize1)*(imsize2/imsize1)).astype('float32')   # 256*256
            Atx = Atx[crop_num:-crop_num, crop_num:-crop_num]   # 128*128
            return Atx

    '''CDP'''
    if measurement_type != 'Fourier': 
        def A(x):
            imsize = x.shape[0]   # 128            
            x = np.repeat(x[..., np.newaxis], SamplingRate, axis=2)
            m = np.reshape(mask, [imsize, imsize, SamplingRate], order='F')   # 128*128*4
            x = m*x
            Ax = np.zeros((imsize, imsize, SamplingRate)).astype('complex128')   # 128*128*4_complex
            for i in range(SamplingRate):
                Ax[...,i] = np.fft.fft2(x[...,i])*(1/imsize)   # 128*128_complex
            return Ax   # 128*128*4_complex
        
        
        def At(Ax):
            imsize = Ax.shape[0]   # 128
            Atx = np.zeros((imsize, imsize, SamplingRate)).astype('complex128')   # 128*128*4
            for i in range(SamplingRate):
                Atx[...,i] = np.fft.ifft2(Ax[...,i])   # 128*128
            mask_ = np.reshape(np.conj(mask), [imsize, imsize, SamplingRate], order='F')   # 128*128*4_complex
            Atx = np.sum(mask_*Atx, axis=2)*imsize   # 128*128_complex
            return np.real(Atx).astype('float32')   # 128*128

    def f(z):
        return 1/(2*prox_opts.sigma_w**2)*np.linalg.norm(np.abs(z)-y)**2   # 1
        
    def fgrad(z):
        return 1/prox_opts.sigma_w**2*(z-y*z/np.abs(z))   # 256*256_complex / 128*128*4_complex

    def denoi(x):
        x = x/255.   # 128*128
        x = np.expand_dims(x, 0)   # 1*128*128
        x = np.expand_dims(x, 1)   # 1*1*128*128
        
        x_real = gabor_conv(torch.from_numpy(x), tag='r', low=True).cuda()   # 1*9*128*128
        x_imag = gabor_conv(torch.from_numpy(x), tag='i', low=True).cuda()   # 1*9*128*128
        x_real, x_imag = Variable(x_real), Variable(x_imag)
        with torch.no_grad(): # this can save much memory
            dn = model(x_real, x_imag)   # 1*16*128*128
        dn_real, de_imag = dn[0,:8,...]*255., dn[0,8:,...]*255.   # 8*128*128, 8*128*128
        return dn_real.cpu().numpy(), de_imag.cpu().numpy()   # 8*128*128, 8*128*128

    def g(x0, x1):
        dn_real, de_imag = denoi(x0)   # 8*128*128, 8*128*128
        
        x1 = np.expand_dims(x1, 0)   # 1*128*128
        x1 = np.expand_dims(x1, 1)   # 1*1*128*128
        x1_real = gabor_conv(torch.from_numpy(x1), tag='r', low=False).numpy()[0,]  # 8*128*128
        x1_imag = gabor_conv(torch.from_numpy(x1), tag='i', low=False).numpy()[0,]  # 8*128*128
        
        return np.sum(prox_opts.gamma*np.linalg.norm(x1_real-dn_real, axis=(1,2))**2) + np.sum(
                prox_opts.gamma*np.linalg.norm(x1_imag-de_imag, axis=(1,2))**2)   # 1

    def prox(x0, x1_hat, tau0):
        shape = x1_hat.shape   # 128*128
        _, Gabor_r, Gabor_i = generate_gabor()
        gamma = np.array(prox_opts.gamma)
        gamma = gamma[:, np.newaxis, np.newaxis]   # (8,1,1)
        Fx1_hat = np.fft.fft2(x1_hat)   # 128*128_complex
        
        Fr = Fgabor(Gabor_r, shape=shape)   # 8*128*128_complex
        Fi = Fgabor(Gabor_i, shape=shape)   # 8*128*128_complex
        Fr_conj = np.conj(Fr)   # 8*128*128_complex		
        Fi_conj = np.conj(Fi)   # 8*128*128_complex
		
        zr, zi = denoi(x1_hat)   # 8*128*128, 8*128*128 
        Fzr = np.fft.fft2(zr)   # 8*128*128_complex
        Fzi = np.fft.fft2(zi)   # 8*128*128_complex 
        
        Fx1_num = Fx1_hat + 2*tau0*np.sum(gamma * (Fr_conj*Fzr + Fi_conj*Fzi), axis=0)   # 128*128_complex
        Fx1_den = 1 + 2*tau0*np.sum(gamma * (Fr_conj*Fr + Fi_conj*Fi), axis=0)   # 128*128_complex
        Fx1 = Fx1_num/Fx1_den    # 128*128_complex
        x1 = np.real(np.fft.ifft2(Fx1)).astype('float32')   # 128*128
        return x1

    fasta = fast.Fasta(A, At, f, fgrad, g, prox, x_HIO, fasta_opts)

    return fasta.run()

