import numpy as np

def A(x, mask, SamplingRate):
    imsize = x.shape[0]   # 128    
    x = mask*np.repeat(x[..., np.newaxis], SamplingRate, axis=2)
    Ax = np.zeros((imsize, imsize, SamplingRate)).astype('complex128')   # 128*128*4_complex
    for i in range(SamplingRate):
        Ax[...,i] = np.fft.fft2(x[...,i])*(1/imsize)   # 128*128_complex
    return Ax   # 128*128*4_complex

def Poisson_noise(Ax, alpha):
    norm = np.abs(Ax)
    w, h, r = norm.shape
    intensity_noise = alpha*norm*np.random.randn(w, h, r)
    y = norm ** 2 + intensity_noise
    y = y*(y>0)
    y = np.sqrt(y+1e-8)
    err= y - norm
    sigma_w= np.std(err)    
    return y, sigma_w

def Gaussian_noise(Ax, SNR):
    norm = np.abs(Ax)
    w, h, r = norm.shape
    noise = np.random.randn(w, h, r) 	 #产生N(0,1)噪声数据
    noise = noise-np.mean(noise) 								#均值为0
    norm_power = np.linalg.norm(norm ** 2) ** 2 / norm.size	           #此处是信号的std**2
    noise_variance = norm_power/np.power(10, (SNR/10))         #此处是噪声的std**2
    intensity_noise = (np.sqrt(noise_variance) / np.std(noise) )*noise    ##此处是噪声的std**2
    y = norm ** 2  + intensity_noise
    y = y*(y>0)
    y = np.sqrt(y+1e-8)
    err= y - norm
    sigma_w= np.std(err)
    return y, sigma_w