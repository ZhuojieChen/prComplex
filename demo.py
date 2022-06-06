import os
import cv2
import time
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from glob import glob
from scipy.misc import imresize

import prComplex
import fasta as fast
from metric import psnr, ssim
from util import A, Poisson_noise, Gaussian_noise
import torch
import random


'''1. Setting'''

############################################
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

max_iter = 50   # default 50; for Pn_81 is 100
tolerance = 1e-7
isprint = True
imshow = False
issave = False
imsize = 128   # default 128
SamplingRate = 4   # {1,2,3,4}
savedir = 'results'
datasets = ['natural']
measurement_type = 'bipolar'   # {'Fourier', 'uniform', 'bipolar'}
noise_type = 'Gn'              # {'Pn','Gn'}
alphas = [9,27,81]             # {2,3,4} for Fourier, {9,27,81} for CDP
SNRs = [10,15,20]              # {10,15,20}
noise_levels = alphas if noise_type == 'Pn' else SNRs
############################################

############################################
parser = argparse.ArgumentParser(description="Prox Options")
parser.add_argument("--sigma_w", type=float, default=1, help='sigma of noise')
parser.add_argument("--logdir", type=str, default="nets", help='path of denoising network')
parser.add_argument("--sigma_hat", type=int, default=50, help="which denoising network for employing")
parser.add_argument('--gamma_init', default=[1,1,1,1,1,1,1,1], help='regularization coef.')
prox_opts = parser.parse_args()
############################################    


'''2. Main Body'''

for noise_level in noise_levels:
    print('\n***********************'+noise_type+str(noise_level)+'***********************')
    
    for dataset in datasets:
        print('\n-------------'+dataset+'-------------\n')
        if noise_type == 'Pn':
            coe = 1 if noise_level < 10 else 0.1
        elif noise_type == 'Gn':
            coe = 1 if noise_level > 15 else 0.1
        if measurement_type == 'Fourier':
            coe = 1 if dataset == 'natural' else 10
        prox_opts.gamma = [gamma * coe for gamma in prox_opts.gamma_init]
        
        sp_dir = './data/'+dataset+'/sharp/*.png'
        sp_file = sorted(glob(sp_dir))
        if measurement_type == 'Fourier':
            bl_dir = './data/'+dataset+'/'+measurement_type+'_R%d/'%SamplingRate +noise_type+'%d/bl/*.mat'%noise_level
            bl_file = sorted(glob(bl_dir))
            init_dirs = ['./data/'+dataset+'/'+measurement_type+'_R%d/'%SamplingRate +noise_type+'%d/init_0/*.mat'%noise_level,
                         './data/'+dataset+'/'+measurement_type+'_R%d/'%SamplingRate +noise_type+'%d/init_1/*.mat'%noise_level,
                         './data/'+dataset+'/'+measurement_type+'_R%d/'%SamplingRate +noise_type+'%d/init_2/*.mat'%noise_level,
                         './data/'+dataset+'/'+measurement_type+'_R%d/'%SamplingRate +noise_type+'%d/init_3/*.mat'%noise_level,
                         './data/'+dataset+'/'+measurement_type+'_R%d/'%SamplingRate +noise_type+'%d/init_4/*.mat'%noise_level,]
            if dataset == 'natural':
                if noise_level == 2:
                    sigma_w_list = [0.9453, 0.9531, 0.9737, 0.9456, 0.9340, 1.0022]
                elif noise_level == 3:
                    sigma_w_list = [1.3010, 1.3327, 1.3682, 1.3032, 1.2874, 1.3905]
                else:
                    sigma_w_list = [1.6096, 1.6567, 1.7138, 1.6126, 1.6107, 1.7330] 
            else:
                if noise_level == 2:   
                    sigma_w_list = [0.8337, 1.0369, 0.8739, 1.0309, 0.8147, 0.7821]
                elif noise_level == 3:
                    sigma_w_list = [1.1026, 1.4951, 1.1633, 1.4314, 1.0795, 1.0501]
                else:
                    sigma_w_list = [1.3196, 1.8997, 1.4261, 1.7842, 1.2930, 1.2801]
            mask = None
        elif measurement_type == 'uniform':
            mask = np.exp(1j*2*np.pi*np.random.rand(imsize, imsize, SamplingRate))
            init_dirs = ['nothing']
        elif measurement_type == 'bipolar':
            mask = np.random.binomial(n=1, p=0.5, size=(imsize, imsize, SamplingRate))*2-1  
            init_dirs = ['nothing']
        
        
        cost_time_list, psnr_list, ssim_list = [], [], []
        for i in range(len(sp_file)):   # num of images in some dataset  len(sp_file)
            
            psnr_sol_best, cost_time_best = 0, 0
            for j in range(len(init_dirs)):    # Fourier:5 & CDP:1
                print(i, j)
                if measurement_type == 'Fourier': 
                    init_dir = init_dirs[j]
                    init_file = sorted(glob(init_dir))
                
                # data preparation #
                filename = os.path.join(sp_file[i])
                x = np.array(cv2.imread(filename, -1), dtype=np.float32) 
                x_truth = imresize(x, [imsize, imsize], 'bilinear', mode='F')   # 128*128
        
                if measurement_type == 'Fourier':
                    bl = sio.loadmat(os.path.join(bl_file[i]))
                    y = bl['y_']   # 256*256
                    init = sio.loadmat(os.path.join(init_file[i]))
                    x_init = init['x_hat_HIO'].astype('float32')   # 128*128
                    sigma_w = sigma_w_list[i]
                else:
                    y = A(x_truth, mask, SamplingRate)  # 128*128*4 
                    if noise_type == 'Pn':
                        y, sigma_w = Poisson_noise(y, noise_level)   # 128*128*4, 1
                    elif noise_type == 'Gn':
                        y, sigma_w = Gaussian_noise(y, noise_level)   # 128*128*4, 1
                    x_init = np.ones((imsize,imsize)).astype('float32')  # 128*128
                prox_opts.sigma_w = sigma_w
                
                # iteration #
                t0 = time.time()
                fasta_opts = fast.Options(max_iter=max_iter, tolerance=tolerance, adaptive=True, stopRule=fast.Options.HYBRIDRES)
                prox_opts.sigma_hat = 50 #50
                out_1 = prComplex.solve(y, x_init, fasta_opts, prox_opts, SamplingRate, measurement_type, mask)
                sol_1 = out_1.solution
                t1 = time.time()
                fasta_opts = fast.Options(max_iter=max_iter, tolerance=tolerance, adaptive=True, stopRule=fast.Options.HYBRIDRES)
                prox_opts.sigma_hat = 40 #40
                out_2 = prComplex.solve(y, sol_1, fasta_opts, prox_opts, SamplingRate, measurement_type, mask)
                sol_2 = out_2.solution
                t2 = time.time()
                fasta_opts = fast.Options(max_iter=max_iter, tolerance=tolerance, adaptive=True, stopRule=fast.Options.HYBRIDRES)
                prox_opts.sigma_hat = 20 #20
                out_3 = prComplex.solve(y, sol_2, fasta_opts, prox_opts, SamplingRate, measurement_type, mask)
                sol_3 = out_3.solution
                t3 = time.time()
                fasta_opts = fast.Options(max_iter=max_iter, tolerance=tolerance, adaptive=True, stopRule=fast.Options.HYBRIDRES)
                prox_opts.sigma_hat = 10 #10
                out_4 = prComplex.solve(y, sol_3, fasta_opts, prox_opts, SamplingRate, measurement_type, mask)
                sol_4 = out_4.solution
                t4 = time.time()
                
                # compute metric #
                psnr_HIO, ssim_HIO = psnr(x_init, x_truth), ssim(x_init, x_truth)
                psnr_sol_1, ssim_sol_1 = psnr(sol_1, x_truth), ssim(sol_1, x_truth)
                psnr_sol_2, ssim_sol_2 = psnr(sol_2, x_truth), ssim(sol_2, x_truth)
                psnr_sol_3, ssim_sol_3 = psnr(sol_3, x_truth), ssim(sol_3, x_truth)
                psnr_sol_4, ssim_sol_4 = psnr(sol_4, x_truth), ssim(sol_4, x_truth)
                if noise_type == 'Pn':
                    if noise_level == 81:
                        psnr_sol = psnr_sol_2
                        ssim_sol = ssim_sol_2
                        sol = sol_2
                        cost_time = t2 - t0
                    elif noise_level == 27:
                        psnr_sol = psnr_sol_3
                        ssim_sol = ssim_sol_3
                        sol = sol_3
                        cost_time = t4 - t0
                    else:
                        psnr_sol = psnr_sol_4
                        ssim_sol = ssim_sol_4
                        sol = sol_4
                        cost_time = t4 - t0
                if noise_type == 'Gn':
                    if noise_level == 10:
                        psnr_sol = psnr_sol_2
                        ssim_sol = ssim_sol_2
                        sol = sol_2
                        cost_time = t2 - t0
                    elif noise_level == 15:
                        psnr_sol = psnr_sol_3
                        ssim_sol = ssim_sol_3
                        sol = sol_3
                        cost_time = t3 - t0
                    else:
                        psnr_sol = psnr_sol_4
                        ssim_sol = ssim_sol_4
                        sol = sol_4
                        cost_time = t4 - t0
                
                # print results #        
                if isprint:
                    print("sigma_w: ", sigma_w)
                    print("sol_1: ", psnr_sol_1,'/',ssim_sol_1)
                    print("sol_2: ", psnr_sol_2,'/',ssim_sol_2)
                    print("sol_3: ", psnr_sol_3,'/',ssim_sol_3)
                    print("sol_4: ", psnr_sol_4,'/',ssim_sol_4)
                    print("cost_time", cost_time)
                        
                # show image #
                if imshow:
                    plt.subplot(2,4,1)
                    plt.imshow(x_truth,'gray')
                    plt.title('Image')
                    plt.subplot(2,4,2)
                    plt.imshow(np.log(y+1e-3),'gray')
                    plt.title('measurement')
                    plt.subplot(2,4,3)
                    plt.imshow(x_init,'gray')
                    plt.title('Init_%.2f'%psnr_HIO)
                    plt.subplot(2,4,5)
                    plt.imshow(sol_1,'gray')
                    plt.title('sol_1_%.2f'%psnr_sol_1) 
                    plt.subplot(2,4,6)
                    plt.imshow(sol_2,'gray')
                    plt.title('sol_2_%.2f'%psnr_sol_2) 
                    plt.subplot(2,4,7)
                    plt.imshow(sol_3,'gray')
                    plt.title('sol_3_%.2f'%psnr_sol_3) 
                    plt.subplot(2,4,8)
                    plt.imshow(sol_4,'gray')
                    plt.title('sol_4_%.2f'%psnr_sol_4) 
                
                if psnr_sol > psnr_sol_best:
                    psnr_sol_best = psnr_sol
                    ssim_sol_best = ssim_sol
                cost_time_best += cost_time
                    
            # psnr_list.append(psnr_sol_best)
            # ssim_list.append(ssim_sol_best)
            psnr_list.append(max(psnr_sol_1,psnr_sol_2,psnr_sol_3,psnr_sol_4))
            ssim_list.append(max(ssim_sol_1,ssim_sol_2,ssim_sol_3,ssim_sol_4))
            cost_time_list.append(cost_time_best)
                
            # save image #
            if issave:
                import cv2 as cv
                (filepath, tempfilename) = os.path.split(sp_file[i])
                (shotname, extension) = os.path.splitext(tempfilename)
                savepath = './' + savedir + '/' +dataset+'/'+measurement_type+'_R%d/'%SamplingRate +noise_type+'%d/'%noise_level
                os.makedirs(savepath, exist_ok=True)
                cv.imwrite(savepath+shotname + '_%.2f'%(psnr_sol_best) + '_%.4f'%(ssim_sol_best) + '.png', sol)
                
        print('--------------------------')
        print('Avg.PSNR: ', np.mean(psnr_list))
        print('Avg.SSIM: ', np.mean(ssim_list))
        print('Avg.Time: ', np.mean(cost_time_list))

   
    

