import torch
from thop import profile
from ptflops import get_model_complexity_info
from torchstat import stat
from cvcnn import CVCNN_flops,DnCNN_B

torch.backends.cudnn.benchmark = True
if __name__ == '__main__':
    # net = kernel_error_model(args).cuda()
    net = CVCNN_flops().cuda()
    # net = DEBLUR().cuda()
    input1 = torch.randn([1,18,128,128]).cuda()
    # input2 = torch.randn([1,9,128,128]).cuda()

    flops, params = profile(net, inputs=(input1,))
    # macs, params = get_model_complexity_info(net, (18,128,128), as_strings=True,
    #                                          print_per_layer_stat=True, verbose=True)
    # macs, params = stat(net, (18,128,128))

    print(flops)
    print(params)


