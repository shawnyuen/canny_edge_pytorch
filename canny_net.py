# Code borrowed from https://github.com/DCurro/CannyEdgePytorch
# Code borrowed from https://bitbucket.org/JianboJiao/ssus2mri/src/master/
# knowledge about canny operator and sobel operator:
# 1. https://en.wikipedia.org/wiki/Canny_edge_detector
# 2. https://en.wikipedia.org/wiki/Sobel_operator

import torch
import torch.nn as nn
import numpy as np
from scipy.signal import gaussian # go to read # scipy.signal.windows.gaussian # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.gaussian.html

class CannyNet(nn.Module):
    def __init__(self, threshold=10.0, use_cuda=True, requires_grad=False):
        super(CannyNet, self).__init__()

        self.threshold = threshold
        self.use_cuda = use_cuda

        filter_size = 5
        generated_filters = gaussian(filter_size,std=1.0).reshape([1,filter_size])
        # (1, 5)
        # 0.135335 0.606531 1 0.606531 0.135335

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))
        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]]) # https://en.wikipedia.org/wiki/Sobel_operator
        # 1 0 -1
        # 2 0 -2
        # 1 0 -1

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([[ 0, 0, 0],
                             [ 0, 1, -1],
                             [ 0, 0, 0]])

        filter_45 = np.array([[0, 0, 0],
                              [ 0, 1, 0],
                              [ 0, 0, -1]])

        filter_90 = np.array([[ 0, 0, 0],
                              [ 0, 1, 0],
                              [ 0,-1, 0]])

        filter_135 = np.array([[ 0, 0, 0],
                               [ 0, 1, 0],
                               [-1, 0, 0]])

        filter_180 = np.array([[ 0, 0, 0],
                               [-1, 1, 0],
                               [ 0, 0, 0]])

        filter_225 = np.array([[-1, 0, 0],
                               [ 0, 1, 0],
                               [ 0, 0, 0]])

        filter_270 = np.array([[ 0,-1, 0],
                               [ 0, 1, 0],
                               [ 0, 0, 0]])

        filter_315 = np.array([[ 0, 0, -1],
                               [ 0, 1, 0],
                               [ 0, 0, 0]])

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, img):
        img_r = img[:,0:1] # red channel
        img_g = img[:,1:2] # green channel
        img_b = img[:,2:3] # blue channel

        # Step1: Apply Gaussian filter to smooth the image in order to remove the noise
        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        # Step2: Find the intensity gradients of the image using Sobel operator
        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)
        # Step2: Determine edge gradient and direction
        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/3.14159))
        grad_orientation += 180.0
        grad_orientation =  torch.round( grad_orientation / 45.0 ) * 45.0

        # Step3: Non-maximum suppression, edge thinning
        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        batch = inidices_positive.size()[0]
        height = inidices_positive.size()[2]
        width = inidices_positive.size()[3]
        pixel_count = height * width * batch
        pixel_range = torch.FloatTensor([range(pixel_count)])
        if self.use_cuda:
            pixel_range = torch.cuda.FloatTensor([range(pixel_count)])

        indices = (inidices_positive.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(batch, 1, height, width)

        indices = (inidices_negative.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(batch, 1, height, width)

        channel_select_filtered = torch.stack([channel_select_filtered_positive, channel_select_filtered_negative])

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0

        thin_edges = grad_mag.clone()
        thin_edges[is_max==0] = 0.0

        # Step4: Single threshold
        thresholded = thin_edges.clone()
        thresholded[thin_edges<self.threshold] = 0.0
        thresholded = (thresholded>0.0).float()
        
        # Step5: Edge tracking by hysteresis

        return thresholded


if __name__ == '__main__':
    #CannyNet()
    import os
    from skimage import io
    img_path = os.path.join(os.getcwd(), "WechatIMG64.jpg")
    img = io.imread(img_path)/255.0 # height, width, channel
    img = np.transpose(img, [2, 1, 0]) # channel width height
    canny_operator = CannyNet(threshold=1.8, use_cuda=False, requires_grad=False)
    result = canny_operator(torch.Tensor(np.expand_dims(img, axis=0))) # batch channel width height
    res = np.squeeze(result.numpy())
    res = np.transpose(res, [1, 0])
    res = (res*255).astype(np.uint8)
    res_path = os.path.join(os.getcwd(), "WechatIMG64_CannyEdges.png")
    io.imsave(res_path, res)
