from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
import math


class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        gwc_feature = torch.cat((l2, l3, l4), dim=1)

        if not self.concat_feature:
            return {"gwc_feature": gwc_feature}
        else:
            concat_feature = self.lastconv(gwc_feature)
            return {"gwc_feature": gwc_feature, "concat_feature": concat_feature}


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6


class Cross_Fusion(nn.Module):
    def __init__(self, mode):
        super(Cross_Fusion, self).__init__()
        self.mode = mode
        if self.mode > 0:
            self.query_conv = unet_upconv(64, 256)
        elif self.mode < 0:
            self.query_conv = unet_conv(16, 256)
        else:
            self.query_conv = nn.Sequential(nn.Conv2d(32, 256, kernel_size=3, stride=1, padding=1),
                                            nn.ReLU(True),
                                            nn.BatchNorm2d(256))

        self.key_conv = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.value_conv = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.fusion_conv = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

    def forward(self, stereo_cost, audio_defea):
        #import pdb; pdb.set_trace()
        audio_query = self.query_conv(audio_defea)
        audio_query = audio_query.reshape(stereo_cost.shape)

        stereo_key = self.key_conv(stereo_cost)
        stereo_value = self.value_conv(stereo_cost)

        fusion_fea = audio_query * stereo_key
        fusion_res = self.fusion_conv(fusion_fea)
        stereo_cost = stereo_value * fusion_res

        return stereo_cost

class GwcNet(nn.Module):
    def __init__(self, maxdisp, use_concat_volume=False):
        super(GwcNet, self).__init__()
        self.maxdisp = maxdisp
        self.use_concat_volume = use_concat_volume

        self.num_groups = 40

        if self.use_concat_volume:
            self.concat_channels = 12
            self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)
        else:
            self.concat_channels = 0
            self.feature_extraction = feature_extraction(concat_feature=False)

        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        
        # cross fusion of audio and stereo depth feature
        self.cross_fusion1 = Cross_Fusion(1)
        self.cross_fusion2 = Cross_Fusion(0)
        self.cross_fusion3 = Cross_Fusion(-1)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    
    # gwcnet forward with audio feature 
    def forward(self, left, right, audio_fea):
        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)

        gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4,
                                      self.num_groups)
        if self.use_concat_volume:
            concat_volume = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],
                                                self.maxdisp // 4)
            volume = torch.cat((gwc_volume, concat_volume), 1)
        else:
            volume = gwc_volume



        cost0 = self.dres0(volume) 
        cost0 = self.dres1(cost0) + cost0

        
        #***************************************************
        #     cross fusion of audio and stereo depth feature
        #****************************************************
        out1 = self.dres2(cost0) 
        out1 = self.cross_fusion1(out1, audio_fea[0])
        out2 = self.dres3(out1)  
        out2 = self.cross_fusion2(out2, audio_fea[1])
        out3 = self.dres4(out2)  
        out3 = self.cross_fusion3(out3, audio_fea[2])
        

        

        if self.training:
            cost0 = self.classif0(cost0) 
            cost1 = self.classif1(out1) 
            cost2 = self.classif2(out2) 
            cost3 = self.classif3(out3) 

            cost0 = F.upsample(cost0, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)

            cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)

            cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            return [pred0, pred1, pred2, pred3]

        else:
            cost3 = self.classif3(out3)
            cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)
            return [pred3]
    
    





def GwcNet_G(d):
    return GwcNet(d, use_concat_volume=False)


def GwcNet_GC(d):
    return GwcNet(d, use_concat_volume=True)


class SimpleAudioDepthNet(nn.Module):
    ## strucure adapted from VisualEchoes [ECCV 2020]
    """A Simple 3-Conv CNN followed by a fully connected layer
    """
    def __init__(self, conv1x1_dim, audio_shape, audio_feature_length, output_nc=1):
        super(SimpleAudioDepthNet, self).__init__()
        self._n_input_audio = audio_shape[0]
        # kernel size for different CNN layers
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
        # strides for different CNN layers
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]
        cnn_dims = np.array(audio_shape[1:], dtype=np.float32)

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = self._conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        self.conv1 = create_conv(self._n_input_audio, 32, kernel=self._cnn_layers_kernel_size[0], paddings=0, stride=self._cnn_layers_stride[0])
        self.conv2 = create_conv(32, 64, kernel=self._cnn_layers_kernel_size[1], paddings=0, stride=self._cnn_layers_stride[1])
        self.conv3 = create_conv(64, conv1x1_dim, kernel=self._cnn_layers_kernel_size[2], paddings=0, stride=self._cnn_layers_stride[2])
        layers = [self.conv1, self.conv2, self.conv3]
        self.feature_extraction = nn.Sequential(*layers)
        self.conv1x1 = create_conv(conv1x1_dim * cnn_dims[0] * cnn_dims[1], audio_feature_length, 1, 0)

        
        self.rgbdepth_upconvlayer1 = unet_upconv(512, 512) #1016 (audio-visual feature) = 512 (visual feature) + 504 (audio feature)
        self.rgbdepth_upconvlayer2 = unet_upconv(512, 256)
        self.rgbdepth_upconvlayer3 = unet_upconv(256, 128)
        self.rgbdepth_upconvlayer4 = unet_upconv(128, 64)
        self.rgbdepth_upconvlayer5 = unet_upconv(64, 32)
        self.rgbdepth_upconvlayer6 = unet_upconv(32, 16)
        self.rgbdepth_upconvlayer7 = unet_upconv(16, output_nc, True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)

    def _conv_output_dim(
        self, dimension, padding, dilation, kernel_size, stride
    ):
        r"""Calculates the output height and width based on the input
        height and width to the convolution layer.
        ref: https://pytorch.org/docs/master/nn.html#torch.nn.Conv2d
        """
        assert len(dimension) == 2
        out_dimension = []
        for i in range(len(dimension)):
            out_dimension.append(
                int(
                    np.floor(
                        (
                            (
                                dimension[i]
                                + 2 * padding[i]
                                - dilation[i] * (kernel_size[i] - 1)
                                - 1
                            )
                            / stride[i]
                        )
                        + 1
                    )
                )
            )
        return tuple(out_dimension)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.shape[0], -1, 1, 1)
        x = self.conv1x1(x)
        
        audio_feat = x
        
        rgbdepth_upconv1feature = self.rgbdepth_upconvlayer1(audio_feat) # [B, 512, 2, 2]
        rgbdepth_upconv2feature = self.rgbdepth_upconvlayer2(rgbdepth_upconv1feature) # [B, 256, 4, 4]
        rgbdepth_upconv3feature = self.rgbdepth_upconvlayer3(rgbdepth_upconv2feature) # [B, 128, 8, 8]
        rgbdepth_upconv4feature = self.rgbdepth_upconvlayer4(rgbdepth_upconv3feature) # [B, 64, 16, 16]
        rgbdepth_upconv5feature = self.rgbdepth_upconvlayer5(rgbdepth_upconv4feature) # [B, 32, 32, 32]
        rgbdepth_upconv6feature = self.rgbdepth_upconvlayer6(rgbdepth_upconv5feature) # [B, 16, 64, 64]
        depth_prediction = self.rgbdepth_upconvlayer7(rgbdepth_upconv6feature) # [B, 1, 128, 128]
        #return depth_prediction, audio_feat
        return depth_prediction, [rgbdepth_upconv4feature, rgbdepth_upconv5feature, rgbdepth_upconv6feature], audio_feat


def warp(x, disp):
    '''
    warp an image/tensor(im2) back to im1, according to the disparity

    x: [B, C, H, W] (im2)
    disp: [B, 1, H, W] (disparity)
        
    '''
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()
    if x.is_cuda:
        grid = grid.cuda()
    flow_y = torch.zeros([B, 1, H, W]).cuda()
    flow = torch.cat((disp, flow_y), 1)
    vgrid = Variable(grid) - flow
    # scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0
    vgrid = vgrid.permute(0,2,3,1)
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)
    mask[mask<0.9999] = 0
    mask[mask>0] = 1

    return output*mask



class Relative_Confidence(nn.Module):
    def __init__(self):
        super(Relative_Confidence, self).__init__()

        # Audio Confidence
        
        self.audio_conv1 = nn.Sequential(convbn(7, 32, 3, 1, 1, 1),
                                   nn.ReLU(inplace=True))
        self.audio_conv2 = nn.Sequential(convbn(32, 32, 3, 1, 1, 1),
                                   nn.ReLU(inplace=True))
        self.audio_conv3 = nn.Sequential(convbn(32, 16, 3, 1, 1, 1),
                                   nn.ReLU(inplace=True))
        
        # Stereo Confidence
        self.stereo_conv1 = nn.Sequential(convbn(7, 32, 3, 1, 1, 1),
                                   nn.ReLU(inplace=True))
        self.stereo_conv2 = nn.Sequential(convbn(32, 32, 3, 1, 1, 1),
                                   nn.ReLU(inplace=True))
        self.stereo_conv3 = nn.Sequential(convbn(32, 16, 3, 1, 1, 1),
                                   nn.ReLU(inplace=True))

        # Relative Confidence
        self.fusion_conv = nn.Sequential(convbn(32, 32, 3, 1, 1, 1),
                                   nn.ReLU(inplace=True))
        self.relative_conv = nn.Sequential(nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
                             nn.Sigmoid())



    def forward(self, left, depth_est, error_map, audio_depth, audio_error_map):
        
        x = torch.cat((left, audio_depth, audio_error_map), dim=1)
        x = self.audio_conv1(x)
        x = self.audio_conv2(x)
        x = self.audio_conv3(x)
        

        y = torch.cat((left, depth_est, error_map), dim=1)
        y = self.stereo_conv1(y)
        y = self.stereo_conv2(y)
        y = self.stereo_conv3(y)

        
        z = torch.cat((x,y), dim=1)
        z = self.fusion_conv(z)
        confidence = self.relative_conv(z)

        return confidence



class AudioVisualModel(nn.Module):
    def __init__(self, maxdisp):
        super(AudioVisualModel, self).__init__()
        #initialize model
        self.net_audio = SimpleAudioDepthNet(8, audio_shape=[2,257,166], audio_feature_length=512) # replica 166 mp3d 121

        self.net_gwcnet = GwcNet(maxdisp, use_concat_volume=False)

        self.net_relative_confidence = Relative_Confidence()
     

    def forward(self, left, right, audio, volatile=False):
        audio_depth_temp, audio_feat, audio_mid_feat = self.net_audio(audio)
        audio_depth = audio_depth_temp.squeeze(1)
        
        disp_ests = self.net_gwcnet(left, right, audio_feat) 
        
        depth_ests = []
        for disp_est in disp_ests:

            #**************************************************************
            #     relative confidence estimation
            #**************************************************************
            disp_est_temp = disp_est.unsqueeze(1) 
            left_est = warp(right, disp_est_temp)
            error_map = torch.abs(left_est - left)

            audio_disp_est_temp = 5.0 / (audio_depth_temp)
            left_est = warp(right, audio_disp_est_temp)
            audio_error_map = torch.abs(left_est - left)

            confidence = self.net_relative_confidence(left, disp_est_temp, error_map, audio_disp_est_temp, audio_error_map)
            confidence = confidence.squeeze(1)

            depth_est = confidence*(5.0/(disp_est)) + (1-confidence)*audio_depth
            depth_ests.append( depth_est )
            
            
            
        return depth_ests 


def StereoEchoesModel(d):
    return AudioVisualModel(d)
