''' Spatial-Temporal Transformer Networks
'''
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
from vit_pytorch import ViT
import timm
from core.spectral_norm import spectral_norm as _spectral_norm
# from torchsummary import summary
# import matplotlib
# matplotlib.use('Agg')

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class InpaintGenerator(BaseNetwork):
    def __init__(self, Transformer_layers, Transformer_heads, channel, patchsize, backbone, classes=2, init_weights=True):
        super(InpaintGenerator, self).__init__()
        self.channel = channel#256
        self.backbone = backbone
        self.classes = classes
        stack_num = Transformer_layers ## 8 ## number of layers of Transformer
        #patchsize = [(108, 60), (36, 20), (18, 10), (9, 5)]
        #patchsize = [(54, 30), (18, 10), (9,5)]
        #patchsize = [(7, 7), (7, 7), (7, 7)]

        patchsize = [(patchsize[x], patchsize[x]) for x in Transformer_heads]
        blocks = []
        for _ in range(stack_num):
            if self.backbone == "Conv" or self.backbone=="EfficientNet":
                blocks.append(TransformerBlock(patchsize, d_input=80, hidden=channel)) ## d_input is 80 for Efficient backbone
            elif self.backbone == "Vit":
                #blocks.append(TransformerBlock(patchsize, d_input=65, hidden=channel)) ## d_input is 65 for Vit from vit_pytorch
                blocks.append(TransformerBlock(patchsize, d_input=3, hidden=channel)) ## d_input is 65 for Vit from timm
            else:
                raise ValueError("Indicate right backbone!")

        self.transformer = nn.Sequential(*blocks)

        ## Vit initialization
        vit = ViT(
            image_size=224,
            patch_size=28,
            num_classes=1000,
            dim=1024,
            depth=6,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

        if self.backbone == "Conv":
            ## Encoder Backbone (with initialization): 4 layers vanille CNN
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, channel, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )

        # # decoder: decode frames from features
        # self.decoder = nn.Sequential(
        #     deconv(channel, 128, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     deconv(64, 64, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        # )


        if self.backbone == "Conv":
            ## encoder binary classification
            self.fc1 == nn.Linear(channel*30*54, 1024) ## encoder embedding
        elif self.backbone == "ResNet50":
            self.fc1 = nn.Linear(channel*30*54, 1024) ## encoder embedding
        elif self.backbone == "ResNext50":
            self.fc1 = nn.Linear(channel*30*54, 1024) ## encoder embedding
        elif self.backbone == "EfficientNet":
            self.fc1 = nn.Linear(channel*28*28, 1024) ## EfficientNet backbone
        elif self.backbone == "Vit":
            #self.fc1 = nn.Linear(channel*32*32, 1024) ## Vit from vit_pytorch
            self.fc1 = nn.Linear(channel * 16 * 16, 1024)  ## Vit from timm
        else:
            raise ValueError("Indicate right backbone!")

        self.fc2 = nn.Linear(1024, self.classes)

        # ## decoder binary classification
        # #self.fc3 = nn.Linear(3*240*432, 1024) ## decoder embedding
        # self.fc3 = nn.Linear(3*120*216, 1024) ## decoder embedding
        # self.fc4 = nn.Linear(1024, 2)

        if init_weights:
            self.init_weights()

        if self.backbone != "Conv":
            if self.backbone == "ResNet50":
                ## Encoder Backbone (pretrained, w/o initialization): ResNet50
                self.encoder = models.wide_resnet50_2(pretrained=True)
            elif self.backbone == "ResNext50":
                ## Encoder Backbone (pretrained, w/o initialization): Resnext50
                model_backbone = models.resnext50_32x4d(pretrained=True)  ##pretrained on imagenet50, don't init weights
                #model_backbone = models.resnext50_32x4d(pretrained=False) ##training from scratch            
            elif self.backbone == "EfficientNet":
                ## Encoder Backbone(pretrained, w/o initialization): Efficientnet
                model_backbone = EfficientNet.from_pretrained("efficientnet-b0", advprop=True) ##pretrained
                # model_backbone = EfficientNet.from_name("efficientnet-b0") ##from scratch
            elif self.backbone == "Vit":
                ## Encoder Backbone : Vit
                #model_backbone = vit ##Vit scratch from vit_pytorch

                ## Encoder Backbone : Vit from timm, base vit: heads=12, depth=8
                model_backbone = timm.create_model('vit_base_patch16_224', pretrained=False) ## vit from timm: scratch
                #model_backbone = timm.create_model('vit_base_patch16_224', pretrained=True) ## vit from timm: pretrained with imagenet
            else:
                raise ValueError("Indicate right backbone!")

            self.encoder = model_backbone


        # ACT
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)
        self.dropout =nn.Dropout(0.5)

    def forward(self, masked_frames, masks):
        # extracting features
        b, t, c, height, width = masked_frames.size()
        masks = masks.contiguous().view(b*t, 1, height, width)

        if self.backbone == 'Conv' or self.backbone == 'ResNet50' or self.backbone == 'ResNext50':
            ## embeddings from backbone vanille CNN, ResNet50, ResNext50
            enc_feat = self.encoder(masked_frames.view(b*t, c, h, w))
        elif self.backbone == 'EfficientNet':
            ## embeddings from backbone EfficientNet
            enc_feat = self.encoder.extract_features(masked_frames.view(b*t, c, height, width)) ## enc_feat shape:(b*t,1280,7x7), decided by the input 224x224 and EfficientNet
            enc_feat = enc_feat.view(b * t, 80, 28, 28)  ### enc_feat (b*t,1280,7x7)=>=>conv1x1 =>embedding(self.channel, 28,28)
        elif self.backbone == 'Vit':
            # embeddings from backbone Vit from vit_pytorch (scratch)
            # vit_mask = torch.ones(1, 8, 8).bool()  # optional mask, designating which patch to attend to
            # enc_feat = self.encoder(masked_frames.view(b * t, c, height, width), mask=vit_mask)  # vit_mask designating which patch to attend to, optional
            # enc_feat = enc_feat.view(b * t, 65, 32, 32)  ### enc_feat (b*t,65,1024)=>(b*t,65,32*32)

            # embeddings from backbone Vit pretrained
            enc_feat = self.encoder.forward_features(masked_frames.view(b * t, c, height, width)) ## vit embeddings is the cls_token
            enc_feat = enc_feat.view(b * t, 3, 16, 16)  ### enc_feat (b*t,65,768)=>(b*t,65,32*32)
        else:
            raise ValueError("Indicate backbone!")

        _, c, h, w = enc_feat.size()
        
        y = torch.flatten(enc_feat, 1)
        y = self.fc1(y)
        y = self.LeakyReLU(y)
        y = self.dropout(y)
        y = self.fc2(y)

        ## Normalization of the embedding y by Norm-2, otherwise the extreme large/small
        ## y will explore the loss
        #y_enc = F.normalize(y, p=2, dim=1)
        y_enc = y
        # #### branch decoder)  ##########
        # y = torch.flatten(dec_feat, 1)
        # y = self.fc3(y)
        # y = self.LeakyReLU(y)
        # y = self.dropout(y)
        # y_dec = self.fc4(y)
        # ############################################################################
        # ### MTL for FAS: branch transformer, branch depth map softmax-layer ########
        # ############################################################################
        return y_enc
        # return output, y_enc, y_dec
        # return output, y_enc

    #def infer(self, faces, masks):
    def infer(self, masked_frames, masks):
        # t, c, h, w = masks.size()
        # masks = masks.view(t, c, h, w)
        # masks = F.interpolate(masks, scale_factor=1.0/8)
        # b, t, c, h, w = faces.size()
        # feat = self.encoder.extract_features(faces.view(b*t, c, h, w))
        # feat = feat.view(t, self.channel, 28, 28)
        # _, c, h, w = feat.size()
        # enc_feat = self.transformer(
        #     {'x': feat, 'm': masks, 'b': 1, 'c': c})['x']
        #
        # y = torch.flatten(enc_feat, 1)
        # y = self.fc1(y)
        # y = self.LeakyReLU(y)
        # #y = self.dropout(y)
        # y = self.fc2(y)
        #
        # ## Normalization of the embedding y by Norm-2, otherwise the extreme large/small
        # ## y will explore the loss
        # y_enc = F.normalize(y, p=2, dim=1)
        # #y_enc = y
        # return y_enc

        # extracting features
        b, t, c, height, width = masked_frames.size()
        masks = masks.view(b * t, 1, height, width)
        enc_feat = self.encoder.extract_features(masked_frames.view(b * t, c, height, width))
        enc_feat = enc_feat.view(b * t, 80, 28, 28)
        _, c, h, w = enc_feat.size()
        masks = F.interpolate(masks, scale_factor=1.0 / (height / h))  ## masks keep the same size as enc_feat
        enc_feat = self.transformer(
            {'x': enc_feat, 'm': masks, 'b': b, 'c': self.channel})['x']

        y = torch.flatten(enc_feat, 1)
        y = self.fc1(y)
        y = self.LeakyReLU(y)
        y = self.dropout(y)
        y = self.fc2(y)

        y_enc = y
        return y_enc


class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        return self.conv(x)


# #############################################################################
# ############################# Transformer  ##################################
# #############################################################################


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, m):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        ## The values of scores in the positions with the indices
        ## where the values of the sampe positions equal to 1 in the mask will be replaced by -1e9.
        ## This is in order to mask the non ROI zone (mask==1). However  in this work all zone is the ROI zone,
        ## thus all zone has been kept and no zone is replaced by -1e9.
        scores.masked_fill(m, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, patchsize, d_input, d_model):
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(
            d_input, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(
            d_input, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(
            d_input, d_model, kernel_size=1, padding=0)
        self.   output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.attention = Attention()

    def forward(self, x, m, b, c):
        bt, _, h, w = x.size()
        t = bt // b
        #d_k = c // len(self.patchsize)
        #d_k = int(c / len(self.patchsize)+1.0)
        output = []
        _query = self.query_embedding(x)
        _key = self.key_embedding(x)
        _value = self.value_embedding(x)
        ### multiple heads  : heads number = len(self.patchsize), each head has d_k=channel/len(self.patchsize) dimension
        for (width, height), query, key, value in zip(self.patchsize,
                                                      torch.chunk(_query, len(self.patchsize), dim=1),
                                                      torch.chunk(_key, len(self.patchsize), dim=1),
                                                      torch.chunk(_value, len(self.patchsize), dim=1)):
            d_k = query.shape[-3]
            out_w, out_h = w // width, h // height
            mm = m.view(b, t, 1, out_h, height, out_w, width)
            mm = mm.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, height*width)
            mm = (mm.mean(-1) > 0.5).unsqueeze(1).repeat(1, t*out_h*out_w, 1)
            # 1) embedding and reshape
            query = query.view(b, t, d_k, out_h, height, out_w, width)
            query = query.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            key = key.view(b, t, d_k, out_h, height, out_w, width)
            key = key.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            value = value.view(b, t, d_k, out_h, height, out_w, width)
            value = value.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(
                b,  t*out_h*out_w, d_k*height*width)
            '''
            # 2) Apply attention on all the projected vectors in batch.
            tmp1 = []
            for q,k,v in zip(torch.chunk(query, b, dim=0), torch.chunk(key, b, dim=0), torch.chunk(value, b, dim=0)):
                y, _ = self.attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
                tmp1.append(y)
            y = torch.cat(tmp1,1)
            '''
            y, _ = self.attention(query, key, value, mm)
            # 3) "Concat" using a view and apply a final linear.
            y = y.view(b, t, out_h, out_w, d_k, height, width)
            y = y.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt, d_k, h, w)
            output.append(y)
        output = torch.cat(output, 1)
        x = self.output_linear(output)
        return x


# Standard 2 layerd FFN of transformer
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, patchsize, d_input, hidden=128):
        super().__init__()
        self.attention = MultiHeadedAttention(patchsize, d_input, d_model=hidden)
        self.feed_forward = FeedForward(hidden)

    def forward(self, x):
        x, m, b, c = x['x'], x['m'], x['b'], x['c']
        x = x + self.attention(x, m, b, c)
        x = x + self.feed_forward(x)
        return {'x': x, 'm': m, 'b': b, 'c': c}


# ######################################################################
# ######################################################################


class Discriminator(BaseNetwork):
    def __init__(self, in_channels=3, use_sigmoid=False, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 64

        self.conv = nn.Sequential(
            spectral_norm(nn.Conv3d(in_channels=in_channels, out_channels=nf*1, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf*1, nf*2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 2, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5),
                      stride=(1, 2, 2), padding=(1, 2, 2))
        )

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # T, C, H, W = xs.shape
        xs_t = torch.transpose(xs, 0, 1)
        xs_t = xs_t.unsqueeze(0)  # B, C, T, H, W
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return out


def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module
