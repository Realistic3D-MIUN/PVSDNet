import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
import torchvision
import rff.layers as rff
import parameters_pvsdnet as params
import helperFunctions as helper

def getLinearLayer(in_feat, out_feat, activation=nn.ReLU(True)):
    return nn.Sequential(
        nn.Linear(in_features=in_feat, out_features=out_feat, bias=True),
        activation
    )

def getConvLayer(in_channel,out_channel,stride=1,padding=1,activation=nn.ReLU()):
    return nn.Sequential(nn.Conv2d(in_channel, 
                    out_channel,
                    kernel_size=3,
                    stride=stride,
                    padding=padding,
                    padding_mode='reflect'),
                    activation)

def getConvTransposeLayer(in_channel, out_channel,kernel=3,stride=1,padding=1,activation=nn.ReLU()):
    return nn.Sequential(nn.ConvTranspose2d(in_channel,
                                            out_channel,
                                            kernel_size = kernel,
                                            stride=stride,
                                            padding=padding),
                                            activation)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1):
        return input.view(input.size(0), 1, params.params_height//8, params.params_width//8)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.stride = stride

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out = out + self.shortcut(residual)
        out = self.relu(out)
        return out

class MLPEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = params.params_m
        self.positional_encoding = rff.PositionalEncoding(sigma=1,m=self.m)
        self.layer1 = getLinearLayer(2*3*self.m, 1024) # 2*3*m = 12, here m=32
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = getLinearLayer(1024, 2048)
        self.dropout2 = nn.Dropout(0.2)
        self.layer3 = getLinearLayer(2048, (params.params_height//8)*(params.params_width//8))
        self.unflat = UnFlatten()

        self.up_layer1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_layer2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_layer3 = nn.Upsample(scale_factor=2, mode='nearest')
    

    def forward(self, x):
        x = self.positional_encoding(x)

        x = self.layer1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.dropout2(x)
        x = self.layer3(x)

        x = self.unflat(x)

        x = self.up_layer1(x)
        x = self.up_layer2(x)
        x = self.up_layer3(x)
        return x
    
class UpperEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet152(pretrained=False)
        layers = list(model.children())
        self.ResNetEncoder = torch.nn.Sequential(*layers[:5].copy())
        del model

    def forward(self, x):
        x1 = x[:, 0:3, :, :]
        x1 = self.ResNetEncoder(x1)
        return x1

    def apply_resnet_encoder(self, x):
        x1 = x[:, 0:3, :, :]
        x1 = self.ResNetEncoder(x1)
        return x1


class LowerEncoder(nn.Module):
    def __init__(self,total_image_input=1):
        super().__init__()
        self.encoder_pre = ResidualBlock((total_image_input*3)+1, 20)
        self.encoder_layer1 = ResidualBlock(20, 30)
        self.encoder_layer2 = ResidualBlock(30, 50)

        self.encoder_layer3 = nn.Sequential(
            ResidualBlock(50, 100),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.encoder_layer4 = ResidualBlock(100, 200)
        self.encoder_layer5 = nn.Sequential(
            ResidualBlock(200, 200),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.encoder_layer6 = ResidualBlock(200, 200)
        self.encoder_layer7 = nn.Sequential(
            ResidualBlock(200, 200),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.encoder_layer8 = ResidualBlock(200, 500)
        self.encoder_layer9 = nn.Sequential(
            ResidualBlock(500, 500),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
                                
        self.encoder_layer10 = ResidualBlock(500, 500)
        self.encoder_layer11 = ResidualBlock(500, 500)
        
    def forward(self, x):
        x = self.encoder_pre(x)
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        skip1 = self.encoder_layer3(x)
        
        x = self.encoder_layer4(skip1)
        skip2 = self.encoder_layer5(x)

        x = self.encoder_layer6(skip2)
        skip3 = self.encoder_layer7(x)

        x = self.encoder_layer8(skip3)
        skip4 = self.encoder_layer9(x)

        x = self.encoder_layer10(skip4)
        x = self.encoder_layer11(x)

        return x, [skip1, skip2, skip3, skip4]

class MergeDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder_layer1 = ResidualBlock(500, 500)
        self.decoder_layer2 = ResidualBlock(500, 500)
        self.decoder_layer3 = ResidualBlock(500, 500)
        
        self.decoder_layer4 = nn.Sequential(
            nn.ConvTranspose2d(500, 200, 2, stride=2, padding=0), 
            nn.ReLU(True)
        )
        self.decoder_layer5 = ResidualBlock(200, 200)
    
        self.decoder_layer6 = nn.Sequential(
            nn.ConvTranspose2d(200, 200, 2, stride=2, padding=0), 
            nn.ReLU(True)
        )
        self.decoder_layer7 = ResidualBlock(200, 200)

        self.decoder_layer8 = nn.Sequential(
            nn.ConvTranspose2d(200, 100, 2, stride=2, padding=0), 
            nn.ReLU(True)
        )
        self.decoder_layer9 = ResidualBlock(100, 100)

        self.decoder_layer10 = nn.Sequential(
            nn.ConvTranspose2d(100, 100, 2, stride=2, padding=0), 
            nn.ReLU(True)
        )
        self.decoder_layer11 = ResidualBlock(100, 100)
        self.decoder_layer12 = ResidualBlock(100, 50)
        self.decoder_layer13 = ResidualBlock(50, 40)
        self.decoder_layer14 = ResidualBlock(40, 20)
        self.decoder_layer15 = nn.Sequential(
            nn.Conv2d(20, 8, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.decoder_layer16 = nn.Sequential(
            nn.Conv2d(8, 3, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, lower_skip_list, upper_skip_list):
        x = self.decoder_layer1(x)
        x = self.decoder_layer2(x)
        x = x + lower_skip_list[3] + upper_skip_list[1]

        x = self.decoder_layer3(x)
        x = self.decoder_layer4(x)
        x = x + lower_skip_list[2] + upper_skip_list[0]
        
        x = self.decoder_layer5(x)
        x = self.decoder_layer6(x)
        x = x + lower_skip_list[1]

        x = self.decoder_layer7(x)
        x = self.decoder_layer8(x)
        x = x + lower_skip_list[0]

        x = self.decoder_layer9(x)
        x = self.decoder_layer10(x)
        x = self.decoder_layer11(x)
        x = self.decoder_layer12(x)
        x = self.decoder_layer13(x)
        x = self.decoder_layer14(x)
        x = self.decoder_layer15(x)
        x = self.decoder_layer16(x)        
        return x


class DepthDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder_layer1 = ResidualBlock(500, 1400)
        self.decoder_layer2 = ResidualBlock(1400, 1200)
        self.decoder_layer3 = ResidualBlock(1200, 1000)
        
        self.decoder_layer4 = nn.Sequential(
            nn.ConvTranspose2d(1000, 800, 2, stride=2, padding=0), 
            nn.ReLU(True)
        )
        self.decoder_layer5 = ResidualBlock(800, 600)
    
        self.decoder_layer6 = nn.Sequential(
            nn.ConvTranspose2d(600, 400, 2, stride=2, padding=0), 
            nn.ReLU(True)
        )
        self.decoder_layer7 = ResidualBlock(400, 200)

        self.decoder_layer8 = nn.Sequential(
            nn.ConvTranspose2d(200, 100, 2, stride=2, padding=0), 
            nn.ReLU(True)
        )
        self.decoder_layer9 = ResidualBlock(100, 100)

        self.decoder_layer10 = nn.Sequential(
            nn.ConvTranspose2d(100, 100, 2, stride=2, padding=0), 
            nn.ReLU(True)
        )
        self.decoder_layer11 = ResidualBlock(100, 100)
        self.decoder_layer12 = ResidualBlock(100, 50)
        self.decoder_layer13 = ResidualBlock(50, 40)
        self.decoder_layer14 = ResidualBlock(40, 20)
        self.decoder_layer15 = nn.Sequential(
            nn.Conv2d(20, 8, 3, stride=1, padding=1),
            nn.ReLU(True)
        )
        self.decoder_layer16 = nn.Sequential(
            nn.Conv2d(8, 1, 3, stride=1, padding=1),
            nn.ReLU(True)
        )

        self.up_refinement_0 = ResidualBlock(200, 800)
        self.up_refinement_1 = ResidualBlock(500, 1200)

        self.low_refinement_1 = ResidualBlock(200, 400)
        self.low_refinement_2 = ResidualBlock(200, 800)
        self.low_refinement_3 = ResidualBlock(500, 1200)

        
        
    def forward(self, x, lower_skip_list, upper_skip_list):
        x = self.decoder_layer1(x)
        x = self.decoder_layer2(x)

        low_skip_3 = self.low_refinement_3(lower_skip_list[3])
        up_skip_1 = self.up_refinement_1(upper_skip_list[1])
        x = x + low_skip_3 + up_skip_1

        x = self.decoder_layer3(x)
        x = self.decoder_layer4(x)

        low_skip_2 = self.low_refinement_2(lower_skip_list[2])
        up_skip_0 = self.up_refinement_0(upper_skip_list[0])
        x = x + low_skip_2 + up_skip_0
        
        x = self.decoder_layer5(x)
        x = self.decoder_layer6(x)

        low_skip_1 = self.low_refinement_1(lower_skip_list[1])
        x = x + low_skip_1

        x = self.decoder_layer7(x)
        x = self.decoder_layer8(x)
        x = x + lower_skip_list[0]

        x = self.decoder_layer9(x)
        x = self.decoder_layer10(x)
        x = self.decoder_layer11(x)
        x = self.decoder_layer12(x)
        x = self.decoder_layer13(x)
        x = self.decoder_layer14(x)
        x = self.decoder_layer15(x)
        x = self.decoder_layer16(x)        
        return x


class PVSNet_Lite(nn.Module):
    def __init__(self,total_image_input=1):
        super().__init__()
        self.target_positional_embedding = MLPEncoder()
        self.upper_encoder = UpperEncoder()
        self.lower_encoder = LowerEncoder(total_image_input)
        self.merge_decoder = MergeDecoder()

        self.upper_encoder_extra_1 = nn.Sequential(
            ResidualBlock(256, 200),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.upper_encoder_extra_2 = nn.Sequential(
            ResidualBlock(200, 500),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
    def forward(self, x, pos):
        target_position_feature = self.target_positional_embedding(pos)

        # First Encoder Branch
        upper_features_1 = self.upper_encoder.apply_resnet_encoder(x)
        upper_features_1 = self.upper_encoder_extra_1(upper_features_1)
        upper_features_2 = self.upper_encoder_extra_2(upper_features_1)

        # Second Encoder Branch
        stacked_tensor = torch.cat((x,target_position_feature),dim=1)
        lower_feature, skip_list = self.lower_encoder(stacked_tensor)

        # Decoder
        merged_feature = self.merge_decoder(lower_feature, skip_list, [upper_features_1, upper_features_2])
        return merged_feature




class PVSDNet_Lite(nn.Module):
    def __init__(self,total_image_input=1):
        super().__init__()
        self.target_positional_embedding = MLPEncoder()
        self.upper_encoder = UpperEncoder()
        self.lower_encoder = LowerEncoder(total_image_input)
        self.merge_decoder = MergeDecoder()
        self.depth_decoder = DepthDecoder()

        self.upper_encoder_extra_1 = nn.Sequential(
            ResidualBlock(256, 200),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.upper_encoder_extra_2 = nn.Sequential(
            ResidualBlock(200, 200),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        print("Loading pre-trained nvs net")
        base_net = PVSNet_Lite(total_image_input)
        #base_net = helper.load_Checkpoint("./checkpoint/checkpoint_init_pvsnet.pth", base_net, load_cpu=True)

        self.target_positional_embedding = base_net.target_positional_embedding
        self.upper_encoder = base_net.upper_encoder
        self.lower_encoder = base_net.lower_encoder
        self.merge_decoder = base_net.merge_decoder
        self.upper_encoder_extra_1 = base_net.upper_encoder_extra_1
        self.upper_encoder_extra_2 = base_net.upper_encoder_extra_2
        del base_net
        print("Loading pre-trained nvs net: Done")

        
    def forward(self, x, pos):
        target_position_feature = self.target_positional_embedding(pos)

        # First Encoder Branch
        upper_features_1 = self.upper_encoder.apply_resnet_encoder(x)
        upper_features_1 = self.upper_encoder_extra_1(upper_features_1)
        upper_features_2 = self.upper_encoder_extra_2(upper_features_1)

        # Second Encoder Branch
        stacked_tensor = torch.cat((x,target_position_feature),dim=1)
        lower_feature, skip_list = self.lower_encoder(stacked_tensor)

        # Decoder
        merged_feature = self.merge_decoder(lower_feature, skip_list, [upper_features_1, upper_features_2])

        # Depth Decoder
        depth_feature = self.depth_decoder(lower_feature, skip_list, [upper_features_1, upper_features_2])
        return merged_feature, depth_feature