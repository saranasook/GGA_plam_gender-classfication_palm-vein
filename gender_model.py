import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


###--- 1. GGAModule & Model Definition (as previously defined) ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool                   = nn.AdaptiveAvgPool2d(1)
        self.max_pool                   = nn.AdaptiveMaxPool2d(1)
        self.fc                         = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid                    = nn.Sigmoid()
    
    def forward(self,x):
        avg_out                         = self.fc(self.avg_pool(x))
        max_out                         = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1                      = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid                    = nn.Sigmoid()
    
    def forward(self, x):
        avg_out                         = torch.mean(x, dim=1, keepdim=True)
        max_out, _                      = torch.max(x, dim=1, keepdim=True)
        x                               = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))


class GGAModule(nn.Module):
    def __init__(self, in_planes):
        super(GGAModule, self).__init__()
        self.ca                         = ChannelAttention(in_planes)
        self.sa                         = SpatialAttention()

    def forward(self, x):
        x                               = x * self.ca(x)
        x                               = x * self.sa(x)
        return x

######## Backbone = Resnet-50 ################
class ResNet50_GGA_Binary(nn.Module):
    def __init__(self):
        super(ResNet50_GGA_Binary, self).__init__()
        resnet                          = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.gga                        = GGAModule(2048)
        self.avgpool                    = nn.AdaptiveAvgPool2d((1, 1))
        # Single output node for binary classification
        self.fc                         = nn.Linear(2048, 1) 
        self.sigmoid                    = nn.Sigmoid() # Sigmoid for probability output

    def forward(self, x):
        x                               = self.backbone(x)
        x                               = self.gga(x)
        x                               = self.avgpool(x)
        x                               = torch.flatten(x, 1)
        x                               = self.fc(x)
        return self.sigmoid(x)

class DenseNet161_GGA(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet161_GGA, self).__init__()
        # Load pre-trained DenseNet-161
        densenet = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
        
        # DenseNet features: final block outputs 2208 channels
        self.features = densenet.features
        self.gga = GGAModule(2208)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(2208,num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.relu(x, inplace=True) # Final ReLU before global pool
        x = self.gga(x) # Inject GGA to refine vascular features
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)



class DenseNet161_GGA_Binary(nn.Module):
    def __init__(self):
        super(DenseNet161_GGA_Binary, self).__init__()
        # Load pre-trained DenseNet-161
        densenet = models.densenet161(weights=models.DenseNet161_Weights.DEFAULT)
        
        # DenseNet features: final block outputs 2208 channels
        self.features = densenet.features
        self.gga = GGAModule(2208)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(2208,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.relu(x, inplace=True) # Final ReLU before global pool
        x = self.gga(x) # Inject GGA to refine vascular features
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class DesnseNet161(nn.Module):
    def __init__(self, class_size=2, pretrained=True):
        super(DesnseNet161, self).__init__()

        ### Load the base model ###
        weights                 = models.DenseNet161_Weights.DEFAULT if pretrained else None
        self.backbone           = models.densenet161(weights=weights)

        ### Extract the input dimension of the original classifier ####
        in_features             = self.backbone.classifier.in_features

        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features,1),
            nn.Sigmoid()
        )
    def forward(self, x):

        return self.backbone(x)



# ----------------------------------------- ResNet-101 Implementation -----------------------------------#
class ResNet101_GGA_Binary(nn.Module):
    def __init__(self):
        super(ResNet101_GGA_Binary, self).__init__()
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        # ResNet-101 layer4 output is 2048 channels
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.gga = GGAModule(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        x = self.gga(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.sigmoid(self.fc(x))

class ResNet101(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True):
        super(ResNet101, self).__init__()
        
        weights             = models.ResNet101_Weights.DEFAULT if pretrained else None
        self.model          = models.resnet101(weights=weights)
        
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # ResNet uses '.fc' for the final layer
        in_features         = self.model.fc.in_features
        
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)





# ------------------------------------ VGG-19 Implementation ----------------------------#
class VGG19_GGA_Binary(nn.Module):
    def __init__(self):
        super(VGG19_GGA_Binary, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        # VGG-19 feature map output is 512 channels
        self.features       = vgg.features
        self.gga            = GGAModule(512)
        self.avgpool        = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier     = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gga(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class VGG19(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True):
        super(VGG19, self).__init__()
        
        weights = models.VGG19_BN_Weights.DEFAULT if pretrained else None
        self.model = models.vgg19_bn(weights=weights)
        
        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False
        
        # VGG uses a '.classifier' block (which is a Sequential container)
        # We replace the very last layer (index 6) of that block
        in_features = self.model.classifier[6].in_features
        
        self.model.classifier[6] = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
