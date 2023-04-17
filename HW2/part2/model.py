import torch
import torch.nn as nn
import torchvision.models as models

class MyNet(nn.Module): 
    def __init__(self, num_out=10):
        super(MyNet, self).__init__()
        
        ################################################################
        # TODO:                                                        #
        # Define your CNN model architecture. Note that the first      #
        # input channel is 3, and the output dimension is 10 (class).  #
        ################################################################

        self.conv1 = nn.Sequential(nn.Conv2d(3,6,kernel_size=5, stride=1),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             )
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,kernel_size=5),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.fc1 = nn.Sequential(nn.Linear(400, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120,90), nn.ReLU())
        self.fc3 = nn.Linear(90,num_out)


    def forward(self, x):

        ##########################################
        # TODO:                                  #
        # Define the forward path of your model. #
        ##########################################

        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        # It is important to check your shape here,
        # so that you know how manys nodes are there in first FC in_features
        #print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)        
        out = x
        return out
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        ############################################
        # NOTE:                                    #
        # Pretrain weights on ResNet18 is allowed. #
        ############################################

        # (batch_size, 3, 32, 32)
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.resnet = models.resnet18(weights=weights)
        # (batch_size, 512)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
        # (batch_size, 10)

        #######################################################################
        # TODO (optinal):                                                     #
        # Some ideas to improve accuracy if you can't pass the strong         #
        # baseline:                                                           #
        #   1. reduce the kernel size, stride of the first convolution layer. # 
        #   2. remove the first maxpool layer (i.e. replace with Identity())  #
        # You can run model.py for resnet18's detail structure                #
        #######################################################################
        

    def forward(self, x):
        return self.resnet(x)

class residual_block(nn.Module):
    def __init__(self, in_channels):
        super(residual_block, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                             out_channels=in_channels, 
                                             kernel_size=3, padding=1), 
                                   nn.BatchNorm2d(in_channels))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                             out_channels=in_channels, 
                                             kernel_size=3, padding=1), 
                                   nn.BatchNorm2d(in_channels))
        self.relu = nn.ReLU()
        
    def forward(self,x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out

class myResnet(nn.Module):
    def __init__(self, in_channels=3, num_out=10):
        super(myResnet, self).__init__()
        self.stem_conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.layer1 = residual_block(64)
        self.cnn_layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                        nn.MaxPool2d(kernel_size=2, stride=2),
                                        nn.BatchNorm2d(64, affine = True),
                                        nn.ReLU())
        self.layer3 = residual_block(64)
        self.cnn_layer4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1),
                                        nn.MaxPool2d(kernel_size=2, stride=2),
                                        nn.BatchNorm2d(128, affine = True),
                                        nn.ReLU())
        self.layer5 = residual_block(128)
        self.fc1 = nn.Sequential(nn.Linear(4608, 512), nn.ReLU())
        self.fc2 = nn.Linear(512, num_out)

    def forward(self,x):
        x = self.stem_conv(x)
        x = self.layer1(x)
        x = self.cnn_layer2(x)
        x = self.layer3(x)
        x = self.cnn_layer4(x)
        x = self.layer5(x)
        x = torch.flatten(x, 1)
        
        # It is important to check your shape here so that you know how manys nodes are there in first FC in_features
#         print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)     
        out = x
        return out

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
if __name__ == '__main__':
    model = ResNet18()
    print(model)
