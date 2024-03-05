# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# modified from https://github.com/facebookresearch/DomainBed/blob/main/domainbed/networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torch.utils.model_zoo as model_zoo
from domainbed.lib.TYY_stodepth_lineardecay import resnet18_StoDepth_lineardecay, resnet50_StoDepth_lineardecay
from domainbed.lib import wide_resnet
# from domainbed.mybatchnorm import DropoutFriendBatchNorm1d
from domainbed.resnet50dp import resnet50dp 
#from domainbed.resnet50_unfold import resnet50
import copy
model_urls = {'v1':{'resnet18': "https://download.pytorch.org/models/resnet18-f37072fd.pth",
                    'resnet50':'https://download.pytorch.org/models/resnet50-0676ba61.pth'},
             'v2':{'resnet50':'https://download.pytorch.org/models/resnet50-11ad3fa6.pth'},
             # 'swav800':{'resnet50':'https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar'},
             # 'swav400':{'resnet50':'https://dl.fbaipublicfiles.com/deepcluster/swav_400ep_pretrain.pth.tar'},
             # 'vicreg': {'resnet50':'https://dl.fbaipublicfiles.com/vicreg/resnet50.pth'}
             }

def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x


class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    
    def __init__(self, input_shape, hparams):
        super(ResNet, self).__init__()
        if 'dropout_rate_per_group' in hparams and max(hparams['dropout_rate_per_group']) > 0:
            assert hparams['resnet18'] is False
            
            
            tmp = torchvision.models.resnet50(pretrained=True)
            tmpparams = tmp.state_dict()

            self.network = resnet50dp(dropout_rate_per_group = hparams['dropout_rate_per_group'], \
                dropout_kernel=hparams['dropout_kernel'], dropout_type=hparams['dropout_type'],dropout_conv_only=hparams['dropout_conv_only'])
            params = self.network.state_dict()
            for key in params:
                if key in tmpparams:
                    params[key] = torch.clone(tmpparams[key])
            self.network.fc = Identity()        
            
            del tmp;
            del tmpparams;
            
            self.n_outputs = 2048
            
            #0/0
        else:
            if hparams['resnet18']:
                self.n_outputs = 512
                
                if hparams['stodepth'] < 1:
                    if hparams['stodepth_uniform']:
                        self.network = resnet18_StoDepth_lineardecay(pretrained=True, prob_0_L=[hparams['stodepth'],hparams['stodepth']], multFlag=False) 
                        print('stodepth uniform')
                    else:  
                        self.network = resnet18_StoDepth_lineardecay(pretrained=True, prob_0_L=[1,hparams['stodepth']], multFlag=False) 
                else:
                    self.network = torchvision.models.resnet18(pretrained=False)
                
                version = 'v1'
                if 'pretrain_version' in hparams:
                    version = hparams['pretrain_version']
                if version != 'none':
                    model.load_state_dict(model_zoo.load_url(model_urls[version]['resnet18']))
                else:
                    print('scratch training')

            else:
                if hparams['stodepth'] < 1:
                    if hparams['stodepth_uniform']:
                        self.network = resnet50_StoDepth_lineardecay(pretrained=True, prob_0_L=[hparams['stodepth'],hparams['stodepth']], multFlag=False) 
                    else:
                        self.network = resnet50_StoDepth_lineardecay(pretrained=True, prob_0_L=[1,hparams['stodepth']], multFlag=False) 
                    
                    print('stodepth')
                else:
                    # if 'unfoldresnet' in hparams and hparams['unfoldresnet']:
                    #     self.network = resnet50()
                    # else:
                    #     self.network = torchvision.models.resnet50(pretrained=False)
                    self.network = torchvision.models.resnet50(pretrained=False)
                
                version = 'v1'
                if 'pretrain_version' in hparams:
                    version = hparams['pretrain_version']
                print('version',version)
                if version != 'none':
                    state_dict = model_zoo.load_url(model_urls[version]['resnet50'])
                    if 'swav' in version:
                        new_state_dict = {}
                        for key in state_dict:
                            if key.startswith('module.'):
                                new_state_dict[key[len("module."):]] = state_dict[key]

                        state_dict = new_state_dict

                    msg = self.network.load_state_dict(state_dict, strict=False)
                    print('model pretrained weights loading msg')
                    print(msg)
                else:
                    print('scratch training')
                #assert len(msg.missing_keys) <= 2 
                self.n_outputs = 2048


            
        # self.network = remove_batch_norm_from_resnet(self.network)

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        del self.network.fc
        self.network.fc = Identity()
        if 'freeze_bn' in hparams and hparams['freeze_bn']:
            print('freeze batch norm...')
            self.freeze_bn()

        self.hparams = hparams
        #Jianyu aug 10th, 2023: move drop to classfiers after batchnorm, if there is a batchnorm
        #self.dropout = nn.Dropout(hparams['resnet_dropout'])

        if 'bias_only' in self.hparams  and self.hparams['bias_only']:
            self.freeze_nonbias()
            print('freeze non bias')

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        #return self.dropout(self.network(x))
        return self.network(x)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if 'freeze_bn' in self.hparams and self.hparams['freeze_bn']:
            self.freeze_bn()
        if 'bias_only' in self.hparams  and self.hparams['bias_only']:
            self.freeze_nonbias()

    def freeze_bn(self):
        for m in self.network.modules():

            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def freeze_nonbias(self):
        for name, param in self.network.named_parameters():
            if 'bias' not in name:
                param.requires_grad_ = False
                #print(name)


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)



def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], hparams["mlp_width"], hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False, hparams=None):
    

    if is_nonlinear:
        return torch.nn.Sequential(
                torch.nn.Dropout(hparams['resnet_dropout']),
                torch.nn.Linear(in_features, in_features // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features // 2, in_features // 4),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features // 4, out_features))
        
    else:
        return torch.nn.Sequential(
                torch.nn.Dropout(hparams['resnet_dropout']),
                torch.nn.Linear(in_features, out_features)
                )
       
# class WholeFish(nn.Module):
#     def __init__(self, input_shape, num_classes, hparams, weights=None):
#         super(WholeFish, self).__init__()
#         featurizer = Featurizer(input_shape, hparams)
#         classifier = Classifier(
#             featurizer.n_outputs,
#             num_classes,
#             hparams['nonlinear_classifier'])
#         self.net = nn.Sequential(
#             featurizer, classifier
#         )
#         if weights is not None:
#             self.load_state_dict(copy.deepcopy(weights))

#     def reset_weights(self, weights):
#         self.load_state_dict(copy.deepcopy(weights))

#     def forward(self, x):
#         return self.net(x)
