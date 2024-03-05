# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import os
import copy
import numpy as np
from collections import defaultdict, OrderedDict
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from domainbed import networks
from domainbed.lib.misc import (
    random_pairs_of_minibatches, ParamDict, MovingAverage, l2_between_dicts
)


ALGORITHMS = [
    'ERM',
    "SWA",
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, init_step=False, path_for_init=None):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'], self.hparams)
        self.eval_mode_classifier_bn = False 
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.linear_prob = init_step
        # ## DiWA load shared initialization ##
        # if path_for_init is not None:
        #     if os.path.exists(path_for_init):
        #         checkpoint = torch.load(path_for_init)
        #         if 'model_dict' in checkpoint:
        #             checkpoint = checkpoint['model_dict']
        #             state_dict = {}
        #             for key in checkpoint:
        #                 if key.startswith('network.'):
        #                     state_dict[key[len('network.'):]] = checkpoint[key]
        #             checkpoint = state_dict
        #         print(hparams['skip_load_classifier'])
        #         for key in checkpoint:
        #             print(key)
        #         if 'skip_load_classifier' in hparams and hparams['skip_load_classifier']:

        #             if '1.1.weight' in checkpoint:
        #                 checkpoint.pop('1.1.weight')
        #                 checkpoint.pop('1.1.bias')
        #             elif '1.weight' in checkpoint:
        #                 checkpoint.pop('1.weight')
        #                 checkpoint.pop('1.bias')

                
        #         msg = self.network.load_state_dict(checkpoint,strict=False)
        #         print(msg)
        #         self.network.cuda()
        #         self.featurizer.cuda()
        #         self.classifier.cuda()

        #     else:
        #         assert init_step, "Your initialization has not been saved yet"



        ## DiWA choose weights to be optimized ##
        if not init_step:
            #parameters_to_be_optimized = self.network.parameters()
            #print(self.featurizer.__dict__)

            # if getattr(self.featurizer.network, "optimize_params", False):
            #     featurizer_param = self.featurizer.network.optimize_params()
            # else:
            #     featurizer_param = [param for param in self.featurizer.parameters() if param.requires_grad_]
            # #featurizer_param = self.featurizer.network.optimize_params()
            #if 'unfoldresnet' in hparams and hparams['unfoldresnet']: 
            #    featurizer_param = [param for name, param in self.featurizer.named_parameters() if '_v' in name]
            #else:
            #    featurizer_param = [param for param in self.featurizer.parameters() if param.requires_grad_]
            featurizer_param = [param for param in self.featurizer.parameters() if param.requires_grad_]

            parameters_to_be_optimized = [{'params': featurizer_param,'lr':hparams['lr']},\
            {'params':[param for param in self.classifier.parameters()],'lr':hparams['lr']*hparams['lastlrfac'] }]


            for var in parameters_to_be_optimized:
                print("nparam groups ", len(var['params']))
        else:
            
            # linear probing
            parameters_to_be_optimized = self.classifier.parameters()

            
        if self.hparams['optimizer'].lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                parameters_to_be_optimized,
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
                )
        elif self.hparams['optimizer'].lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                parameters_to_be_optimized,
                momentum=self.hparams['momentum'],
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay']
                )
        else:
            raise NotImplementedError
        
        print(self.optimizer)

        if self.hparams['scheduler'].lower() == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer = self.optimizer,
                milestones=hparams['milestones'], gamma=hparams['gamma'])
        elif self.hparams['scheduler'].lower() == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = self.optimizer,\
                T_max = hparams['steps'])
        else:
            self.scheduler = None 

    # def fix_rep_bn(self, mean, var): 
    #     #assert isinstance(self.classifier[0], nn.BatchNorm1d)
    #     self.classifier[0].running_mean = mean
    #     self.classifier[0].running_var = var
    #     self.classifier[0].eval() 
    #     self.eval_mode_classifier_bn = True 
    #     if 'weight' in self.classifier[0].__dict__:
    #         self.classifier[0].weight.data.fill_(1) 
    #         self.classifier[0].bias.data.fill_(0) 

    # def train(self, mode=True):
    #     """
    #     Override the default train() to freeze the BN parameters
    #     """
    #     super().train(mode)
    #     if self.eval_mode_classifier_bn:
    #         self.classifier[0].eval() 
    #         #print('classifier bn eval mode')
    #         #print(self.classifier[0].weight)
    #         #print(self.classifier[0].bias)

     

    # def freeze_bn(self):
    #     self.classifier[0].eval()


    def update(self, minibatches, unlabeled=None):
        
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        
        if self.linear_prob:
            with torch.no_grad():
                feat = self.featurizer(all_x)
            loss = F.cross_entropy(self.classifier(feat), all_y)
        else:
            loss = F.cross_entropy(self.predict(all_x), all_y)
        # print(self.classifier[0].training)
        # print(self.classifier[0].running_mean)
        # print(self.classifier[0].running_var)
        # print(self.classifier[0].weight)
        # print(self.classifier[0].bias)
        if 'unfoldresnet' in self.hparams and self.hparams['unfoldresnet'] and 'l1' in self.hparams:
            featurizer_param = [param for name, param in self.featurizer.named_parameters() if '_v' in name]
            l1_penalty = 0.
            for param in featurizer_param:
                l1_penalty += torch.norm(param, 1)
            loss = loss + self.hparams['l1'] * l1_penalty
            print('l1l1l1', l1_penalty)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return {'loss': loss.item()}
    
    def forward_rep(self,minibatches):    
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        
        with torch.no_grad():
            feat = self.featurizer(all_x)
        return feat

    def predict(self, x):
        return self.network(x)

    # def state_dict(self):
    #     return 
    ## DiWA for saving initialization ##
    def save_path_for_future_init(self, path_for_init):
        assert not os.path.exists(path_for_init), "The initialization has already been saved"
        torch.save(self.network.state_dict(), path_for_init)


class SWA(ERM):
    
    def __init__(self, network, ma_start_iter, *args, **kwargs):
        ERM.__init__(self, *args, **kwargs)

        self.network_ma = copy.deepcopy(network)

        self.ma_start_iter = ma_start_iter

        self.global_iter = 0
        self.ma_count = 0


    def predict(self, x):
        self.network_ma.eval()
        return self.network_ma(x)

    def update(self,network):
        
        if self.global_iter -1 >= self.ma_start_iter:
            self.ma_count += 1
            
            for param_q, param_k in zip(network.parameters(), self.network_ma.parameters()):
                param_k.data = (param_k.data * self.ma_count + param_q.data)/(1.+self.ma_count)
            
            for m1, m2 in zip(network.modules(), self.network_ma.modules()):
                if isinstance(m1, nn.BatchNorm1d):
                    m2.running_mean = (m2.running_mean * self.ma_count + m1.running_mean ) / (1. + self.ma_count)
                    m2.running_var = (m2.running_var * self.ma_count + m1.running_var ) / (1. + self.ma_count)

            if  isinstance(self.network_ma[1], nn.Sequential): 
                for i, layer in enumerate(self.network_ma[1]):
                    if isinstance(layer, nn.BatchNorm1d):
                        self.network_ma[1][i].running_mean = (self.network_ma[1][i].running_mean * self.ma_count + network[1][i].running_mean )/(1. + self.ma_count)
                        self.network_ma[1][i].running_var = (self.network_ma[1][i].running_var * self.ma_count + network[1][i].running_var )/(1. + self.ma_count)

        else:


            for param_q, param_k in zip(network.parameters(), self.network_ma.parameters()):
                param_k.data = torch.clone(param_q.data)
            
            for m1, m2 in zip(network.modules(), self.network_ma.modules()):
                if isinstance(m1, nn.BatchNorm1d):
                    m2.running_mean =  torch.clone(m1.running_mean)
                    m2.running_var = torch.clone(m1.running_var)


            if  isinstance(self.network_ma[1], nn.Sequential): 
                for i, layer in enumerate(self.network_ma[1]):
                    if isinstance(layer, nn.BatchNorm1d):
                        self.network_ma[1][i].running_mean = network[1][i].running_mean
                        self.network_ma[1][i].running_var = network[1][i].running_var

        self.global_iter += 1
