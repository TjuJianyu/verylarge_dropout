# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from domainbed.lib import misc


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    #SMALL_IMAGES = ['Debug28', 'RotatedMNIST', 'ColoredMNIST']

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.

    _hparam('data_augmentation', True, lambda r: True)
    _hparam('resnet18', False, lambda r: False)
    _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    #_hparam('class_balanced', False, lambda r: False)
    # TODO: nonlinear classifiers disabled
    _hparam('nonlinear_classifier', False,
            lambda r: bool(r.choice([False, False])))

    #optimizer 
    _hparam('optimizer', 'sgd', lambda r: 'sgd')
    _hparam('momentum', 0.9, lambda r: 0.9)
    
    # backbone dropout
    #_hparam('dropout_rate_per_group', [0,0,0,0],lambda r: [0,0,0,0])
    #_hparam('dropout_conv_only', False,lambda r: False)
    #_hparam('dropout_type', '2d',lambda r: '2d') # 2d or block 
    #_hparam('dropout_kernel', 7,lambda r: 7)
    _hparam('ma_start_iter', 300,lambda r: 300)
    
    #step lr scheduler
    _hparam('milestones', [5000,10000],lambda r: [5000,10000])
    _hparam('gamma', 0.1,lambda r: 0.1)
    _hparam('scheduler', 'step', lambda r: 'step')
    _hparam('stodepth', 1., lambda r: 1.)
    _hparam('stodepth_uniform', 1, lambda r: 1)
    
    _hparam('freeze_bn', True, lambda r: True)
    _hparam('lastlrfac', 1., lambda r: 1.)
    _hparam('pretrain_version', 'v2', lambda r: 'v2')
    _hparam('skip_load_classifier', 0, lambda r: 0)
    
    _hparam('lr', 1e-3, lambda r: r.choice([5e-4, 1e-3]))
    
    _hparam('weight_decay', 1e-4, lambda r: r.choice([1e-4, 5e-5, 1e-5]))
    
    _hparam('batch_size', 32, lambda r: 32)

    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}
