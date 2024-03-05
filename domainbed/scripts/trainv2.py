# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


# The order of import numpy and import torch has a huge impact of dataloader speed 
# import torch first
# https://github.com/pytorch/pytorch/issues/101188
import torch
import torchvision
import torch.utils.data


import argparse
import collections
import json
import os
import random
import sys
import time
import uuid


import PIL


from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader
from collections import defaultdict
import copy

import time
from torchvision import transforms
from torchvision.datasets import  ImageFolder
import torch


import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="VLCS")
    parser.add_argument('--algorithm', type=str, default="ERM")
    #parser.add_argument('--task', type=str, default="domain_generalization",
    #    choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
       help="For domain adaptation, \% of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    ## DiWA ##
    parser.add_argument('--init_step', action='store_true')
    parser.add_argument('--path_for_init', type=str, default=None)
    parser.add_argument('--optimizer', type=str, default='Adam') #[Adam, sgd]
    parser.add_argument('--momentum', type=float, default=0.9) # momentum in sgd, beta1 in adam
    parser.add_argument('--beta2', type=float, default=0.99) # beta2 in adam
    parser.add_argument('--save_logits', type=int, default=0) 
    parser.add_argument('--outsplit_noaug', type=int, default=0) 
    #parser.add_argument('--init_clf_std',  action='store_true')
    args = parser.parse_args()
        

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    #If we ever want to implement checkpointing, just persist these values
    #every once in a while, and then load them from disk here.
    resume = os.path.join(args.output_dir,'model.pkl')

    if os.path.exists(resume) and os.path.getsize(resume) > 1024: # bytes
        print(f"resume from {resume}")
        try:
            checkpoint = torch.load(resume)
            algorithm_dict = checkpoint['model_dict']
            swa_algorithm_dict = checkpoint['swa_model_dict']
            optimizer_dict = checkpoint['optimizer_dict']
            scheduler_dict = checkpoint['scheduler_dict']
            start_step = checkpoint['step']
            best_score = checkpoint['best_score']
            swa_best_score = checkpoint['swa_best_score']
            last_results_keys = checkpoint['last_results_keys']
            ensemble_logits = checkpoint['ensemble_logits']
        except Exception as error:
            print("error during loading checkpoint...")
            print(error)
            best_score = -float("inf")
            swa_best_score = -float("inf")
            last_results_keys = None
            start_step = 0
            algorithm_dict = None
            swa_algorithm_dict = None
            optimizer_dict = None 
            scheduler_dict = None 
            ensemble_logits = {}

    else:
        best_score = -float("inf")
        swa_best_score = -float("inf")
        last_results_keys = None
        start_step = 0
        algorithm_dict = None
        swa_algorithm_dict = None
        optimizer_dict = None 
        scheduler_dict = None 
        ensemble_logits = {}

        

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))
    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                int(len(in_)*args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))

        in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        if args.outsplit_noaug:
            print('outsplit do not contain augmentation')
            out = copy.deepcopy(out)
            out.underlying_dataset.transform = dataset.noaugment_transform
        

        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))


    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]


    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=128,
        num_workers=dataset.N_WORKERS, shuffle=False)
        for env, _ in (in_splits + out_splits + uda_splits)]


    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
        for i in range(len(uda_splits))]

    
    n_steps = args.steps or dataset.N_STEPS
    hparams['steps'] = n_steps
    #if args.algorithm in ["ERM", "MA", "RSC", "ERM_wdinit","DiverseClf"]:
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams,
        init_step=args.init_step,
        path_for_init=args.path_for_init)


    algorithm_class = algorithms.get_algorithm_class("SWA")
    
    swa_algorithm = algorithm_class(algorithm.network,hparams['ma_start_iter'],  \
        dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams,
        init_step=args.init_step,
        path_for_init=args.path_for_init,)

    # else: ## todo eliminate else...
    #     algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    #     algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
    #         len(dataset) - len(args.test_envs), hparams, 
    #         init_step=args.init_step,
    #         path_for_init=args.path_for_init)

    #     algorithm_class = algorithms.get_algorithm_class("SWA")
        
    #     swa_algorithm = algorithm_class(algorithm.network,hparams['ma_start_iter'],  \
    #         dataset.input_shape, dataset.num_classes,
    #         len(dataset) - len(args.test_envs), hparams,
    #         init_step=args.init_step,
    #         path_for_init=args.path_for_init,)

    if algorithm_dict is not None and swa_algorithm_dict is not None:
        print('load model dict ')
        algorithm.load_state_dict(algorithm_dict)
        algorithm.network.cuda()

        print('load swa (weight average) model dict')
        swa_algorithm.load_state_dict(swa_algorithm_dict)
        swa_algorithm.network.cuda()

    if optimizer_dict is not None:
        print('load optimizer ')
        algorithm.optimizer.load_state_dict(optimizer_dict)
        #algorithm.optimizer.to(device)
    if scheduler_dict is not None:
        print('load schueduler')
        algorithm.scheduler.load_state_dict(scheduler_dict)
        #algorithm.scheduler.to(device)

    algorithm.to(device)
    swa_algorithm.to(device)
    print(algorithm.optimizer)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def estimate_rep_mean_var(model, train_minibatches_iterator, steps=1000):
        model.eval()
        sum_x, sum_x_square, count = 0, 0, 0
        mean, var = 0, 1
        with torch.no_grad():
            for step in range(steps):
                minibatches_device = [(x.to(device,non_blocking=True), y.to(device, non_blocking=True))
                    for x,y in next(train_minibatches_iterator)]
                feat = algorithm.forward_rep(minibatches_device)
                sum_x += feat.sum(axis=0)
                sum_x_square += (feat ** 2).sum(axis=0)
                count += len(feat)
                cur_mean = sum_x / count 
                cur_var = sum_x_square / count - cur_mean**2 
                
                mean = cur_mean
                var = cur_var
        model.train() 
        print('estimate rep mean var error:')
        print((cur_mean - mean).abs().mean().item(),(cur_mean - mean).abs().max().item(),(cur_var - var).abs().mean().item(), (cur_var - var).abs().max().item() )
        return mean, var 



    def save_checkpoint(filename, step, best_score, last_results_keys, results=None):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict(),
            "swa_model_dict": swa_algorithm.state_dict(),
            "ensemble_logits": ensemble_logits,
            "optimizer_dict": algorithm.optimizer.state_dict(),
            "scheduler_dict": algorithm.scheduler.state_dict() if algorithm.scheduler is not None else None,
            'step':step,
            'best_score':best_score,
            'swa_best_score': swa_best_score,
            'last_results_keys':last_results_keys,
        }
        ## DiWA ##
        if results is not None:
            save_dict["results"] = results
        torch.save(save_dict, os.path.join(args.output_dir, filename))
    
    ensemble_start_iter = hparams['ma_start_iter']
    # if args.init_clf_std:
    #     print('estimating representation mean and variance...')
    #     mean, var = estimate_rep_mean_var(algorithm, train_minibatches_iterator, steps= 100)
    #     print(mean, var)
    #     algorithm.fix_rep_bn(mean,var)

    if args.save_logits:

        # minibatches_device = [(x.to(device), y.to(device))
        #     for x,y in next(train_minibatches_iterator)]

        evals = zip(eval_loader_names, eval_loaders, eval_weights)
        for name, loader, weights in evals:
            acc, logits, all_y = misc.accuracy_withlogits(algorithm, loader, weights, device)
            logits.dump(os.path.join(args.output_dir,f'logits_env{name}.npy'))
            all_y.dump(os.path.join(args.output_dir,f'y_env{name}.npy'))
            print(name,acc)
        # evals = zip(eval_loader_names, eval_loaders, eval_weights)
        # for name, loader, weights in evals:
        #     acc, logits, all_y = misc.accuracy_withlogits(algorithm, loader, weights, device)
        #     logits.dump(os.path.join(args.output_dir,f'logits_env{name}.npy'))
        #     all_y.dump(os.path.join(args.output_dir,f'y_env{name}.npy'))
        #     print(name,acc)
    else:
        
        datat, updatetime, evalt = 0,0,0
        t0 = time.time()
        for step in range(start_step, n_steps):
            
            checkpoint_vals = collections.defaultdict(lambda: [])
            step_start_time = time.time()

            minibatches_device = [(x.to(device,non_blocking=True), y.to(device, non_blocking=True))
                for x,y in next(train_minibatches_iterator)]
            
            t1 = time.time()
            datat += (t1 - t0)
            #print('dt',t1-t0)
            # if args.task == "domain_adaptation":
            #     uda_device = [x.to(device)
            #         for x,_ in next(uda_minibatches_iterator)]
            # else:
            #     uda_device = None
            uda_device = None
            
            step_vals = algorithm.update(minibatches_device, uda_device) 
            swa_algorithm.update(algorithm.network)
            t2 = time.time()
            updatetime += (t2 - t1)
            #print('ut',t2-t1)
            
            checkpoint_vals['step_time'].append(time.time() - step_start_time)
           

            for key, val in step_vals.items():
                checkpoint_vals[key].append(val)


            if (step % checkpoint_freq == 0) or (step == n_steps - 1):
                #print([algorithm.optimizer.param_groups[idx]['lr'] for idx in range(len(algorithm.optimizer.param_groups))])
            
                results = {
                    'step': step,
                    'epoch': step / steps_per_epoch,
                }

                for key, val in checkpoint_vals.items():
                    results[key] = np.mean(val)

                evals = zip(eval_loader_names, eval_loaders, eval_weights)

                ensemble_results = copy.deepcopy(results)
                swa_results = copy.deepcopy(results)
                
                for name, loader, weights in evals:
                    envidx = int(name.split('_')[0][len('env'):])
                    if envidx not in args.test_envs and name.split('_')[1] != 'out':
                        results[name+'_acc'] = -1
                        ensemble_results[name + '_acc'] = -1
                        continue

                    acc, logits, all_y = misc.accuracy_withlogits(algorithm, loader, weights, device)
                
                    if step >= ensemble_start_iter:
                        if name not in ensemble_logits:
                            
                            ensemble_logits[name] = logits 
                        else:
                            ensemble_logits[name] += logits

                        ensemble_results[name + '_acc'] = (torch.argmax(ensemble_logits[name],axis=1) == all_y).float().mean().item()
                    else:
                        ensemble_results[name + '_acc'] = acc 
                    
                    results[name+'_acc'] = acc


                if step >= ensemble_start_iter:
                    evals = zip(eval_loader_names, eval_loaders, eval_weights)
                    for name, loader, weights in evals:
                        envidx = int(name.split('_')[0][len('env'):])
                        if envidx not in args.test_envs and name.split('_')[1] != 'out':
                            swa_results[name+'_acc'] = -1
                            continue
                        acc = misc.accuracy(swa_algorithm, loader, weights, device)
                        swa_results[name+'_acc'] = acc
                else:
                    swa_results = copy.deepcopy(results)

               

                evalt += time.time() - t2
                print( datat, updatetime, evalt)
                #print(ensemble_logits)
                datat, updatetime, evalt = 0,0,0
                results['mem_gb'] = ensemble_results['mem_gb'] = swa_results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)


                results_keys = sorted(results.keys())
                if results_keys != last_results_keys:
                    misc.print_row(results_keys, colwidth=12)
                    last_results_keys = results_keys

                misc.print_row([results[key] for key in results_keys], colwidth=12)
                misc.print_row([ensemble_results[key] for key in results_keys], colwidth=12)
                misc.print_row([swa_results[key] for key in results_keys], colwidth=12)
                
                results.update({
                    'hparams': hparams,
                    'args': vars(args)
                })
                ensemble_results.update({
                    'hparams': hparams,
                    'args': vars(args)
                })
                swa_results.update({
                    'hparams': hparams,
                    'args': vars(args)
                })
                
                epochs_path = os.path.join(args.output_dir, 'results.jsonl')
                ensemble_epochs_path = os.path.join(args.output_dir, f'ensemble{ensemble_start_iter}_results.jsonl')
                swa_epochs_path = os.path.join(args.output_dir, f'swa{ensemble_start_iter}_results.jsonl')
                
                with open(epochs_path, 'a') as f:
                    f.write(json.dumps(results, sort_keys=True) + "\n")
                with open(ensemble_epochs_path, 'a') as f:
                    f.write(json.dumps(ensemble_results, sort_keys=True) + "\n")
                with open(swa_epochs_path, 'a') as f:
                    f.write(json.dumps(swa_results, sort_keys=True) + "\n")
                
                start_step = step + 1
                
                current_score = misc.get_score(results, args.test_envs)
                swa_current_score = misc.get_score(results, args.test_envs)

                if current_score > best_score:
                    best_score = current_score
                    print(f"Saving new best score at step: {step} at path: model_best.pkl")
                    save_checkpoint('model_best.pkl', start_step, best_score, last_results_keys)
                # if swa_current_score > swa_best_score: 
                #     swa_best_score = swa_current_score
                #     print(f"Saving swa new best score at step: {step} at path: swa_model_best.pkl")
                #     save_checkpoint('swa_model_best.pkl', start_step, best_score, last_results_keys)

                #save_checkpoint('model.pkl', start_step, best_score, last_results_keys)
                if args.save_model_every_checkpoint:
                    save_checkpoint(f'model_step{step}.pkl')

                save_checkpoint('model.pkl', start_step, best_score, last_results_keys)
            
            t0 = time.time()

        # ## DiWA ##
        # if args.init_step:
        #    algorithm.save_path_for_future_init(args.path_for_init)
        
        with open(os.path.join(args.output_dir, 'done'), 'w') as f:
            f.write('done')
