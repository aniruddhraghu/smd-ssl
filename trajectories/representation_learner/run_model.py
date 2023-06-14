"""
run_model.py
"""

import copy
import random
import numpy as np
import time
from collections import defaultdict
import torch.optim
from torch.autograd import set_detect_anomaly
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler 

import json, os, pickle
import pandas as pd
from tqdm import tqdm
import wandb

from sklearn.metrics import roc_auc_score, average_precision_score

from ..utils import *
from ..constants import *
from ..representation_learner.meta_model import *
from .args import Args

from .example_dataset import PatientDataset


def train_meta_model(
    meta_model, train_dataloader, args, reloaded=False, epoch=0,
    tuning_dataloader=None, train_embedding_after=-1, tqdm=tqdm, 
    just_gen_data=False, wandb=None, wandb_state=None
):
    if wandb_state == 'FT':
        pass
    else:
        wandb_state = 'pretrain'
        wandb_exp = args.run_dir.split('/')[-1]
        wandb_name = wandb_exp + '_pretrain'
        # wandb.init(project = "traj", 
        #            entity="trajectories",
        #            config = vars(args),
        #            name = wandb_name, 
        #            tags=["pretrain", wandb_exp],
        #            resume=False,
        #            settings=wandb.Settings(start_method="fork"))
    wandb_counters = defaultdict(int)
    
    if just_gen_data:
        optimizer, scheduler = None, None
    else:
        optimizer = torch.optim.Adam(
            meta_model.parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay if args.do_weight_decay else 0,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, args.learning_rate_step,
            args.learning_rate_decay if args.do_learning_rate_decay else 1,
        )

    early_stop_count=0
    prev_err=10e9
    best_auc = 0.
    best_sd = copy.deepcopy(meta_model.model.state_dict())

    epoch_rng = range(epoch+1 if reloaded else 0, args.epochs)
    if tqdm is not None: epoch_rng = tqdm(epoch_rng, desc='Epoch: N/A', leave=False)

    train_ld = {}
    val_ld = {}

    for epoch in epoch_rng:
        if args.do_simclr or args.do_vicreg:
            if epoch < args.component_only_epochs:
                meta_model.model.component_weight = 1.0
                meta_model.model.global_weight = 0.0
            else:
                meta_model.model.component_weight = args.component_weight
                meta_model.model.global_weight = args.global_weight
        
        if not just_gen_data:
            scheduler.step()

            if train_embedding_after >= epoch:
                # to ensure it is unfrozen after reloading
                meta_model.unfreeze_representation()

            meta_model.train()
            optimizer.zero_grad()

        dataloader_rng = train_dataloader
        if tqdm is not None:
            dataloader_rng = tqdm(
                dataloader_rng, desc='Batch: N/A', total=len(train_dataloader), leave=False)


        ################################################################
        #  Run a loop on the training dataset. 
        #  NOTE: wandb logging commented out
        ################################################################

        for i, batch in enumerate(dataloader_rng):
            if just_gen_data:
                total_loss = torch.tensor(0)
                continue
            if args.do_detect_anomaly: set_detect_anomaly(True)

            if batch['signals_timeseries'].shape[0] == 1:
                print("Skipping singleton batch.")
                continue

            _, _, all_outputs, total_loss = meta_model.forward(batch)

            try:
                total_loss.backward()
            except:
                print(total_loss.shape, total_loss)
                raise

            if i % args.batches_per_gradient == 0:
                optimizer.step()
                optimizer.zero_grad()
            if args.do_detect_anomaly: 
                set_detect_anomaly(False)

            if tqdm is not None: 
                dataloader_rng.set_description('Batch: %.2e' % total_loss)
            train_ld = update_lossdict(train_ld, all_outputs)
            all_losses = {'train_ld' : train_ld, 
                          'val_ld' :   val_ld,}

            save_path = os.path.join(args.run_dir, 'all_losses.pt')
            torch.save(all_losses, save_path)
            
            if wandb_state == 'pretrain':
                if args.do_vicreg:
                    pass
                    # for key in ['sim_loss', 'cov_loss', 'var_loss']:
                    #     for prefix in ['traj', 'elem']:
                    #         full_key = f'{prefix}_{key}'
                            # wandb.log({full_key: all_outputs[full_key],
                            #           'train_step': wandb_counters[wandb_state]})
            else: 
                # all outputs contains task_logits, dfs, task_losses
                for k in all_outputs.keys(): 
                    pass
                    # wandb.log({'train_'+k+'_loss' : all_outputs[k][2],
                    #            'train_step' : wandb_counters[wandb_state]})       
            wandb_counters[wandb_state] += 1
        
        if just_gen_data: continue

        if tqdm is None: 
            print("Epoch %d: %.2f" % (epoch, total_loss.item()))
        elif (tqdm is not None) and (tuning_dataloader is not None): 
            pass
        else: 
            epoch_rng.set_description("Epoch %d: %.2f" % (epoch, total_loss.item()))

        if args.do_simclr or args.do_vicreg:
            if (epoch+1) % args.train_save_every == 0:
                meta_model.save(epoch+1)



        ################################################################
        #  Do eval on validation set to see if this is the best model
        ################################################################

        tuning_dataloader.dataset.epoch=epoch
        tuning_dataloader.dataset.save_place=args.dataset_dir

        dataloader_rng = tuning_dataloader
        if tqdm is not None:
            dataloader_rng = tqdm(
                dataloader_rng, desc='Batch: N/A', total=len(tuning_dataloader), leave=False)

        meta_model.eval()
    
        tuning_losses=[]
        val_total_loss, val_traj_loss, val_elem_loss = [], [], []
    
        val_vicreg = defaultdict(list)
        ft_loss = defaultdict(list)

        CUR_TASK = None
        
        auc_res = {}
        for i, batch in enumerate(dataloader_rng):
            with torch.no_grad():
                hidden_states, pooled_output, all_outputs, total_loss = meta_model.forward(batch)
            ### all_outputs contains a tuple of (logits, labels, losses) for each task 
            ### In the saved result, extract only the losses for logging purposes.
            if args.do_simclr or args.do_vicreg:
                rel_outputs = all_outputs
            else:
                rel_outputs = {task_name:task_outputs[-1] for task_name,task_outputs in all_outputs.items()}
                CUR_TASK = list(all_outputs.keys())[0]
                proc_output = {}
                logit,label,loss = list(all_outputs.values())[0]
                proc_output[f'{CUR_TASK}_logit'] = logit
                proc_output[f'{CUR_TASK}_label'] = label
                auc_res = update_lossdict(auc_res, proc_output)

            val_ld = update_lossdict(val_ld, rel_outputs)
            all_losses = {
                'train_ld' : train_ld,
                'val_ld' :   val_ld,
                }
            save_path = os.path.join(args.run_dir, 'all_losses.pt')
            torch.save(all_losses, save_path)
            tuning_losses.append(total_loss.cpu().data.numpy().ravel())
            
            if wandb_state == 'pretrain':
                if args.do_vicreg:
                    for key in ['sim_loss', 'cov_loss', 'var_loss']:
                        for prefix in ['traj', 'elem']:
                            full_key = f'{prefix}_{key}'
                            val_vicreg[full_key].append(all_outputs[full_key])
            else:                
                # all outputs contains task_logits, dfs, task_losses
                for k in all_outputs.keys(): 
                    ft_loss[k].append( all_outputs[k][2].item() ) 
        
        
        if args.do_simclr or args.do_vicreg:
            pass
            # wandb.log({'val_total_loss': np.mean(val_total_loss), 
            #            'val_traj_loss': np.mean(val_traj_loss)  ,
            #            'val_elem_loss': np.mean(val_elem_loss)  ,
            #            'val_step': epoch})
            # if args.do_vicreg:
            #     for key in ['sim_loss', 'cov_loss', 'var_loss']:
            #         for prefix in ['traj', 'elem']:
            #             full_key = f'{prefix}_{key}'
                        # wandb.log({f'val_{full_key}': np.mean(val_vicreg[full_key]),
                        #           'val_step': epoch})
        else: 
            preds = np.concatenate(auc_res[f'{CUR_TASK}_logit'])[:,1]
            labels = np.concatenate(auc_res[f'{CUR_TASK}_label'])
            cur_auc = roc_auc_score(labels, preds)
            # for k in ft_loss.keys(): 
        #         wandb.log({'val_'+k+'_loss' : np.mean(ft_loss[k]) ,
        #                        'val_step': epoch})
        #     wandb.log({'val_'+CUR_TASK+'_auc' : cur_auc ,
        #                        'val_step': epoch})
        # wandb.log({'epoch':epoch})
        
        meta_model.train()
        total_err = np.mean(np.concatenate(tuning_losses))

        if args.do_simclr or args.do_vicreg:
            cond = total_err < prev_err
        else:
            cond = cur_auc > best_auc
        
        if cond:
            # This is the best model
            if not (args.do_simclr or args.do_vicreg):
                meta_model.save(epoch+1, extra_savename='best_model_')
            else:
                meta_model.save(0, extra_savename='best_model_')
            best_sd = copy.deepcopy(meta_model.model.state_dict())
            if args.do_simclr or args.do_vicreg:
                prev_err=total_err
            else:
                best_auc = cur_auc
            early_stop_count=0
            if tqdm is None: 
                print("Epoch %d: %.2f" % (epoch, total_loss.item()))
            else: 
                epoch_rng.set_description("Epoch %d: %.2f" % (epoch, total_loss.item()))
        else:
            early_stop_count+=1
            if early_stop_count==100:
                print(f"Early stopping at epoch {epoch}. Best model at epoch {epoch} with a loss of {prev_err}")
                break

    ################################################################
    #  Finished training: save the model at the last epoch 
    # and then return the model for potential evaluation.
    ################################################################

    meta_model.save(epoch)
    if tuning_dataloader is None:
        meta_model.save(epoch)
        return meta_model
    else:
        # reload the best state dict
        meta_model.model.load_state_dict(best_sd)
        return meta_model


def do_eval(model, dl):
    model.eval()
    all_res = {}
    eval_tasks = []
    for i, batch in enumerate(dl):
        with torch.no_grad():
            _, _, all_outputs, _ = model.forward(batch)
        proc_output = {}
        for task,(logit, label, loss) in all_outputs.items():
            if i == 0:
                eval_tasks.append(task)
            proc_output[f'{task}_logit'] = logit
            proc_output[f'{task}_label'] = label
            proc_output[f'{task}_loss'] = loss
        all_res = update_lossdict(all_res, proc_output)
    
    # Now each element of all_res is either a list of floats or a list of np arrs
    # Compute some key metrics
    if 'example_task' in eval_tasks:
        preds = np.concatenate(all_res['example_task_logit'])[:,1]
        labels = np.concatenate(all_res['example_task_label'])
        all_res['example_task_auc'] = roc_auc_score(labels, preds)
        all_res['example_task_auprc'] = average_precision_score(labels, preds)   
    return all_res


def evaluate_model(model, val_dl, test_dl):
    model.eval()
    val_res = do_eval(model, val_dl)
    test_res = do_eval(model, test_dl)
    return {'val': val_res, 'test': test_res}


def run_model(
    args, datasets, train_dataloader, tqdm=None, meta_model=None, tuning_dataloader=None):

    if meta_model is None: meta_model = MetaModel(
        args, datasets['train'][0],
        class_names = {}
    )
    reloaded, epoch = meta_model.load()
    if reloaded: 
        print("Resuming from epoch %d" % (epoch+1))

    if args.do_train:
        print('training')
        train_meta_model(
            meta_model, train_dataloader, args, reloaded, epoch, tuning_dataloader, tqdm=tqdm)

    return meta_model

def load_datasets(args, task='ssl'):
    do_splits_dict = {}
    if type(args) is Args:
        if task == 'ssl':
            do_splits_dict['train']  = args.do_train or args.do_eval_train
            do_splits_dict['tuning'] = args.do_eval_tuning
        # If we're doing finetuning, load all the datasets
        else:
            do_splits_dict['train']  = True
            do_splits_dict['tuning'] = True
            do_splits_dict['test'] = True
        max_seq_len = args.max_seq_len
    else: raise AssertionError(f"Args must be of a recognized type! Is {type(args)}.")

    datasets = {}
    for split, do in do_splits_dict.items():
        if not do:
            datasets[split] = None
            continue

        datasets[split] = PatientDataset(task=task, 
                                         signal_seconds=args.signal_seconds,
                                         signal_mask=args.signal_mask,
                                         history_cutout_prob=args.history_cutout_prob,
                                         history_cutout_frac=args.history_cutout_frac,
                                         spatial_dropout_rate=args.spatial_dropout_rate, 
                                         corrupt_rate=args.corrupt_rate,
                                        )
        datasets[split].train_tune_test = split
        if split == 'train':
            datasets[split].max_seq_len = max_seq_len
    return datasets


def setup_datasets_and_dataloaders(args, task='ssl'):
    datasets = load_datasets(args, task=task)

    if not args.do_train: 
        return datasets

    sampler = RandomSampler(datasets['train'])

    train_dataloader = DataLoader(
        datasets['train'], sampler=sampler, batch_size=args.batch_size,
        num_workers=args.num_dataloader_workers, pin_memory=True)
    if task == 'ssl':
        val_dataloader = None
        if args.do_eval_tuning:
            val_dataloader = DataLoader(
                datasets['tuning'], batch_size=args.batch_size,
                num_workers=args.num_dataloader_workers, pin_memory=True)
        return datasets, train_dataloader, val_dataloader, None
    else:
        # If we're in a FT run, return dataloaders for val and test too.
        val_dataloader = DataLoader(
            datasets['tuning'], batch_size=args.batch_size,
            num_workers=args.num_dataloader_workers, pin_memory=True)
        test_dataloader = DataLoader(
            datasets['test'], batch_size=args.batch_size,
            num_workers=args.num_dataloader_workers, pin_memory=True)
        return datasets, train_dataloader, val_dataloader, test_dataloader


def setup_for_run(args):
    # Make run_dir if it doesn't exist.
    if not os.path.exists(args.run_dir): 
        os.mkdir(os.path.abspath(args.run_dir))
    elif not args.do_overwrite:
        raise ValueError("Save dir %s exists and overwrite is not enabled!" % args.run_dir)

    args.to_json_file(os.path.join(args.run_dir, ARGS_FILENAME))

    return setup_datasets_and_dataloaders(args)

def main(args, tqdm):
    ### SEED EVERYTHING HERE ###
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    datasets, train_dataloader, tuning_dataloader, _ = setup_for_run(args)

    # added to restrict the data in the dataset
    if hasattr(args, 'frac_data'):
        if args.frac_data != 1:
            raise NotImplementedError('Yet to implement dataset splitting')
    return run_model(args, datasets, train_dataloader, tqdm=tqdm, tuning_dataloader=tuning_dataloader)
