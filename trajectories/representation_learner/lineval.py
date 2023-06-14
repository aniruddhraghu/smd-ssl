"""
fine_tune.py
Fine tunes a pre-trained model on a specific (single) task
"""

import torch
import torch.optim
from torch.autograd import set_detect_anomaly
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler

import numpy as np

import json, os, pickle, random
from copy import deepcopy
from tqdm import tqdm
import glob
import wandb

from ..utils import *
from ..constants import *
from .args import Args

from .meta_model import *
from .run_model import setup_datasets_and_dataloaders, args_run_setup, train_meta_model, evaluate_model



def get_reprs_labels(meta_model, dl, task, training=False):
    reprs, labels = [], []
    for i, batch in tqdm(enumerate(dl)):
        if batch['signal'].shape[0] == 1:
            if training:
                print("Skipping singleton batch.")
                continue
        with torch.no_grad():
            _, pooled_output, _, _ = meta_model.forward(batch)
            reprs.append(pooled_output.cpu().numpy())
            labels.append(batch[task].detach().cpu().numpy()) # This should be the label
    reprs = np.concatenate(reprs, axis=0)
    labels = np.concatenate(labels, axis=0)
    return reprs, labels
            
            
def fine_tune_model(
    fine_tune_args, meta_model_args, sample_datum, train_dataloaders_by_data_frac,
    tqdm=None, meta_model=None, tuning_dataloader=None, test_dataloader=None,
):
    print('in fine tune model')
    reloaded = (meta_model is not None)

    verbose = False
    if hasattr(fine_tune_args, 'verbose'):
        verbose = fine_tune_args.verbose

    # Decide on the task weights so that we only train/eval on the chosen task
    task_weights = {i:0.0 for i in ALL_TASKS}
    task_weights[fine_tune_args.task] = 1.0

    
    outputs = []
    for data_frac, train_dataloader in train_dataloaders_by_data_frac.items():
        wandb_exp = fine_tune_args.run_dir.split('/')[-1]
        if data_frac != 1:
            wandb_name = wandb_exp + '_lineval_' + fine_tune_args.task + f'_frac{data_frac}'
        else:
            wandb_name = wandb_exp + '_lineval_' + fine_tune_args.task

        # wandb.init(project = "traj", 
        #            entity="patient_trajectories",
        #            config = vars(fine_tune_args),
        #            name = wandb_name, 
        #            tags=["lineval", wandb_exp, fine_tune_args.task],
        #            resume=False,
        #            settings=wandb.Settings(start_method="fork"))
        wandb_state = 'lineval'
        
        
        fine_tune_dir_name = fine_tune_args.task
        if data_frac != 1: fine_tune_dir_name += f"_{str(data_frac).replace('.', '-')}"

        fine_tune_run_dir = os.path.join(fine_tune_args.run_dir, fine_tune_dir_name)
        assert os.path.isdir(fine_tune_run_dir), f"{fine_tune_run_dir} must exist!"
        
        
        if meta_model is None:
            meta_model = MetaModel(
                meta_model_args, sample_datum,
                class_names = None,
                task_weights = task_weights,
                verbose = verbose,
            )

        load_epoch = fine_tune_args.load_epoch
        if load_epoch == -1:
            load_epoch='latest'
        if not(reloaded):
            reloaded, epoch = meta_model.load(epoch=load_epoch)
            epoch=0
            reloaded=False
        
        # Get representations in a loop for train, val, and test sets
        # Train sklearn model on train, x-val on validation
        # Eval on test, save results. 
        
        meta_model.eval()
        
        train_reprs, train_labels = get_reprs_labels(meta_model, train_dataloader, fine_tune_args.task, training=True)
        val_reprs, val_labels = get_reprs_labels(meta_model, tuning_dataloader, fine_tune_args.task)
        test_reprs, test_labels = get_reprs_labels(meta_model, test_dataloader, fine_tune_args.task)
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        all_cs = [0.1, 0.5, 1, 5, 10]
        val_aucs, val_auprcs = [], []
        test_aucs, test_auprcs = [], []
        for c in all_cs:
            clf = LogisticRegression(random_state=0,C=c,max_iter=100000)
            clf.fit(train_reprs, train_labels)

            val_score = roc_auc_score(val_labels, clf.predict_proba(val_reprs)[:,1])
            test_score = roc_auc_score(test_labels, clf.predict_proba(test_reprs)[:,1])
            val_aucs.append(val_score)
            test_aucs.append(test_score)

            val_score = average_precision_score(val_labels, clf.predict_proba(val_reprs)[:,1])
            test_score = average_precision_score(test_labels, clf.predict_proba(test_reprs)[:,1])
            val_auprcs.append(val_score)
            test_auprcs.append(test_score)
        
        best_res_idx = np.argmax(val_aucs)
        
        val_auc = val_aucs[best_res_idx]
        val_auprc = val_auprcs[best_res_idx]
        test_auc = test_aucs[best_res_idx]
        test_auprc = test_auprcs[best_res_idx]
        
        # wandb.log({f'{fine_tune_args.task}_auc_val' : val_auc})
        # wandb.log({f'{fine_tune_args.task}_auprc_val' : val_auprc})
        # wandb.log({f'{fine_tune_args.task}_auc_test' : test_auc})
        # wandb.log({f'{fine_tune_args.task}_auprc_test' : test_auprc})
        
        best_c = all_cs[best_res_idx]
        clf = LogisticRegression(random_state=0,C=best_c,max_iter=100000)
        clf.fit(train_reprs, train_labels)
        
        test_preds = clf.predict_proba(test_reprs)[:,1]
        
        roc_auc_bootstrap = do_bootstrap(test_preds, test_labels, roc_auc_score)
        av_prec_bootstrap = do_bootstrap(test_preds, test_labels, average_precision_score)
        
        # store in file 
        savefile =  os.path.join(fine_tune_run_dir, 'lineval_roc_auc.npy')
        np.save(savefile, roc_auc_bootstrap)
        
        savefile =  os.path.join(fine_tune_run_dir, 'lineval_av_prec.npy')
        np.save(savefile, av_prec_bootstrap)
        
        savefile = os.path.join(fine_tune_run_dir, 'best_preds_labels.npy')
        np.save(savefile, [test_preds, test_labels])
        
        savefile = os.path.join(fine_tune_run_dir, 'all_reprs_labels.pkl')
        pickle.dump([[train_reprs, train_labels], [val_reprs, val_labels], [test_reprs, test_labels]],
                   open(savefile, 'wb'))

        
def do_bootstrap(preds,labels, func, n=1000):
    # Compute bootstraps
    res = []
    rng = np.random.RandomState(seed=0)
    for _ in range(n):
        idxs = rng.choice(len(labels), size=len(labels), replace=True)
        parr= preds[idxs]
        tarr = labels[idxs]
        score = func(tarr, parr)
        res.append(score)
    return np.array(res)
        
        
def main(fine_tune_args, tqdm):
    
    ### SEED EVERYTHING HERE ###
    random.seed(fine_tune_args.frac_fine_tune_data_seed)
    torch.manual_seed(fine_tune_args.frac_fine_tune_data_seed)
    np.random.seed(fine_tune_args.frac_fine_tune_data_seed)

    assert os.path.isdir(fine_tune_args.run_dir), "Run dir must exist!"

    fine_tune_args.to_json_file(os.path.join(fine_tune_args.run_dir, FINE_TUNE_ARGS_FILENAME))

    assert fine_tune_args.task in ALL_TASKS,\
        f"Invalid fine tune task: {fine_tune_args.task}"

    meta_model_args = Args.from_json_file(os.path.join(fine_tune_args.run_dir, ARGS_FILENAME))
    
    print('Disabling contrastive learning for fine tuning!')
    meta_model_args.do_simclr = False
    meta_model_args.do_vicreg = False

    print('Loading LR, batch size, epochs from the FT args!')
    meta_model_args.learning_rate = fine_tune_args.learning_rate
    meta_model_args.batch_size = fine_tune_args.batch_size
    meta_model_args.epochs = fine_tune_args.epochs
    meta_model_args.signal_seconds = fine_tune_args.signal_seconds

        
    datasets, train_dataloader, val_dataloader, test_dataloader = setup_datasets_and_dataloaders(meta_model_args, 
                                                                                                 task=fine_tune_args.task)

    orig_len=len(datasets['train'])

    assert datasets['train'].max_seq_len == meta_model_args.max_seq_len
    assert train_dataloader.dataset.max_seq_len == meta_model_args.max_seq_len

    sample_datum = datasets['train'][0]

    # NOTE: this could be extended to support dataloaders of different size
    train_dataloaders_by_data_frac = {1: train_dataloader}

    fine_tune_dir_name = fine_tune_args.task

    fine_tune_run_dir = os.path.join(fine_tune_args.run_dir, fine_tune_dir_name)

    if not os.path.exists(fine_tune_run_dir): os.makedirs(fine_tune_run_dir)

    for (do, suffix) in [(fine_tune_args.do_frozen_representation, "FTD"), 
                            (fine_tune_args.do_free_representation, "FTF")]:
        if not do: continue

        fine_tune_meta_model_args = deepcopy(meta_model_args)
        fine_tune_meta_model_args.run_dir = os.path.join(fine_tune_run_dir, suffix)

        args_run_setup(fine_tune_meta_model_args)

    return fine_tune_model(
        fine_tune_args, meta_model_args, sample_datum, train_dataloaders_by_data_frac,
        tqdm=tqdm, tuning_dataloader=val_dataloader, test_dataloader=test_dataloader,
    )
