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
from .run_model import setup_datasets_and_dataloaders, train_meta_model, evaluate_model


def fine_tune_model(
    fine_tune_args, meta_model_args, sample_datum, train_dataloaders_by_data_frac,
    tqdm=None, meta_model=None, tuning_dataloader=None, test_dataloader=None,
):
    print('Finetuning model...')
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
            wandb_name = wandb_exp + '_FT_' + fine_tune_args.task + f'_frac{data_frac}'
        else:
            wandb_name = wandb_exp + '_FT_' + fine_tune_args.task

        # wandb.init(project = "traj", 
        #            entity="trajectories",
        #            config = vars(fine_tune_args),
        #            name = wandb_name, 
        #            tags=["FT", wandb_exp, fine_tune_args.task],
        #            resume=False,
        #            settings=wandb.Settings(start_method="fork"))
        wandb_state = 'FT'
        
        
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

        if fine_tune_args.do_frozen_representation:
            meta_model_FTD = meta_model
            meta_model_FTD_args = deepcopy(meta_model_args)

            meta_model_FTD.run_dir = os.path.join(fine_tune_run_dir, "FTD")
            meta_model_FTD.freeze_representation()
            meta_model_FTD_args.run_dir = meta_model_FTD.run_dir

            if not os.path.isdir(meta_model_FTD.run_dir): os.makedirs(meta_model_FTD.run_dir)

            # Train it from scractch with the representation frozen and task_weights appropriately set.
            best_model = train_meta_model(
                meta_model_FTD, train_dataloader, meta_model_FTD_args, reloaded=reloaded, epoch=epoch,
                tuning_dataloader=tuning_dataloader,
                train_embedding_after=fine_tune_args.train_embedding_after, wandb=wandb, wandb_state=wandb_state
            )
            outputs.append(meta_model_FTD)
            eval_results = evaluate_model(best_model, tuning_dataloader, test_dataloader)
            save_path = os.path.join(meta_model_FTD_args.run_dir, 'eval_metrics.pt')
            meta_model = None
        elif fine_tune_args.do_free_representation:
            meta_model_FTF = meta_model
            meta_model_FTF_args = deepcopy(meta_model_args)

            meta_model_FTF.run_dir = os.path.join(fine_tune_run_dir, "FTF")
            meta_model_FTF_args.run_dir = meta_model_FTF.run_dir

            if not os.path.isdir(meta_model_FTF.run_dir): os.makedirs(meta_model_FTF.run_dir)

            # Train it from scratch with the representation frozen and task_weights appropriately set.
            best_model = train_meta_model(
                meta_model_FTF, train_dataloader, meta_model_FTF_args, reloaded=reloaded, epoch=epoch,
                tuning_dataloader=tuning_dataloader,
                train_embedding_after=fine_tune_args.train_embedding_after, wandb=wandb, wandb_state=wandb_state
            )
            outputs.append(meta_model_FTF)
            eval_results = evaluate_model(best_model, tuning_dataloader, test_dataloader)
            save_path = os.path.join(meta_model_FTF_args.run_dir, 'eval_metrics.pt')
            meta_model = None
    
        torch.save(eval_results, save_path)
        # for k,v in eval_results.items(): 
        #     for metric_name, metric_value in v.items(): 
        #         wandb.log({metric_name+'_'+k : metric_value}) # metric is a single number   
    return outputs

def main(fine_tune_args, tqdm):
    
    ### SEED EVERYTHING HERE ###
    random.seed(fine_tune_args.frac_fine_tune_data_seed)
    torch.manual_seed(fine_tune_args.frac_fine_tune_data_seed)
    np.random.seed(fine_tune_args.frac_fine_tune_data_seed)

    assert os.path.isdir(fine_tune_args.run_dir), "Run dir must exist!"
    assert (
        fine_tune_args.do_frozen_representation or
        fine_tune_args.do_free_representation
    ), "Need to do either FTF or FTD!"

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

    assert datasets['train'].max_seq_len == meta_model_args.max_seq_len
    assert train_dataloader.dataset.max_seq_len == meta_model_args.max_seq_len

    sample_datum = datasets['train'][0]

    # NOTE: this could be extended to support dataloaders of different size
    train_dataloaders_by_data_frac = {1: train_dataloader}

    fine_tune_dir_name = fine_tune_args.task

    fine_tune_run_dir = os.path.join(fine_tune_args.run_dir, fine_tune_dir_name)

    if not os.path.exists(fine_tune_run_dir): os.makedirs(fine_tune_run_dir)

    data_frac_seed = random.randint(0, int(1e10))
    with open(os.path.join(fine_tune_run_dir, 'data_frac_seed.txt'), mode='w') as f:
        f.write(str(data_frac_seed))

    for (do, suffix) in [(fine_tune_args.do_frozen_representation, "FTD"), 
                            (fine_tune_args.do_free_representation, "FTF")]:
        if not do: 
            continue

        fine_tune_meta_model_args = deepcopy(meta_model_args)
        fine_tune_meta_model_args.run_dir = os.path.join(fine_tune_run_dir, suffix)

        if not os.path.exists(fine_tune_meta_model_args.run_dir): 
            os.mkdir(os.path.abspath(fine_tune_meta_model_args.run_dir))

        fine_tune_meta_model_args.to_json_file(os.path.join(fine_tune_meta_model_args.run_dir, ARGS_FILENAME))


    return fine_tune_model(
        fine_tune_args, meta_model_args, sample_datum, train_dataloaders_by_data_frac,
        tqdm=tqdm, tuning_dataloader=val_dataloader, test_dataloader=test_dataloader,
    )
