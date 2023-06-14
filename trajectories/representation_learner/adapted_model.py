from collections import defaultdict

import torch, torch.optim, torch.nn as nn, torch.nn.functional as F, torch.nn.init as init
from torch.autograd import Variable, set_detect_anomaly
from torch.utils.data import (
    DataLoader, Dataset, RandomSampler, SubsetRandomSampler, Subset, SequentialSampler
)
from torch.utils.data.distributed import DistributedSampler

from math import floor

from ..utils import *
from ..constants import *

from .nt_xent import NTXentLoss
from .vicreg_loss import vicreg_loss_func

from .cnn_enc import resnet18

from copy import deepcopy


def get_task_losses(task_class_weights):
    task_losses = {}
    for t in task_class_weights.keys():
        if t == 'pap_estimation_fg':
            task_losses[t] = nn.MSELoss()
        else:
            task_losses[t] = nn.CrossEntropyLoss()
    return nn.ModuleDict(task_losses)

POOLING_METHODS = ('max', 'avg', 'last')
class CNNGRUModel(nn.Module):
    def __init__(
        self, data_shape=[5, 1, 2400], tabular_feats=4, static_feats=6, use_cuda=False, n_gpu = 0,
        task_weights = None, hidden_dim=512, num_layers=2, bidirectional=False,
        pooling_method = 'last', fc_layer_sizes = [], verbose=False,
        expander_fcs = [], only_ecg = False, only_tabular=False,
        args=None
    ):
        super().__init__()

        self.verbose=verbose

        assert pooling_method in POOLING_METHODS, "Don't know how to do %s pooling" % pooling_method

        self.task_weights = task_weights

        # initialise the model and the weights
        self.cnn_dim = hidden_dim
        self.tab_enc_dim = args.tab_enc_dim
        if args.only_ecg:
            self.hidden_dim = self.cnn_dim 
        elif args.only_tabular:
            self.hidden_dim = self.tab_enc_dim + self.tab_enc_dim # Right now, project structured data into same dim space
        else:
            self.hidden_dim = self.cnn_dim + self.tab_enc_dim + self.tab_enc_dim

        self.num_layers = num_layers
        self.bidirectional = bidirectional
        try:
            self.traj_dropout = args.traj_dropout
        except AttributeError:
            self.traj_dropout = 0
        
        self.gru = nn.GRU(
            input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional,
        )

        self.enc = resnet18(num_outputs=self.cnn_dim, input_channels=data_shape[1])
        
        self.tabular_enc = nn.Sequential(
              nn.Linear(tabular_feats, 64),
              nn.BatchNorm1d(64),
              nn.ReLU(),
              nn.Linear(64, self.tab_enc_dim),
              nn.BatchNorm1d(self.tab_enc_dim),
              nn.ReLU())

        self.labs_enc = nn.Sequential(
              nn.Linear(static_feats, 64),
              nn.BatchNorm1d(64),
              nn.ReLU(),
              nn.Linear(64, self.tab_enc_dim),
              nn.BatchNorm1d(self.tab_enc_dim),
              nn.ReLU())

        
        out_dim = self.hidden_dim * 2 if bidirectional else self.hidden_dim

        fc_stack = []
        for fc_layer_size in fc_layer_sizes:
            fc_stack.append(nn.Linear(out_dim, fc_layer_size))
            fc_stack.append(nn.ReLU())
            out_dim = fc_layer_size

        self.fc_stack = nn.Sequential(*fc_stack)

        # Traj-level expander stack
        expander_stack = []
        for idx, fc_layer_size in enumerate(expander_fcs):
            expander_stack.append(nn.Linear(out_dim, fc_layer_size))
            if idx < len(expander_fcs) - 1:
                expander_stack.append(nn.BatchNorm1d(fc_layer_size))
                expander_stack.append(nn.ReLU())
            out_dim = fc_layer_size
        self.expander_stack = nn.Sequential(*expander_stack)
        
        
        # Signal level expander stack
        expander_stack_component = []
        out_dim = self.cnn_dim
        for idx, fc_layer_size in enumerate(expander_fcs):
            expander_stack_component.append(nn.Linear(out_dim, fc_layer_size))
            if idx < len(expander_fcs) - 1:
                expander_stack_component.append(nn.BatchNorm1d(fc_layer_size))
                expander_stack_component.append(nn.ReLU())
            out_dim = fc_layer_size
        self.expander_stack_component = nn.Sequential(*expander_stack_component)

        # Use the info in args to set up some other things
        self.do_simclr = args.do_simclr
        self.do_vicreg = args.do_vicreg

        self.simclr_temp = args.simclr_temp
        self.vicreg_mu, self.vicreg_lambda = args.vicreg_mu, args.vicreg_lambda
        self.global_weight = args.global_weight
        self.component_weight = args.component_weight
        self.detach = args.detach
        # Modality choices
        self.only_ecg = args.only_ecg
        self.only_tabular = args.only_tabular

        if self.bidirectional:
            self.h_0 = torch.zeros(2*num_layers,  1, self.hidden_dim).float().to('cuda' if use_cuda else 'cpu')
        else:
            self.h_0 = torch.zeros(num_layers,  1, self.hidden_dim).float().to('cuda' if use_cuda else 'cpu')

        self.pooling_method = pooling_method

        self.use_cuda = use_cuda
        self.n_gpu = n_gpu

        self.task_dims = TASK_OUTPUT_DIMENSIONS
        
        if self.task_weights is None and not (self.do_simclr or self.do_vicreg):
            raise NotImplementedError('Must specify task weight if doing supervised learning')

        if self.task_weights: 
            self.task_heads = {}
            for t, d in self.task_dims.items():
                if t in self.task_weights and self.task_weights[t] > 0:
                    mod = nn.Linear(self.hidden_dim,d)
                    self.task_heads[t] = mod
                    
            self.task_heads = nn.ModuleDict(self.task_heads)
            self.task_losses = get_task_losses(self.task_weights)


    def freeze_representation(self):
        for p in self.gru.parameters(): p.requires_grad = False
        for p in self.enc.parameters(): p.requires_grad = False
        for p in self.tabular_enc.parameters(): p.requires_grad = False
        for p in self.labs_enc.parameters(): p.requires_grad = False
        for p in self.fc_stack.parameters(): p.requires_grad = False

    def unfreeze_representation(self):
        for p in self.gru.parameters(): p.requires_grad = True
        for p in self.enc.parameters(): p.requires_grad = True
        for p in self.tabular_enc.parameters(): p.requires_grad = True
        for p in self.labs_enc.parameters(): p.requires_grad = True
        for p in self.fc_stack.parameters(): p.requires_grad = True


    # Standard supervised forward pass
    # forward should be called with a dictionary, via, e.g., model(**batch)
    def forward(
        self,
        dfs, # Should be a dict...
        h_0=None
    ):

        for k in ('signals_timeseries','structured_timeseries' 'statics',):
            if k in dfs: dfs[k] = dfs[k].float()
        for k in ['example_task','end_idx']:
            if k in dfs: 
                dfs[k] = dfs[k].squeeze().long()

        input_sequence = dfs['signals_timeseries']
        batch_size, seq_len, channels, num_samples = list(input_sequence.shape)

        # reshape the signals timeseries input sequence into a [BSxSEQ_LEN, channels, num_samples] array
        # Run the encoder on this tensor
        # reshape it back to [BS, SEQ_LEN, hidden_size]
        input_sequence = input_sequence.reshape(batch_size*seq_len, channels, num_samples)
        encoded = self.enc(input_sequence)
        encoded = encoded.reshape(batch_size, seq_len, -1)
        
        ### Process the structured timeseries features.
        ### These come in as a  BS , SEQ_LEN , Num_Feat tensor
        ### Reshape to BSxSEQ_LEN, num_feat tensor
        ## Run encoder, then reshape back
        tabular_sequence = dfs['structured_timeseries']
        batch_size, seq_len, feats = list(tabular_sequence.shape)
        tabular_sequence = tabular_sequence.reshape(batch_size*seq_len, feats)
        tabular_encoded = self.tabular_enc(tabular_sequence)
        tabular_encoded = tabular_encoded.reshape(batch_size, seq_len, -1)
        
        ## Process the statics
        ## These are BS, Num_feat  tensor
        ## Expand and repeat so that we cat it at each stage
        labs_sequence = dfs['statics']
        labs_encoded = self.labs_enc(labs_sequence).unsqueeze(1)
        labs_encoded = labs_encoded.repeat(1,seq_len, 1)
        
        if self.only_tabular and self.only_ecg:
            raise NotImplementedError
        elif self.only_tabular:
            encoded = torch.cat([tabular_encoded, labs_encoded], axis=2)
        elif self.only_ecg:
            pass
        else:
            encoded = torch.cat([encoded, tabular_encoded, labs_encoded], axis=2)
        
        if seq_len > 1:
            if h_0 is None:
                h_0 = self.h_0

            if batch_size != 1:
                h_0 = h_0.expand(-1, batch_size, -1).contiguous()

            out_unpooled, h = self.gru(encoded, h_0) # for gru
            
            # We want to compute the pooled representation over the actual 
            # sequence in question, rather than the padded tokens at the end.
            # Use dfs['end_idx] to help us out here.
            if self.pooling_method == 'last': 
                out = out_unpooled[torch.arange(len(out_unpooled)).long(), dfs['end_idx'], :]
            elif self.pooling_method == 'max':
                out = [i[:end+1].max(dim=0)[0] for i, end in zip(out_unpooled, dfs['end_idx'])]
                out = torch.stack(out, dim=0)
            elif self.pooling_method == 'avg':
                out = [i[:end+1].mean(dim=0) for i, end in zip(out_unpooled, dfs['end_idx'])]
                out = torch.stack(out, dim=0)

            out = out.contiguous().view(batch_size, -1) # num directions is 1 for forward-only rnn
        else:
            out = encoded.squeeze(1)
            out_unpooled = encoded.squeeze(1)

        pooled_output = self.fc_stack(out)
        unpooled_output = self.fc_stack(out_unpooled)

        # sequence_output.shape is batch_size, max_seq_length, hidden_dim
        # pooled_output is batch_size, hidden_dim

        # insert all the prediction tasks here
        task_labels = {
            k: df for k, df in dfs.items() if df is not None
        }
        tasks = list(set(task_labels.keys()).intersection(self.task_heads.keys()))

        task_logits = {}
        for t in tasks:
            op= torch.squeeze(self.task_heads[t](pooled_output))
            task_logits[t] = op

        task_losses = {}
        weights_sum = 0
        for task, loss_fn, logits, labels, weight in zip_dicts(
            self.task_losses, task_logits, dfs, self.task_weights
        ):
            weights_sum += weight # We do it like this so that we only track tasks that are actually used.
            loss = weight * loss_fn(logits, labels)
            task_losses[task] = loss


        try:
            total_loss = None
            for l in task_losses.values():
                total_loss = l if total_loss is None else (total_loss + l)
            total_loss /= weights_sum
        except:
            print(task_losses)
            print(weights_sum)
            raise

        out_data = {t: (task_logits[t].detach().cpu().numpy(), 
                        dfs[t].detach().cpu().numpy(), 
                        task_losses[t].detach().cpu().numpy()) for t in tasks}

        return (
            None,
            pooled_output,
            out_data,
            total_loss.unsqueeze(0) if self.n_gpu > 1 else total_loss
        )


    #### The ssl loss computation
    def ssl_forward(
        self,
        dfs, # Should be a dict...
        h_0=None
    ):
        
        signals_timeseries1 = dfs['signals_timeseries1']
        signals_timeseries2 = dfs['signals_timeseries2']

        batch_size, seq_len, channels, num_samples = list(signals_timeseries1.shape)
        
        structured_timeseries1 = dfs['structured_timeseries1']
        structured_timeseries2 = dfs['structured_timeseries2']
        feats = structured_timeseries2.shape[-1]
        
        statics1 = dfs['statics1']
        statics2 = dfs['statics2']

        if h_0 is None:
            h_0 = self.h_0

        if batch_size != 1:
            h_0 = h_0.expand(-1, batch_size, -1).contiguous()
        
        # Collect representations for both sequences
        component_reprs = []
        full_traj_reprs = []
        for labs_seq, tab_seq, inp_seq in zip([statics1, statics2],
                                 [structured_timeseries1, structured_timeseries2],
                                 [signals_timeseries1, signals_timeseries2]):
            # reshape the input sequence into a [BSxSEQ_LEN, channels, num_samples] array
            # Run the encoder on this tensor
            # reshape it back to [BS, SEQ_LEN, hidden_size]
            inp_seq = inp_seq.reshape(batch_size*seq_len, channels, num_samples)
            encoded = self.enc(inp_seq)
            
            
            ### Process the tabular features.
            ### These come in as a  BS , SEQ_LEN , Num_Feat tensor
            ### Reshape to BSxSEQ_LEN, num_feat tensor
            ## Run encoder, then reshape back
            _,_, feats = list(tab_seq.shape)
            tab_seq = tab_seq.reshape(batch_size*seq_len, feats)
            tabular_encoded = self.tabular_enc(tab_seq)

            labs_encoded = self.labs_enc(labs_seq).unsqueeze(1)
            labs_encoded = labs_encoded.repeat(1,seq_len, 1)
            
            encoded_repr = self.expander_stack_component(encoded)
            component_reprs.append(encoded_repr)
                

            if self.only_tabular and self.only_ecg:
                raise NotImplementedError
            elif self.only_tabular:
                encoded = tabular_encoded
                encoded = encoded.reshape(batch_size, seq_len, -1)
                encoded = torch.cat([encoded, labs_encoded], axis=2)
            elif self.only_ecg:
                encoded = encoded.reshape(batch_size, seq_len, -1)
            else:
                encoded = torch.cat([encoded, tabular_encoded], axis=1)
                encoded = encoded.reshape(batch_size, seq_len, -1)
                # add in the labs
                encoded = torch.cat([encoded, labs_encoded], axis=2)


            out_unpooled, h = self.gru(encoded, h_0) # for gru

            if self.pooling_method == 'last': 
                out = out_unpooled[torch.arange(len(out_unpooled)).long(), dfs['end_idx'].long(), :]
            elif self.pooling_method == 'max':
                out = [i[:end+1].max(dim=0)[0] for i, end in zip(out_unpooled, dfs['end_idx'].long())]
                out = torch.stack(out, dim=0)
            elif self.pooling_method == 'avg':
                out = [i[:end+1].mean(dim=0) for i, end in zip(out_unpooled, dfs['end_idx'].long())]
                out = torch.stack(out, dim=0)

            out = out.contiguous().view(batch_size, -1)
            
            # Apply the FC layers before the projection head. 
            pooled_output = self.fc_stack(out)

            pooled_output = self.expander_stack(pooled_output)
            full_traj_reprs.append(pooled_output)

        # sequence_output.shape is batch_size, max_seq_length, hidden_dim
        # pooled_output is batch_size, hidden_dim
        traj_loss_dict = {}
        component_loss_dict = {}
        if self.do_simclr:
            traj_loss = self.get_simclr_loss(full_traj_reprs)

            #### NEGATIVES ARE STRICTLY FROM DIFFERENT TRAJECTORIES! ####
            component_reprs_view1, component_reprs_view2 = component_reprs
            component_reprs_view1 = component_reprs_view1.reshape(batch_size, seq_len, -1)
            component_reprs_view2 = component_reprs_view2.reshape(batch_size, seq_len, -1)
            
            component_losses = [self.get_simclr_loss([component_reprs_view1[:,t], 
                                              component_reprs_view2[:,t]]) for t in range(seq_len)]
            component_loss = sum(component_losses)/len(component_losses)

        elif self.do_vicreg:
            traj_loss, traj_loss_dict = self.get_vicreg_loss(full_traj_reprs)
            traj_loss_dict = {f'traj_{i}':j for i,j in traj_loss_dict.items()}

            #### NEGATIVES ARE STRICTLY FROM DIFFERENT TRAJECTORIES! ####
            component_reprs_view1, component_reprs_view2 = component_reprs
            component_reprs_view1 = component_reprs_view1.reshape(batch_size, seq_len, -1)
            component_reprs_view2 = component_reprs_view2.reshape(batch_size, seq_len, -1)
            
            # Compute loss per-timestep and then average
            component_loss_dict = None
            component_loss = 0.0
            for t in range(seq_len):
                component_loss_t, component_loss_dict_t = self.get_vicreg_loss([component_reprs_view1[:,t], 
                                              component_reprs_view2[:,t]])
                component_loss = component_loss + component_loss_t
                if component_loss_dict is None:
                    component_loss_dict = {f'component_{i}':j for i,j in component_loss_dict_t.items()}
                else:
                    component_loss_dict = {f'component_{i}': component_loss_dict[f'component_{i}']+j for i,j in component_loss_dict_t.items()}
    
            component_loss_dict = {i:j/seq_len for i,j in component_loss_dict.items()}
        
            # Prevent exploding values -- sometimes can occur?
            component_loss = torch.clamp(component_loss/seq_len, min=None, max=100)
            traj_loss = torch.clamp(traj_loss, min=None, max=100)
        else:
            raise NotImplementedError

        total_loss = self.global_weight*traj_loss + self.component_weight*component_loss
        
        loss_dict = {'total_loss' : total_loss.item(),
             'traj_loss': traj_loss.item(),
             'component_loss': component_loss.item()}
        
        loss_dict.update(traj_loss_dict)
        loss_dict.update(component_loss_dict)

        return (
            None,
            pooled_output.detach().cpu().numpy(),
            loss_dict,
            total_loss.unsqueeze(0) if self.n_gpu > 1 else total_loss
        )


    def get_simclr_loss(self, reprs):
        zis, zjs = reprs
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        device = zis.device
        BS = zis.shape[0]
        loss_obj = NTXentLoss(device, BS, self.simclr_temp, use_cosine_similarity=True)

        loss = loss_obj(zis, zjs)
        return loss

    def get_vicreg_loss(self, reprs):
        zis, zjs = reprs
        loss = vicreg_loss_func(zis, zjs, self.vicreg_mu, self.vicreg_lambda)
        return loss

