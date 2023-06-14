import json, os, torch, torch.optim, torch.nn as nn, torch.nn.init as init
from collections import OrderedDict

from ..utils import *
from ..constants import *
from ..representation_learner.adapted_model import *


def strip_module_and_load(recipient, loaded_state_dict):
    try:
        old_state_dict = recipient.state_dict()
        recipient.load_state_dict(loaded_state_dict, strict=False)
    except RuntimeError as e:
        if 'Missing key(s) in state_dict: "fc_stack.0.weight", "fc_stack.0.bias"' in str(e):
            print("Fixing old broken GRU...")
            new_state_dict = OrderedDict({
                k: v for k, v in loaded_state_dict.items() if k not in ('fc.weight', 'fc.bias')
            })
            new_state_dict['fc_stack.0.weight'] = loaded_state_dict['fc.weight']
            new_state_dict['fc_stack.0.bias'] = loaded_state_dict['fc.bias']


            k = "task_losses.tasks_binary_multilabel.pos_weight"
            if k in old_state_dict and k not in new_state_dict:
                print("Fixing missing pos_weight key...")
                print(old_state_dict[k])
                new_state_dict[k] = old_state_dict[k]
        elif 'Missing key(s) in state_dict: "task_losses.tasks_binary_multilabel.pos_weight"' in str(e):
            print("Fixing missing pos_weight key...")
            new_state_dict = OrderedDict(loaded_state_dict)
            print(old_state_dict[
                "task_losses.tasks_binary_multilabel.pos_weight"
            ])

            new_state_dict["task_losses.tasks_binary_multilabel.pos_weight"] = old_state_dict[
                "task_losses.tasks_binary_multilabel.pos_weight"
            ]
        elif 'Missing key(s) in state_dict: "task_losses.tasks_binary_multilabel.BCE_LL.pos_weight' in str(e):
            # error introduced in bug fix for BCE
            print("Fixing missing BCE.LL pos_weight key...")
            new_state_dict = OrderedDict(loaded_state_dict)
            print(old_state_dict[
                "task_losses.tasks_binary_multilabel.BCE_LL.pos_weight"
            ])
            print("task_losses.tasks_binary_multilabel.BCE_LL.pos_weight" in old_state_dict.keys())

            new_state_dict["task_losses.tasks_binary_multilabel.BCE_LL.pos_weight"] = old_state_dict[
                "task_losses.tasks_binary_multilabel.BCE_LL.pos_weight"
            ]
        else:
            prefix = "module."
            new_state_dict = OrderedDict({
                (k[len(prefix):] if k.startswith(prefix) else k): v for k, v in loaded_state_dict.items()
            })

        recipient.load_state_dict(new_state_dict)

class MetaModel():
    def __init__(self, args, sample_datum, class_names=None, verbose=False, task_weights=None, use_cuda=torch.cuda.is_available()):
        print("curr path:", args.run_dir)
        assert os.path.isdir(args.run_dir)
        if class_names is None: class_names = {}

        self.run_dir = args.run_dir
        self.debug = False

        device = torch.device("cuda" if use_cuda else "cpu")
        n_gpu = 0 if not use_cuda else torch.cuda.device_count()
        self.device = device
        self.n_gpu = n_gpu

        # SSL params
        self.do_simclr = args.do_simclr
        self.do_vicreg = args.do_vicreg
        self.expander_fcs = args.expander_fcs
        self.simclr_temp = args.simclr_temp
        self.vicreg_lambda = args.vicreg_lambda
        self.vicreg_mu = args.vicreg_mu

        sig_channels, sig_samples = sample_datum['signals_timeseries'].shape[1], sample_datum['signals_timeseries'].shape[2]
        sig_features = sample_datum['structured_timeseries'].shape[1]
        static_features = sample_datum['statics'].shape[0]

        
        if args.modeltype == 'cnn_gru':
            model = CNNGRUModel(data_shape=[args.max_seq_len, sig_channels, sig_samples], 
                tabular_feats=sig_features, static_feats=static_features,
                use_cuda=torch.cuda.is_available(),
                hidden_dim=args.cnn_enc_dim, num_layers=args.gru_num_hidden,
                bidirectional=args.do_bidirectional, task_weights=task_weights,
                pooling_method=args.gru_pooling_method,
                verbose = verbose,
                expander_fcs = args.expander_fcs,
                args=args)
        else:
            raise NotImplementedError
            

        for m in (model, ):
            if m is None: continue
            m.to(device)
            if n_gpu > 1: m = torch.nn.DataParallel(m).cuda()

        parameters = model.parameters()

        self.model = model
        self.parameters = parameters

        self.trainable_models = [self.model, ]

        self.n_gpu = n_gpu
        self.device = device

        self.run_dir = args.run_dir
        self.save_name = args.model_file_template.format(**args.to_dict())


    def parameters(self): 
        return self.parameters

    def freeze_representation(self):
        self.model.freeze_representation()

    def unfreeze_representation(self):
        self.model.unfreeze_representation()

    def train(self):
        for m in self.trainable_models:
            if m is not None: 
                m.train()

    def eval(self):
        for m in self.trainable_models:
            if m is not None: 
                m.eval()

    def state_dict(self):
        state_dict = {
            'model': self.model.state_dict(),
        }
        return state_dict

    def save(self, epoch=0, extra_savename=''):
        to_save = {'epoch': epoch, **self.state_dict()}

        save_path = os.path.join(self.run_dir, '%s%s.epoch-%d' % (extra_savename, self.save_name, epoch))
        torch.save(to_save, save_path)

    def load(self, epoch='latest'):
        print('Model loading...')
        if epoch == 'latest':
            files = os.listdir(self.run_dir)

            all_epochs = []
            prefix = '%s.epoch-' % self.save_name
            for f in files:
                if not f.startswith(prefix): continue
                all_epochs.append(int(f[len(prefix):]))
            if not all_epochs: 
                print('No model found. Starting from scratch')
                return False, None
            epoch = max(all_epochs)

        assert type(epoch) is int and epoch >= 0, "epoch must be 'latest' or an epoch #"

        load_path = os.path.join(self.run_dir, '%s.epoch-%d' % (self.save_name, epoch))
        if not os.path.isfile(load_path): 
            print('No valid file found. Starting from scratch')
            return False, None

        to_load = torch.load(load_path, map_location=self.device)
        assert to_load['epoch'] == epoch, "Something is wrong... %d v. %d" % (to_load['epoch'], epoch)

        strip_module_and_load(self.model, to_load['model'])

        print('Model loaded from epoch:', epoch)
        return True, epoch


    def forward(self, batch):
        for k, value in batch.items(): 
            if k != 'pt_ids':
                batch[k]=value.float().to(self.device)

        signal = batch['signals_timeseries']

        if not (self.do_simclr or self.do_vicreg):
            hidden_states, pooled_output, all_outputs, total_loss = self.model.forward(batch)
        else:
            # special SSL forward function
            hidden_states, pooled_output, all_outputs, total_loss = self.model.ssl_forward(batch)
        if self.n_gpu > 1: total_loss = total_loss.mean() # Across all gpus...

        return hidden_states, pooled_output, all_outputs, total_loss

