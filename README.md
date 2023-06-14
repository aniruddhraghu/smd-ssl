# SMD-SSL (ICML 2023)

This repository contains code to implement Sequential Multidimensional Self-Supervised Learning (SMD-SSL), ICML 2023. 

Our implementation and the structure of this code is based on the implementation [here](https://github.com/mmcdermott/comprehensive_MTL_EHR), a prior work on pre-training methods for clinical time-series data.

## Setup

To get started, install conda and set up the environment as follows:

```
conda env create --name traj -f env.yml
conda activate traj
```

## Code details

Due to restrictions on the data, we cannot release the datasets at this stage. The code here is therefore written for a general-purpose dataset of trajectory-structured data, which can be adapted to the specific use-case investigated. The key directories in the codebase are:

* `SampleArgs/`: Contains folders with different experimental configurations. An example pre-training configuration is provided in `SampleArgs/testing/args.json`, and an example fine-tuning configuration is provided in `SampleArgs/testing/fine_tune_args.json`. 
*  `Scripts/`: The high-level scripts that are used to run pre-training (`Scripts/run_model.py`), fine-tuning (`Scripts/fine_tune_task.py`), and linear evaluation (`Scripts/lineval_task.py`). 
* `trajectories/` Contains the actual modelling code and the training/evaluation functions.

### Details about modelling code in `trajectories/`

Key files within this directory include:
* `constants.py`: key constant definitions (such as the filenames of pre-training and fine-tuning args) and also declares names of different fine-tuning tasks, and their output spaces. If the codebase is to be used for a new downstream task, this file should be updated.
* `representation_learner/adapted_model.py`: Contains the definition of the actual neural network model (CNN-GRU) that is used to encode trajectory-structured data. This torch model has two forward functions, one for self-supervised pre-training, and one for fine-tuning. Functionality for projection heads, loss computation, etc is within this file.
* `representation_learner/args.py`: Contains all arguments used at pre-training and fine-tuning time. New arguments (e.g., specifying augmentations, additional loss weights, etc) can be added here. 
* `representation_learner/example_dataset.py`: A template dataset class that generates trajectory-styled data. Currently, this code dynamically generates random sequences of waveforms and structured data of the correct dimensionality rather than producing real clinical data -- it can be edited to work with an actual clinical dataset. This file also contains functionality to generate augmentations of the raw data that are then used for multivew self-supervised pre-training.
* `representation_learner/fine_tune.py`: Contains the functionality to run fine-tuning or training from random initialization on a downstream task.
* `representation_learner/lineval.py`: Contains the functionality to conduct linear evaluation with a logistic regression model on a downstream task, from frozen representations.
* `representation_learner/meta_model.py`: Wraps the `adapted_model.py` to provide a cleaner API.
* `representation_learner/run_model.py`: Contains functionality to run training and evaluation loops, and conduct logging to weights and biases (currently commented out -- can uncomment and configure as desired). This file also handles dataset and dataloader creation, which are used during the training/eval loops. 


## Commands to run experiments

To pre-train a model, navigate to the `Scripts/` folder (`cd Scripts`) and run:

```PROJECTS_BASE='.' python run_model.py --do_load_from_dir --run_dir ../SampleArgs/testing/```

and specify the arguments in `SampleArgs/testing/args.json` -- this includes things like number of epochs, augmentation strengths, loss weights, what SSL loss (SimCLR or VICReg), what data modalities, etc.

To fine-tune a model, navigate to the `Scripts/` folder (`cd Scripts`) and run:

```PROJECTS_BASE='.' python fine_tune_task.py --do_load_from_dir --run_dir ../SampleArgs/testing/```

and specify the arguments in `SampleArgs/testing/fine_tune_args.json` -- this includes things like number of epochs of training, whether to freeze representations and just learn a linear head, etc. 

Linear eval can be run with ```PROJECTS_BASE='.' python lineval_task.py --do_load_from_dir --run_dir ../SampleArgs/testing/```

## Example notebooks
The two jupyter notebooks `dataloader_test.ipynb` and `model_test.ipynb` can be used to inspect the dataloader and model outputs. 
