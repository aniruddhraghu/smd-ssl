import sys
sys.path.append('../')

import pickle
from tqdm import tqdm

from trajectories.constants import *
from trajectories.representation_learner.args import FineTuneArgs
from trajectories.representation_learner.lineval import *


if __name__=="__main__":
    args = FineTuneArgs.from_commandline()
    main(args, tqdm)
