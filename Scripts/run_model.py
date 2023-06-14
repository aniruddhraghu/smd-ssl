import sys
sys.path.append('../')

from tqdm import tqdm

from trajectories.constants import *
from trajectories.representation_learner.args import *
from trajectories.representation_learner.run_model import *


if __name__=="__main__":
    args = Args.from_commandline()

    main(args, tqdm=tqdm)
