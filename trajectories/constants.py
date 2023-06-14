import os

PROJECT_NAME                = 'trajectories'
PROJECT_DIR                 = os.path.join(os.environ['PROJECTS_BASE'], PROJECT_NAME)
RUNS_DIR                    = os.path.join(PROJECT_DIR, 'runs')
ARGS_FILENAME               = 'args.json'
FINE_TUNE_ARGS_FILENAME     = 'fine_tune_args.json'

ALL_TASKS = [
    'example_task',
]


TASK_OUTPUT_DIMENSIONS = {
    'example_task':2,
}

TRAIN, TUNING, HELD_OUT = 'train', 'tuning', 'held out'


