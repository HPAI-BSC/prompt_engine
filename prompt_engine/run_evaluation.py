import os

from run import prepare
from src.evaluate import evaluate_from_config



def launch_evaluations(subjects, config):
    for subject in subjects:
        evaluate_from_config(subject, config)


if __name__ == "__main__":
    subjs, configuration = prepare()
    launch_evaluations(subjs, configuration)


