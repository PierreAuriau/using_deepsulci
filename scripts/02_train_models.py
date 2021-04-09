"""
    This script train one or several model using labelled graphs of a given
    cohorts. The learning is very very long (like arround 36 hours for 140
    subjects using GPU).
"""
import os.path as op
from deepsulci.sulci_labeling.capsul import SulciDeepTraining
import json
import sys
from os import makedirs, listdir

from using_deepsulci.cohort import Cohort


def train_cohort(cohort, out_dir, modelname, translation_file=None, cuda=-1):
    proc = SulciDeepTraining()

    # Inputs
    proc.graphs = cohort.get_graphs()
    proc.graphs_notcut = cohort.get_notcut_graphs()
    proc.cuda = cuda
    proc.translation_file = translation_file

    # Steps
    proc.step_1 = True
    proc.step_2 = True
    proc.step_3 = True
    proc.step_4 = bool(len(cohort.get_notcut_graphs()))

    # Outputs
    fname = "cohort-" + cohort.name + "_model-" + modelname
    proc.model_file = op.join(out_dir, fname + "_model.mdsm")
    proc.param_file = op.join(out_dir, fname + "_params.json")
    proc.traindata_file = op.join(out_dir, fname + "_traindata.json")

    # Run
    proc._run_process()


def main():
    # Load environnement file
    env = json.load(open(op.join(op.split(__file__)[0], "env.json")))

    if op.isdir(sys.argv[1]):
        cohorts = []
        for cfile in listdir(sys.argv[1]):
            cohorts.append(Cohort(from_json=op.join(sys.argv[1], cfile)))
        print(len(cohorts), " models to train")
        for c in cohorts:
            print("\t{}: {} subjects".format(c.name, len(c)))
    else:
        cohorts = [sys.argv[1]]

    cuda = int(sys.argv[2]) if len(sys.argv) > 2 else env['default_cuda']

    for cohort in cohorts:
        train_cohort(
            cohort,
            makedirs(op.join(env['working_path'], "models"), exist_ok=True),
            cuda,
            env['translation_file']
        )
    return None


if __name__ == "__main__":
    main()
