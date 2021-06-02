"""
    This script train one or several model using labelled graphs of a given
    cohorts. The learning is very very long (like arround 36 hours for 140
    subjects using GPU).
"""
import os.path as op
from deepsulci.sulci_labeling.capsul.training import SulciDeepTraining
import json
from os import makedirs
from datetime import datetime
import argparse

from using_deepsulci.cohort import Cohort
from using_deepsulci.utils.misc import add_to_text_file


def train_cohort(cohort, out_dir, modelname, translation_file=None, steps=None, cuda=-1):
    proc = SulciDeepTraining()


    # Inputs
    proc.graphs = cohort.get_graphs()
    proc.graphs_notcut = cohort.get_notcut_graphs()
    proc.cuda = cuda
    proc.translation_file = translation_file

    # Steps
    proc.step_1 = not steps or (steps and 1 in steps)
    proc.step_2 = not steps or (steps and 2 in steps)
    proc.step_3 = not steps or (steps and 3 in steps)
    proc.step_4 = not steps or (steps and 4 in steps)
    #bool(len(cohort.get_notcut_graphs()))

    # Outputs
    fname = "cohort-" + cohort.name + "_model-" + modelname
    proc.model_file = op.join(out_dir, fname + "_model.mdsm")
    proc.param_file = op.join(out_dir, fname + "_params.json")
    proc.traindata_file = op.join(out_dir, fname + "_traindata.json")

    # Run
    if op.exists(proc.model_file):
        print("Skipping the training. Model file already exist.")
        print(proc.model_file)
    else:
        proc.run()


def main():
    parser = argparse.ArgumentParser(description='Train CNN model')
    parser.add_argument('-c', dest='cohorts', type=str, nargs='+', default=None, required=False,
                        help='Cohort names')

    parser.add_argument('-s', dest='steps', type=int, nargs='+', default=None,
                        help='Steps to run')
    parser.add_argument('--cuda', dest='cuda', type=int, default=-1,
                        help='Use a speciific cuda device ID or CPU (-1)')
    parser.add_argument('-e', dest='env', type=str, default=None,
                        help="Configuration file")
    args = parser.parse_args()

    # Load environnment file
    env_f = args.env if args.env else op.join(op.split(__file__)[0], "env.json")
    env = json.load(open(env_f))

    cohorts_dir = op.join(env['working_path'], "cohorts")
    outdir = op.join(env['working_path'], "models")
    makedirs(outdir, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H:%M:%S")
    makedirs(op.join(env["working_path"], "logs"), exist_ok=True)
    log_f = op.join(env["working_path"], "logs", "step_02_" + now + ".log")

    cohorts = []
    for c in args.cohorts:
        cohorts.append(Cohort(from_json=op.join(cohorts_dir, "cohort-" + c + ".json")))
    cohorts = sorted(cohorts, key=len)

    for cohort in cohorts:
        print("\n\n ****** START TO TRAIN ", cohort.name)
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        add_to_text_file(log_f, "{} - Start to train {}".format(now, cohort.name))
        train_cohort(cohort, outdir, "unet3d", env['translation_file'], args.steps, args.cuda)
    return None


if __name__ == "__main__":
    main()
