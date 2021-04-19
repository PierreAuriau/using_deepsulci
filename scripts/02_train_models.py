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
from datetime import datetime

from using_deepsulci.cohort import Cohort
from using_deepsulci.utils.misc import add_to_text_file


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
    if op.exists(proc.model_file):
        print("Skipping the training. Model file already exist.")
    else:
        proc._run_process()


def main():
    # Load environnement file
    env = json.load(open(op.join(op.split(__file__)[0], "env.json")))
    cohorts_dir = op.join(env['working_path'], "cohorts")

    infered_file = None
    if len(sys.argv) < 2:
        sys.argv.append(cohorts_dir)
    else:
        infered_file = op.join(cohorts_dir, "cohort-" + sys.argv[1] + ".json")

    if op.isdir(sys.argv[1]):
        cohorts = []
        for cfile in listdir(sys.argv[1]):
            cohorts.append(Cohort(from_json=op.join(sys.argv[1], cfile)))
        print(len(cohorts), " models to train")
        for c in cohorts:
            print("\t{}: {} subjects".format(c.name, len(c)))
    elif op.isfile(sys.argv[1]):
        cohorts = [Cohort(from_json=sys.argv[1])]
    elif op.isfile(infered_file):
        cohorts = [Cohort(from_json=infered_file)]
    else:
        raise Exception("Cannot understand what should be done.")

    cohorts = sorted(cohorts, key=len)

    cuda = int(sys.argv[2]) if isinstance(sys.argv[-1]) else env['default_cuda']

    outdir = op.join(env['working_path'], "models")
    makedirs(outdir, exist_ok=True)

    now = datetime.now().strftime("%Y%m%d_%H:%M:%S")
    makedirs(op.join(env["working_path"], "logs"), exist_ok=True)
    log_f = op.join(env["working_path"], "logs", "step_02_" + now + ".log")

    for cohort in cohorts:
        print("\n\n ****** START TO TRAIN ", cohort.name)
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        add_to_text_file(log_f, "{} - Start to train {}".format(now,
                                                                cohort.name))
        train_cohort(
            cohort,
            outdir,
            "unet3d",
            env['translation_file'],
            cuda
        )
    return None


if __name__ == "__main__":
    main()
