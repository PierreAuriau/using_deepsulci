import sys
import json
import os.path as op
from os import makedirs

from using_deepsulci.cohort import Cohort
from using_deepsulci.processes.classification_evaluation import \
    DeepClassificationEvaluation


def evaluate_model(cohort, translation_file, model_file, param_file, out_file):
    graphs = []
    for s in cohort.subjects:
        graphs.append(s.graph)
    print(len(graphs), "graphs to evaluate")

    proc = DeepClassificationEvaluation()
    proc.graphs = graphs
    proc.translation_file = translation_file
    proc.model_file = model_file
    proc.param_file = param_file
    proc.cuda = -1
    proc.out_file = out_file
    proc._run_process()


def main():
    env = json.load(open(op.join(op.split(__file__)[0], "env.json")))
    model_dir = op.join(env['working_path'], "models")
    cohort_dir = op.join(env['working_path'], "cohorts")

    modelname = sys.argv[1]
    cohortname = sys.argv[2]

    print("Evaluate:", modelname)

    # cohortname = modelname.split("_model")[0]
    cohort_f = op.join(cohort_dir, cohortname + ".json")

    model_f = op.join(model_dir, modelname + "_model.mdsm")
    params_f = op.join(model_dir, modelname + "_params.json")

    out_d = op.join(env['working_path'], "evaluations", modelname)
    makedirs(out_d, exist_ok=True)
    fname = modelname + "_teston-" + cohortname + ".npy"

    evaluate_model(Cohort(from_json=cohort_f), env['translation_file'],
                   model_f, params_f, op.join(out_d, fname))


if __name__ == "__main__":
    main()
