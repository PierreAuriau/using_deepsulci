import sys
import json
import os.path as op
from os import makedirs

from deepsulci.sulci_labeling.capsul.labeling import SulciDeepLabeling
from deepsulci.sulci_labeling.capsul.error_computation import ErrorComputation

from using_deepsulci.cohort import Cohort
from using_deepsulci.processes.classification_evaluation import \
    DeepClassificationEvaluation

from capsul.api import capsul_engine

# def evaluate_model(cohort, translation_file, model_file, param_file, out_file):
#     graphs = []
#     for s in cohort.subjects:
#         graphs.append(s.graph)
#     print(len(graphs), "graphs to evaluate")
#
#     proc = DeepClassificationEvaluation()
#     proc.graphs = graphs
#     proc.translation_file = translation_file
#     proc.model_file = model_file
#     proc.param_file = param_file
#     proc.cuda = -1
#     proc.out_file = out_file
#     proc._run_process()


def evaluate_model(cohort, model_file, param_file, labeled_dir, esi_dir=None):
    # ce = capsul_engine()
    esi_dir = labeled_dir if esi_dir is None else esi_dir
    ss_list = json.load(open(param_file))['sulci_side_list']

    for sub in cohort.subjects:
        g_fname = op.split(sub.graph)[1]
        labeled_graph = op.join(labeled_dir, g_fname)

        # lab_proc = ce.get_process_instance(
        #     'deepsulci.sulci_labeling.capsul.labeling')
        lab_proc = SulciDeepLabeling()
        lab_proc.graph = sub.graph
        lab_proc.roots = sub.roots
        lab_proc.skeleton = sub.skeleton
        lab_proc.model_file = model_file
        lab_proc.param_file = param_file
        lab_proc.labeld_graph = labeled_graph

        # esi_proc = ce.get_process_instance(
        #     'deepsulci.sulci_labeling.capsul.error_computation')
        esi_proc = ErrorComputation()
        esi_proc.t1mri = sub.t1
        esi_proc.true_graph = sub.graph
        esi_proc.labeled_graph = labeled_graph
        esi_proc.sulci_side_list = ss_list
        esi_proc.error_file = op.join(esi_dir, g_fname)


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
