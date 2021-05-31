import sys
import json
import os.path as op
from os import makedirs
import argparse

from deepsulci.sulci_labeling.capsul.labeling import SulciDeepLabeling
from deepsulci.sulci_labeling.capsul.labeling_evaluation import LabelingEvaluation

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
    params = json.load(open(param_file))

    ss_list = params['sulci_side_list']

    if 'cutting_threshold' not in params.keys():
        # TODO: better manage of this or verify that the default value
        print("/!\\ No cutting threshold, setting arbitrary value: 250")
        params['cutting_threshold'] = 250
        json.dump(params, open(param_file, 'w+'))

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
        lab_proc.labeled_graph = labeled_graph
        lab_proc.run()

        # esi_proc = ce.get_process_instance(
        #     'deepsulci.sulci_labeling.capsul.error_computation')
        esi_proc = LabelingEvaluation()
        esi_proc.t1mri = sub.t1
        esi_proc.true_graph = sub.graph
        esi_proc.labeled_graphs = [labeled_graph]
        esi_proc.sulci_side_list = ss_list
        esi_proc.scores_file = op.join(esi_dir, g_fname[:-4] + '_scores.csv')
        esi_proc.run()


def main():
    parser = argparse.ArgumentParser(description='Test trained CNN model')
    parser.add_argument('-c', dest='cohort', type=str, default=None, required=False,
                        help='Testing cohort name')
    parser.add_argument('-m', dest='model', type=str, default=None, required=False,
                        help='Model name')
    # parser.add_argument('--cuda', dest='cuda', type=int, default=-1,
    #                     help='Use a speciific cuda device ID or CPU (-1)')
    parser.add_argument('-e', dest='env', type=str, default=None,
                        help="Configuration file")
    args = parser.parse_args()

    # Load environnment file
    env_f = args.env if args.env else op.join(op.split(__file__)[0], "env.json")
    env = json.load(open(env_f))
    model_dir = op.join(env['working_path'], "models")
    cohort_dir = op.join(env['working_path'], "cohorts")

    print("Evaluate:", args.model)

    # cohortname = modelname.split("_model")[0]
    cohort_f = op.join(cohort_dir, args.cohort + ".json")

    model_f = op.join(model_dir, args.model + "_model.mdsm")
    params_f = op.join(model_dir, args.model + "_params.json")

    out_d = op.join(env['working_path'], "evaluations", args.model)
    makedirs(out_d, exist_ok=True)
    # fname = modelname + "_teston-" + cohortname + ".npy"
    # evaluate_model(Cohort(from_json=cohort_f), env['translation_file'],
    #                model_f, params_f, op.join(out_d, fname))
    evaluate_model(Cohort(from_json=cohort_f), model_f, params_f,
                   labeled_dir=out_d)


if __name__ == "__main__":
    main()
