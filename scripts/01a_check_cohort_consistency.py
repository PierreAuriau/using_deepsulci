import os.path as op
from using_deepsulci.cohort import Cohort
from using_deepsulci.processes.labels_list import SulciList
import argparse
from os import makedirs
import json


def cohort_sulci_list(cohort):
    process = SulciList()
    process.graphs = list(sub.graph for sub in cohort.subjects)
    process.notcut = list(sub.notcut_graph for sub in cohort.subjects)
    process.run_process()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resample cohort files to desired resolution')
    parser.add_argument('-c', dest='cohorts', type=str, nargs='+', default=None, required=False,
                        help='Cohort names')
    # parser.add_argument('-n', dest='njobs', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('-e', dest='env', type=str, default=None, help="Configuration file")
    args = parser.parse_args()

    # Load environnment file
    env_f = args.env if args.env else op.join(op.split(__file__)[0], "env.json")
    env = json.load(open(env_f))

    odir = op.join(env['working_path'], "sulci_list")
    makedirs(odir, exist_ok=True)

    # Set cohorts paths
    chrt_dir = op.join(env['working_path'], "cohorts")
    cohorts_files = []
    for cname in args.cohorts:
        cohort_sulci_list(
            Cohort(from_json=op.join(chrt_dir, "cohort-" + cname + ".json")))
