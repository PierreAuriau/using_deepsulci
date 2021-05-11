"""

"""

# Author : Bastien Cagna (bastiencagna@gmail.com)

import os.path as op
from os import makedirs, listdir
import json
from using_deepsulci.cohort import Cohort, SubjectDataset
from soma import aims, aimsalgo
import sys
import numpy as np
from joblib import Parallel, cpu_count, delayed


def resampled_graph(graph_f, output_vs, out_dir, suffix="_resampled"):
    if not graph_f:
        return None
    fname = op.split(graph_f)[1][:-4] + suffix + ".arg"
    fpath = op.join(out_dir, fname)

    trm = aims.AffineTransformation3d(np.eye(4))
    inv_trm = trm.inverse()

    if not op.exists(fpath):
        print("Resampling " + fpath)
        graph = aims.read(graph_f)
        aimsalgo.transformGraph(graph, trm, inv_trm, output_vs)
        aims.write(graph, fpath)
    return fpath


def resample_subejct_job(sub, vx_size, rs_dir, suffix):
    return SubjectDataset(
        sub.name,
        resampled_graph(sub.graph, vx_size, rs_dir, suffix),
        resampled_graph(sub.notcut_graph, vx_size, rs_dir, suffix)
    )


def add_resampled_graphs(cohort_f, rs_dir, vx_size, suffix):
    cohort = Cohort(from_json=cohort_f)
    print("Resampling cohort: " + cohort.name)
    subjects = []
    # for sub in cohort.subjects:
    #     subjects.append(SubjectDataset(
    #         sub.name,
    #         resampled_graph(sub.graph, vx_size, rs_dir, suffix),
    #         resampled_graph(sub.notcut_graph, vx_size, rs_dir, suffix)
    #     ))
    subjects = Parallel(n_jobs=cpu_count()-1)(delayed(
        resample_subejct_job)(sub, vx_size, rs_dir, suffix)
        for sub in cohort.subjects)

    rs_cohort = Cohort(cohort.name + suffix, subjects)
    cohort_dir, _ = op.split(cohort_f)
    rs_cohort.to_json(op.join(cohort_dir, "cohort-" + rs_cohort.name + ".json"))


def main():
    # Load environnment file
    env = json.load(open(op.join(op.split(__file__)[0], "env.json")))

    # Resample to 2mm iso
    ovs = 2
    output_vs = (ovs, ovs, ovs)
    rsdir = op.join(env['working_path'], "resampled")
    makedirs(rsdir, exist_ok=True)

    # Set cohorts paths
    chrt_dir = op.join(env['working_path'], "cohorts")
    if len(sys.argv) > 1:
        cohorts_files = []
        for cname in sys.argv[1:]:
            cohorts_files.append(
                op.join(chrt_dir, "cohort-" + cname + ".json")
            )
    else:
        cohorts_files = list(op.join(chrt_dir, f) for f in listdir(chrt_dir))

    suffix = "_resampled-{}mm".format(ovs)
    for cfile in cohorts_files:
        if cfile[-7:-5] != "mm":
            add_resampled_graphs(cfile, rsdir, (2, 2, 2), suffix)


if __name__ == "__main__":
    main()
