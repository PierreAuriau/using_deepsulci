import os.path as op
from joblib import cpu_count
from using_deepsulci.cohort import Cohort


def real_njobs(n):
    return min(n, cpu_count()) if n > 0 else cpu_count() - n


def read_cohorts(cpath, cnames):
    cnames = cnames if isinstance(cnames, list) else [cnames]

    cohorts = []
    for cname in cnames:
        cohorts.append(Cohort(from_json=op.join(cpath, "cohort-" + cname + ".json")))
    return cohorts
