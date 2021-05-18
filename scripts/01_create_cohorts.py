"""
    This script create JSON files that will be used to provide graphs when
    learning models. It defines a set of cohort based on several database where
    subjects' graphs have been mannually labeled.

    For now (08/04/2021), three databases are available with a total of 216
    subjects.

    This script need a env.json file saved in this directory that specify:
        - "bv_databases_path": Where all Brainvisa databased are located
        - "working_path": Where output files will be saved

    If any graph is missing, the whole database will be considered as
    unavailable.
"""

# Author : Bastien Cagna (bastiencagna@gmail.com)

import os.path as op
from os import makedirs
import json
import argparse
from using_deepsulci.cohort import bv_cohort, Cohort


def foldico_cohorts(cohort_desc, hemi="both", composed_desc={}):
    """ Create all used cohorts based on several available databases.  """

    hemis = ["L", "R"] if hemi == "both" else [hemi]

    all_cohortes = []
    for h in hemis:
        cohorts = {}
        for cname, desc in cohort_desc.items():
            try:
                cohort = bv_cohort(cname, desc['path'], h,
                                   centers=desc["centers"],
                                   graph_v=desc['graph_v'],
                                   ngraph_v=desc['ngraph_v'],
                                   acquisition=desc['acquisition'],
                                   session=desc['session'],
                                   inclusion=desc['inclusion'],
                                   exclusion=desc['exclusion'],
                                   )
                cohorts[cname] = cohort
                all_cohortes.append(cohort)

                print("{}: {} subjects".format(cohort.name, len(cohort)))
            except IOError as exc:
                print(cname, "is unavailable")
                print("\tError: ", exc)

        for cname, desc in composed_desc.items():
            cohort = Cohort(cname + "_hemi-" + h, subjects=[])
            do_not_add = False
            for cname2 in desc.keys():
                if cname2 not in cohorts.keys():
                    print("{} is unavailable (need {})".format(cname,
                                                               cname2))
                    do_not_add = True
                    break

                if len(desc[cname2]["indexes"]) == 0:
                    cohort = cohort.concatenate(cohorts[cname2])
                else:
                    for subi in desc[cname2]["indexes"]:
                        if subi > len(cohorts[cname2]):
                            print("{} is unavailable (not enough subject in {})"
                                  .format(cname, cname2))
                            do_not_add = True
                            break
                        else:
                            cohort.subjects.append(
                                cohorts[cname2].subjects[subi])
                    if do_not_add:
                        break

            if not do_not_add:
                cohorts[cname] = cohort
                all_cohortes.append(cohort)
                print("{}: {} subjects".format(cohort.name, len(cohort)))

    return all_cohortes


def main():
    parser = argparse.ArgumentParser(description='Create cohorts files (.json)')
    parser.add_argument('-e', dest='env', type=str, default=None, help="Configuration file")
    args = parser.parse_args()

    # Load environnment file
    env_f = args.env if args.env else op.join(op.split(__file__)[0], "env.json")
    env = json.load(open(env_f))

    cohorts_dir = op.join(env['working_path'], "cohorts")
    makedirs(cohorts_dir, exist_ok=True)
    print("Cohorts will be saved to:", cohorts_dir)

    # Create all cohorts for both hemispheres
    cohorts = foldico_cohorts(env['cohorts'],
                              composed_desc=env['composed_cohorts'])

    for cohort in cohorts:
        fname = "cohort-" + cohort.name + ".json"
        cohort.to_json(op.join(cohorts_dir, fname))


if __name__ == "__main__":
    main()
