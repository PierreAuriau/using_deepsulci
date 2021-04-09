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
from using_deepsulci.cohort import bv_cohort, Cohort


def foldico_cohorts(data_dir, hemi):
    """ Create all used cohortes based on several available databases.  """
    cohorts = {}

    try:
        pclean = bv_cohort("PClean", op.join(data_dir, "deepsulci_learning"),
                           hemi, centers=['nmr', 'jumeaux', 'panabase'],
                           ngraph_v="3.1", acquisition='t1',
                           session='base2018_manual')
        cohorts["PClean"] = pclean
    except IOError as exc:
        print("PClean is unavailable")
        print("\tError: ", exc)

    try:
        archi = bv_cohort("Archi", op.join(data_dir, "archi"), hemi, "t1-1mm-1",
                          ngraph_v=-1, session='session1_manual')
        cohorts["Archi"] = archi
    except IOError as exc:
        print("Archi database is unavailable")
        print("\tError: ", exc)

    try:
        hcp = bv_cohort("HCP", op.join(data_dir, "hcp"), hemi, "t1-1mm-1",
                        graph_v="3.1", session='default_session_auto')
        cohorts["HCP"] = hcp
    except IOError as exc:
        print("HCP database is unavailable")
        print("\tError: ", exc)

    if "PClean" in cohorts.keys() and "Archi" in cohorts.keys():
        cohorts["140s"] = pclean.concatenate(archi, new_name="140s")

        cohorts["p30a30"] = Cohort(
            "p30a30", pclean.subjects[:30] + archi.subjects[:30])

    if "PClean" in cohorts.keys() and "Archi" in cohorts.keys() and \
        "HCP" in cohorts.keys():
        cohorts["216s"] = pclean.concatenate(archi).concatenate(hcp, "216s")

    return cohorts


def main():
    # Load environnment file
    env = json.load(open(op.join(op.split(__file__)[0], "env.json")))

    cohorts_dir = op.join(env['working_path'], "cohorts")
    makedirs(cohorts_dir, exist_ok=True)
    print("Cohorts will be saved to:", cohorts_dir)

    # Create all cohorts for both hemispheres
    for h in ["L", "R"]:
        cohortes = foldico_cohorts(env['bv_databases_path'], h)

        for cohort in cohortes.keys():
            c = cohortes[cohort]
            print("{}: {} subjects".format(c.name, len(c)))
            fname = "cohort-" + c.name + ".json"
            c.to_json(op.join(cohorts_dir, fname))


if __name__ == "__main__":
    main()
