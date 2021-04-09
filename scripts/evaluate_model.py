from using_deepsulci.processes.classification_evaluation import DeepClassificationEvaluation
import os.path as op
from os import listdir

TRL_FILE = '/casa/host/build/share/brainvisa-share-5.0/nomenclature/' \
           'translation/sulci_model_2018.trl'


def list_cohorts(data_dir, hemi):
    # List subjects for archi database
    archi = []
    for f in listdir(op.join(data_dir, "archi", "t1-1mm-1")):
        if len(f) == 3 and f != '058':
            archi.append(f)
    archi = sorted(archi)

    archi_graphs = []
    for s in archi:
        archi_graphs.append(
            data_dir + '/archi/t1-1mm-1/' + s + '/t1mri/default_acquisition/'
            'default_analysis/folds/3.3/session1_manual/' +
            hemi + s + '_session1_manual.arg'
        )

    # List subjects for archi database
    pclean = []
    pclean_dirs = []
    for d in ['jumeaux', 'nmr', 'panabase']:
        for f in listdir(op.join(data_dir, "data", "database_learnclean", d)):
            if op.isdir(f):
                pclean.append(f)
                pclean_dirs.append(d)
    pclean = sorted(pclean)

    pclean_graphs = []
    for i, s in enumerate(pclean):
        pclean_graphs.append(
            data_dir + '/deepsulci_learning/' + pclean_dirs[i] + '/' + s +
            '/t1mri/t1/default_analysis/folds/3.3/base2018_manual/' +
            hemi + s + '_base2018_manual.arg'
        )

    # Centralize all databases in a single dict
    cohorts = {
        "archi": {"subjects": archi, "graphs": archi_graphs},
        "pclean": {"subjects": pclean, "graphs": pclean_graphs},
        "140s": {
            "subjects": archi + pclean,
            "graphs": archi_graphs + pclean_graphs
        }
    }

    return cohorts


def evaluate_model(cohort, graph_tmpl, model_file, param_file, out_file):
    graphs = []
    for s in cohort:
        graphs.append(graph_tmpl.replace("$sub", s))
    print(len(graphs), "graphs to evaluate")

    proc = DeepClassificationEvaluation()
    proc.graphs = graphs
    proc.translation_file = TRL_FILE
    proc.model_file = model_file
    proc.param_file = param_file
    proc.cuda = -1
    proc.out_file = out_file
    proc._run_process()


def main():
    data_dir = '/home/bastien/data'
    out_dir = '/home/bastien/data/models_performances'

    # Filename templates
    graph_tmpl = data_dir + '/archi/t1-1mm-1/$sub/t1mri/default_acquisition/' \
                            'default_analysis/folds/3.3/session1_manual/' \
                            'L$sub_session1_manual.arg'
    # roots_tmpl = data_dir + '/archi/t1-1mm-1/$sub/t1mri/default_acquisition/' \
    #                         'default_analysis/segmentation/Lroots_$sub.nii.gz'
    # skeleton_tmpl = data_dir + '/archi/t1-1mm-1/$sub/t1mri/default_acquisition/' \
    #                         'default_analysis/segmentation/Lskeleton_$sub.nii.gz'
    # labeled_tmpl = op.join(data_dir, 'test_cnn_models/$model/$sub_left.arg')

    cohorts = list_cohorts(data_dir)

    cohort = "archi"

    evaluate_model(
        cohorts[cohort],
        graph_tmpl,
        data_dir + '/models/sulci_unet_model_left_archi.mdsm',
        data_dir + '/models/sulci_unet_model_params_left_archi.json',
        out_dir + '/model-archi_hemi-left_teston-' + cohort + '.tsv'
    )


if __name__ == "__main__":
    main()
