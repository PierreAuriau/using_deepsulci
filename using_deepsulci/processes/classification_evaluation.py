import json
import traits.api as traits
import numpy as np
from capsul.api import Process
from deepsulci.sulci_labeling.method.unet import UnetSulciLabeling
from using_deepsulci.utils.io import load_graphs

import pandas as pd

from deepsulci.sulci_labeling.analyse.stats import esi_score, bacc_score, acc_score


class DeepClassificationEvaluation(Process):
    '''
    **Warning:** Graphs should be of the same side!

    '''

    def __init__(self):
        super(DeepClassificationEvaluation, self).__init__()
        self.add_trait('graphs', traits.List(traits.File(output=False),
                                             desc='training base graphs'))
        self.add_trait('translation_file', traits.File(
            output=False, optional=True,
            desc='file (.trl) containing the translation of the sulci to'
                 'applied on the training base graphs (optional)'))
        self.add_trait('model_file', traits.File(
            output=False, desc='Trainned model (.mdsm)'))
        self.add_trait('param_file', traits.File(
            output=False, desc='file (.json) storing the hyperparameters'
                               ' (cutting threshold)'))
        self.add_trait('out_file', traits.File(
            output=False, desc='file (.tsv)'))
        self.add_trait('cuda', traits.Int(
            -1, output=False, desc='device on which to run the training'
                                   '(-1 for cpu, i>=0 for the i-th gpu)'))

    def _run_process(self):
        # Load graphs
        dict_bck2, dict_names = load_graphs(self.graphs, self.translation_file,
                                            verbosity=1)

        # Load params
        with open(self.param_file) as f:
            param = json.load(f)
        self.sulci_side_list = param['sulci_side_list']

        print("Name list:")
        print(self.sulci_side_list)
        print()

        # Load the model
        method = UnetSulciLabeling(
            self.sulci_side_list, num_filter=64, batch_size=1, cuda=self.cuda)
        method.load(self.model_file)

        # Labelize each graph
        esi_scores = []
        acc_scores = []
        results = {k: [] for k in ['graph', 'model', 'sulci', 'accuracy', 'esi']}
        glb_results = {k: [] for k in ['graph', 'model', 'balanced_accuracy', 'esi']}
        for ig, g in enumerate(dict_names.keys()):
            print("\nLabeling", g)
            y_true, y_pred, y_scores = method.labeling(self.graphs[ig])
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)

            for s in np.unique(y_true):
                results['graph'].append(g)
                results['model'].append(self.model_file)
                results['sulci'].append(s)
                sel = y_true == s
                results['accuracy'].append(
                    bacc_score(y_true[sel], y_pred[sel], np.unique(y_true)))
                results['esi'].append(
                    esi_score(y_true[sel], y_pred[sel], np.unique(y_true)))

            scr = bacc_score(y_true, y_pred, np.unique(y_true))
            esi = esi_score(y_true, y_pred, np.unique(y_true))
            glb_results['balanced_accuracy'].append(scr)
            glb_results['esi'].append(esi)

        np.save(self.out_file[:-4] + '_detail.npy', results)
        np.save(self.out_file, glb_results)
