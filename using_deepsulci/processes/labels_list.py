'''
    Labels list
'''

from __future__ import print_function
from __future__ import absolute_import
import traits.api as traits
import numpy as np
import sigraph
import os
import pandas as pd
from capsul.api import Process
from soma import aims
from deepsulci.deeptools.dataset import extract_data


class SulciList(Process):
    '''
    Process to list given and missing labels

    **Warning:** Graphs should be of the same side!

    '''

    def __init__(self):
        super(SulciList, self).__init__()
        self.add_trait('graphs', traits.List(traits.File(output=False),
                                             desc='training base graphs'))
        self.add_trait('graphs_notcut', traits.List(
            traits.File(output=False),
            desc='training base graphs before manual cutting of the'
                 ' elementary folds'))
        self.add_trait('translation_file', traits.File(
            output=False, optional=True,
            desc='file (.trl) containing the translation of the sulci to'
                 'applied on the training base graphs (optional)'))

        self.add_trait('report', traits.File(
            output=True, desc='file (.csv) stats'))

    def _run_process(self):
        agraphs = np.asarray(self.graphs)
        agraphs_notcut = np.asarray(self.graphs_notcut)

        if os.path.exists(self.translation_file):
            flt = sigraph.FoldLabelsTranslator()
            flt.readLabels(self.translation_file)
            trfile = self.translation_file
        else:
            flt, trfile = None, None
            print('Translation file not found.')

        # Read all graphs
        sulci_side_list = set()
        dict_bck2 = {}
        dict_names = {}
        for gfile in agraphs:
            print("Reading", gfile)
            graph = aims.read(gfile)
            if trfile is not None:
                flt.translate(graph)

            print("Extracting sulci data...")
            data = extract_data(graph)
            dict_bck2[gfile] = data['bck2']
            dict_names[gfile] = data['names']
            for n in data['names']:
                sulci_side_list.add(n)
            print(len(dict_names[gfile]), "names")
            print(len(dict_bck2[gfile]), "buckets")

        out_data = {"label": []}
        for fname in agraphs:
            out_data[fname] = []

        for name in sulci_side_list:
            for fname in agraphs:
                out_data[fname] = np.sum(dict_names[fname] == name)

        df = pd.DataFrame(out_data)
        print(df.head(5))
