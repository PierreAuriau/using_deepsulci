import json
import os.path as op
from os import makedirs
import sys
import numpy as np


def summary_infos(result_f):
    # Load results
    data = np.load(result_f, allow_pickle=True).item()
    data = {k: np.array(data[k]) for k in data.keys()}

    # Get average by label
    ulabels = np.unique(data['label'])
    avg_bacc_label, std_bacc_label = {}, {}
    avg_esi_label, std_esi_label = {}, {}
    for lbl in ulabels:
        sel = data['label'] == lbl
        avg_bacc_label[lbl] = np.mean(data['balanced_accuracy'][sel])
        std_bacc_label[lbl] = np.std(data['balanced_accuracy'][sel])
        avg_esi_label[lbl] = np.mean(data['esi'][sel])
        std_esi_label[lbl] = np.std(data['esi'][sel])

    # Get average by graph
    ugraphs = np.unique(data['graph'])
    avg_bacc_graph, std_bacc_graph = {}, {}
    avg_esi_graph, std_esi_graph = {}, {}
    for g in ugraphs:
        sel = data['graph'] == g
        avg_bacc_graph[g] = np.mean(data['balanced_accuracy'][sel])
        std_bacc_graph[g] = np.std(data['balanced_accuracy'][sel])
        avg_esi_graph[g] = np.mean(data['esi'][sel])
        std_esi_graph[g] = np.std(data['esi'][sel])

    # Overall metrics
    avg_bacc = np.mean(data['balanced_accuracy'])
    avg_esi = np.mean(data['esi'])

    return data, {
        'overall_balanced_accuracy': avg_bacc,
        'overall_esi': avg_esi,
        'avarage_balanced_accuracy_by_labels': avg_bacc_label,
        'deviation_balanced_accuracy_by_labels': std_bacc_label,
        'avarage_esi_by_labels': avg_esi_label,
        'deviation_esi_by_labels': std_esi_label,
        'avarage_balanced_accuracy_by_graph': avg_bacc_graph,
        'deviation_balanced_accuracy_by_graph': std_bacc_graph,
        'avarage_esi_by_graph': avg_esi_graph,
        'deviation_esi_by_graph': std_esi_graph,
    }


def html_report(raw_data, data, modelname, cohortname):
    title = "CNN Sucli Recognition Results"
    n_subs = len(np.unique(raw_data['graph']))
    n_labels = len(np.unique(raw_data['label']))

    html = "<html><head><title>" + title + "</title>"
    html += '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">'
    html += '</head><body><h1>' + title + '</h1><table>'
    html += '<tr><th>Model:</th><td>' + modelname + '</td></tr>'
    html += '<tr><th>Test on cohort:</th><td>' + cohortname + '</td></tr>'
    html += '<tr><th>Number of subjects:</th><td>' + str(n_subs) + '</td></tr>'
    html += '<tr><th>Number of labels:</th><td>' + str(n_labels) + '</td></tr>'
    html += '<tr><th>Overall accuracy:</th><td>{:.01f}%</td></tr>'.format(
        data['overall_balanced_accuracy'] * 100)
    html += '<tr><th>Overall ESI:</th><td>{:.01f}%</td></tr>'.format(
        data['overall_esi'] * 100)
    html += '</table>'

    html += '<h2>Metrics by labels</h2>'
    html += '<table class="table"><thead><tr>'
    html += '<th>Label</th><th>Accuracy</th><th>ESI</th></tr></thead><tbody>'
    for lbl in sorted(data['avarage_balanced_accuracy_by_labels'].keys()):
        avg_acc = data['avarage_balanced_accuracy_by_labels'][lbl]
        std_acc = data['deviation_balanced_accuracy_by_labels'][lbl]
        avg_esi = data['avarage_esi_by_labels'][lbl]
        std_esi = data['deviation_esi_by_labels'][lbl]
        html += '<tr><th>' + str(lbl) + '</th><td>{:.01f}% (+/- {:.01f}%)</td>'.\
            format(avg_acc * 100, std_acc * 100)
        html += '<td>{:.01f}% (+/- {:.01f}%)</td><tr>'.\
            format(avg_esi * 100, std_esi * 100)
    html += '</tbody></table>'

    html += '<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js" ' \
            'integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+' \
            'IbbVYUew+OrCXaRkfj" crossorigin="anonymous"></script>'
    html += '<script src="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/' \
            'js/bootstrap.bundle.min.js" integrity="sha384-ho+j7jyWK8fNQe+A12' \
            'Hb8AhRq26LrZ/JpcUGGOn+Y7RsweNrtN/tE3MoK7ZeZDyx" crossorigin="ano' \
            'nymous"></script>'
    html += "</body>"
    return html


def main():
    env = json.load(open(op.join(op.split(__file__)[0], "env.json")))

    modelname = sys.argv[1]
    cohortname = sys.argv[2]

    print("Evaluate:", modelname)

    eval_dir = op.join(env['working_path'], "evaluations", modelname)
    fname = modelname + "_teston-" + cohortname + ".npy"

    raw, data = summary_infos(op.join(eval_dir, fname))
    report = html_report(raw, data, modelname, cohortname)
    with open(op.join(eval_dir, fname[:-4] + ".html"), 'w+') as f:
        f.write(report)
    print("HTML report: " + op.join(eval_dir, fname[:-4] + ".html"))


if __name__ == "__main__":
    main()
