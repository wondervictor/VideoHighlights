# -*- coding: utf-8 -*-

import json
import pickle
import numpy as np
from data_utils import *


def nms_detections(props, scores, overlap=0.7):
    # Code Borrowed from SST
    """Non-maximum suppression: Greedily select high-scoring detections and
    skip detections that are significantly covered by a previously selected
    detection. This version is translated from Matlab code by Tomasz
    Malisiewicz, who sped up Pedro Felzenszwalb's code.
    Parameters
    ----------
    props : ndarray
        Two-dimensional array of shape (num_props, 2), containing the start and
        end boundaries of the temporal proposals.
    scores : ndarray
        One-dimensional array of shape (num_props,), containing the corresponding
        scores for each detection above.
    Returns
    -------
    nms_props, nms_scores : ndarrays
        Arrays with the same number of dimensions as the original input, but
        with only the proposals selected after non-maximum suppression.
    """
    t1 = props[:, 0]
    t2 = props[:, 1]
    ind = np.argsort(scores)
    area = (t2 - t1 + 1).astype(float)
    pick = []
    while len(ind) > 0:
        i = ind[-1]
        pick.append(i)
        ind = ind[:-1]
        tt1 = np.maximum(t1[i], t1[ind])
        tt2 = np.minimum(t2[i], t2[ind])
        wh = np.maximum(0., tt2 - tt1 + 1.0)
        o = wh / (area[i] + area[ind] - wh)
        ind = ind[np.nonzero(o <= overlap)[0]]
    nms_props, nms_scores = props[pick, :], scores[pick]
    return nms_props, nms_scores


def process_single(segments):

    props = []
    scores = []

    for tp in segments:
        props.append(tp[0])
        scores.append(tp[1])

    props = np.array(props)
    scores = np.array(scores)
    props, scores = nms_detections(props, scores)
    result = zip(props, scores)
    return result


def generate_result_file(data, filepath, version='VERSION 1.0'):
    result = dict()
    result['version'] = version

    results = dict()

    for key in data.keys():

        tmp_result = []
        for seg in data[key]:
            tmp_dict = dict()
            tmp_dict['score'] = seg[1]
            tmp_dict['segment'] = seg[0]
            tmp_result.append(tmp_dict)
        results[key] = tmp_result

    result['results'] = results

    r = json.dumps(result)
    print(r)
    with open(filepath, 'w+') as f:
        f.write(r)
    print("generate finished")


if __name__ == '__main__':

    segments1 = load_segments('')
    segments2 = load_segments('')
    segments3 = load_segments('')

    result_segments = {}
    for key in segments1.keys():
        segments = segments1[key]
        segments += segments2[key]
        segments += segments3[key]
        result = process_single(segments)
        result_segments[key] = result

    generate_result_file(result_segments, 'output/result.json', '0.1')


