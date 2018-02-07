# -*- coding: utf-8 -*-

import json
import pickle
import numpy as np


def load_video(video_id, path='/home/ubuntu/data/training/image/'):
    with open(path + video_id + '.pkl', 'rb') as f:
        data = pickle.load(f)
    video_length = data.shape[0]
    return data, video_length


def load_labels(json_path='/home/ubuntu/data/mete.json'):
    with open(json_path, 'r') as f:
        json_content = json.load(f)
    return json_content


def load_train_video(video_id, labels):
    data, video_length = load_video(video_id)
    annotations = labels['database'][video_id]['annotations']
    points = [x['segment'] for x in annotations]
    label = np.zeros([video_length])
    for point in points:
        label[int(point[0]):int(point[1])] = 1
    return data, label, video_length


def generate_segments(video, label, window_size):

    stride = window_size / 3
    thresh = window_size / 2 + 1
    segments = []  # (data, label_forward, label_backward)
    video_length = len(video)
    num_segments = (video_length - window_size + stride) / stride
    start = 0
    for i in xrange(num_segments):
        segment = video[start: start + window_size]
        current_label = label[start: start + window_size]
        start += stride

        if np.sum(current_label) > thresh:
            current_label = 1
        else:
            current_label = 0

        segments.append((segment, current_label))
    return segments


def load_segments(filepath):

    f = open(filepath, 'rb')
    segements = pickle.load(f)
    f.close()
    return segements

