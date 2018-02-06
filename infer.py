# -*- coding: utf-8 -*-

import sys
import json
import random
import pickle
import argparse
import numpy as np
import paddle.v2 as paddle

import multi_sliding_window as msw

window_size = 15

output = msw.output

with open('params.tar', 'r') as f:
    parameters = paddle.parameters.Parameters.from_tar(f)

with open('test.txt', 'r') as f:
    test_files_lines = f.readlines()
test_files_lines = [x.rstrip('\n\r') for x in test_files_lines]


def inference(video_id, parameters):
    video, video_length = msw.load_video(video_id, '/home/ubuntu/data/testing/image/')

    segments = []
    num_segments = video_length / window_size
    start = 0
    for i in xrange(num_segments):
        segment = video[start: start + window_size]
        start += window_size
        segments.append((segment,))

    probs = paddle.infer(output_layer=output, parameters=parameters, input=segments, feeding={'data': 0})

    return probs


def inference_all():
    infer_result = dict()
    counter = 0
    infer_result = dict()
    counter = 0
    for video_file in test_files_lines:
        video_id = video_file.split('.')[0]
        prob = inference(video_id, parameters)
        infer_result[video_id] = prob
        counter += 1
        print("[%s/%s]finish: %s" % (counter, len(test_files_lines), video_id))
    return infer_result


in_result = inference_all()
f = open('size_%s_segment.pkl' % window_size, 'wb')
pickle.dump(in_result, f)
f.close()
