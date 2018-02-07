# -*- coding: utf-8 -*-

import sys
import json
import random
import pickle
import argparse
import numpy as np
import paddle.v2 as paddle
from data_utils import *
from rnn_model import *

with open('params.tar', 'r') as f:
    parameters = paddle.parameters.Parameters.from_tar(f)

with open('test.txt', 'r') as f:
    test_files_lines = f.readlines()
test_files_lines = [x.rstrip('\n\r') for x in test_files_lines]


def inference(video_id, parameters, window_size):
    video, video_length = load_video(video_id, '/home/ubuntu/data/testing/image/')

    segments = []
    num_segments = video_length / window_size
    start = 0
    for i in xrange(num_segments):
        segment = video[start: start + window_size]
        start += window_size
        segments.append((segment,))

    probs = paddle.infer(output_layer=output, parameters=parameters, input=segments, feeding={'data': 0})

    return probs


def aggregate(prediction, probs, window_size):
    result = []

    current_start = -1
    current_end = 0
    # 间隔空白
    blank_threshhold = 0
    # 最小长度
    min_lengths = 1

    state = 0
    # state = 0 未找到start
    # state = 1 找到start，未找到end, 当前值为1
    # state = 2 找到start，未找到end, 当前值为0
    # state = 3 中间隔了很多空白

    for i in xrange(prediction.shape[0]):
        if state == 0 and prediction[i] == 1:
            current_start = i
            state = 1
            continue

        if state == 1 and prediction[i] == 1:
            current_end = i
            continue

        if state == 1 and prediction[i] == 0:
            state = 2
            continue

        if state == 2 and prediction[i] == 1:

            distance = i - current_end
            if distance >= blank_threshhold:

                length = current_end - current_start + 1
                state = 0
                if length <= min_lengths:
                    current_start = -1
                    current_end = -1
                    continue
                prob = sum(probs[current_start: current_end + 1]) / length
                result.append(([current_start * window_size, current_end * window_size], prob))
                current_start = -1
                current_end = -1
                continue
            else:

                if distance < blank_threshhold / 2:
                    current_end = i
                    state = 1
                    continue
                else:
                    state = 3
                    continue

        if state == 3 and prediction[i] == 1:
            current_end = i
            state = 1
            continue

        if state == 3 and prediction[i] == 0:

            length = current_end - current_start + 1

            state = 0
            if length <= min_lengths:
                current_start = -1
                current_end = -1
                continue
            prob = sum(probs[current_start: current_end + 1]) / length
            result.append(([current_start * window_size, current_end * window_size], prob))

            current_start = -1
            current_end = -1
            continue

    if state == 1 and current_start != -1:
        current_end = len(prediction)
        length = current_end - current_start + 1
        if length <= min_lengths:
            current_start = -1
            current_end = -1
            return result
        prob = sum(probs[current_start: current_end + 1]) / length
        result.append(([current_start * window_size, current_end * window_size], prob))

    return result


def generate_result(probs, window_size):
    prediction = np.argmax(probs, axis=1)
    predict_right_prob = probs[:, 1]
    data = aggregate(prediction, predict_right_prob, window_size)
    return data


def generate_segments(probs, window_size):
    result = dict()
    for key in probs.keys():
        prrob = probs[key]
        data = generate_result(prrob, window_size)
        result[key] = data
    return result


def inference_all(window_size):
    infer_result = dict()
    counter = 0
    infer_result = dict()
    counter = 0
    for video_file in test_files_lines:
        video_id = video_file.split('.')[0]
        prob = inference(video_id, parameters, window_size)
        infer_result[video_id] = prob
        counter += 1
        print("[%s/%s]finish: %s" % (counter, len(test_files_lines), video_id))
    return infer_result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--window_size', type=int, default=10, help='Window size')

    args = parser.parse_args()
    window_size = args.window_size
    in_result = inference_all(window_size)
    segments = generate_segments(in_result, window_size)
    f = open('segments/size_%s_segment.pkl' % window_size, 'wb')
    pickle.dump(segments, f)
    f.close()

