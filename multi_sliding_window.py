# -*- coding: utf -*-

import sys
import json
import random
import pickle
import argparse
import numpy as np
import paddle.v2 as paddle

window_size = 15
directory = ''


# DATA
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


def generate_segments(video, label, stride=5):

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

        # forward_label = np.zeros([window_size])
        #         backward_label = np.zeros([window_size])
        #         k = 0
        #         while k < window_size and current_label[k] != 0:
        #             k += 1
        #             forward_label[k] = 1

        #         k = window_size-1
        #         while k >= 0 and current_label[k] != 0:
        #             k = k - 1
        #             backward_label[k] = 1
        segments.append((segment, current_label))
    return segments


def create_reader():
    with open('data.txt', 'r') as f:
        file_names = f.readlines()
    file_names = [x.rstrip('\n\r').replace('.pkl', '') for x in file_names]
    labels = load_labels()
    all_data = []
    for name in file_names:
        data, label, video_length = load_train_video(name, labels)
        segments = generate_segments(data, label)
        all_data += segments
    print("Finish Loading Data")

    def reader():
        random.shuffle(all_data)
        for i in xrange(len(all_data)):
            yield np.array(all_data[i][0]), int(all_data[i][1])

    return reader


def model(batch_input_seq):
    fc_1 = paddle.layer.fc(
        input=batch_input_seq,
        size=1024,
        act=paddle.activation.Relu()
    )

    bidirectional_gru = paddle.networks.bidirectional_gru(
        input=fc_1,
        size=512,
        return_seq=True
    )

    fc_2 = paddle.layer.fc(
        input=bidirectional_gru,
        size=512,
        act=paddle.activation.Relu()
    )

    gru2 = paddle.networks.bidirectional_gru(
        input=fc_2,
        size=256,
    )
    output = paddle.layer.fc(
        input=gru2,
        size=2,
        act=paddle.activation.Softmax()
    )
    return output


paddle.init(use_gpu=False, trainer_count=2)
input_data = paddle.layer.data(name='data', type=paddle.data_type.dense_vector_sequence(2048))
output = model(input_data)


def train():

    input_label = paddle.layer.data(name='label', type=paddle.data_type.integer_value(2))
    cost = paddle.layer.cross_entropy_cost(input=output, label=input_label)
    train_reader = create_reader()
    train_reader = paddle.batch(reader=paddle.reader.shuffle(train_reader, buf_size=256), batch_size=256)
    feeding = {'data': 0, 'label': 1}
    adam_optimizer = paddle.optimizer.Adam(learning_rate=1e-4,
                                           regularization=paddle.optimizer.L2Regularization(rate=8e-4),
                                           model_average=paddle.optimizer.ModelAverage(average_window=0.5))

    parameters = paddle.parameters.create(cost)
    trainer = paddle.trainer.SGD(cost=cost, parameters=parameters, update_equation=adam_optimizer)

    def event_handler(event):
        global step
        global sum_cost

        if isinstance(event, paddle.event.EndIteration):
            sum_cost += event.cost
            if event.batch_id % 50 == 0:
                print "\nPass %d, Batch %d, AVG_COST: %f" % (
                    event.pass_id, event.batch_id, sum_cost / (event.batch_id + 1))
            else:
                sys.stdout.write('.')
                sys.stdout.flush()
        if isinstance(event, paddle.event.EndPass):
            sum_cost = 0.0
            with open('./params/params_%s.tar' % event.pass_id, 'w') as f:
                parameters.to_tar(f)
        step += 1
    trainer.train(reader=train_reader, event_handler=event_handler, feeding=feeding, num_passes=10)

