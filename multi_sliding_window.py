# -*- coding: utf -*-

import sys
import json
import random
import pickle
import argparse
import numpy as np
import paddle.v2 as paddle
from data_utils import *
from rnn_model import *


def create_reader(window_size, is_train):
    with open('data.txt', 'r') as f:
        file_names = f.readlines()
    if is_train:
        file_names = random.sample(file_names, 300)
    else:
        file_names = random.sample(file_names, 100)
    file_names = [x.rstrip('\n\r').replace('.pkl', '') for x in file_names]
    labels = load_labels()
    all_data = []
    for name in file_names:
        data, label, video_length = load_train_video(name, labels)
        segments = generate_segments(data, label, window_size)
        all_data += segments
    print("Finish Loading Data")

    def reader():
        random.shuffle(all_data)
        for i in xrange(len(all_data)):
            yield np.array(all_data[i][0]), int(all_data[i][1])

    return reader


sum_cost = 0.0


def train(window_size, epoches):
    paddle.init(use_gpu=False, trainer_count=2)
    input_label = paddle.layer.data(name='label', type=paddle.data_type.integer_value(2))
    cost = paddle.layer.cross_entropy_cost(input=output, label=input_label)
    train_reader = create_reader(window_size, True)
    test_reader = create_reader(window_size, False)
    train_reader = paddle.batch(reader=paddle.reader.shuffle(train_reader, buf_size=256), batch_size=256)
    feeding = {'data': 0, 'label': 1}
    adam_optimizer = paddle.optimizer.Adam(learning_rate=1e-4,
                                           regularization=paddle.optimizer.L2Regularization(rate=8e-4),
                                           model_average=paddle.optimizer.ModelAverage(average_window=0.5))

    parameters = paddle.parameters.create(cost)
    trainer = paddle.trainer.SGD(cost=cost, parameters=parameters, update_equation=adam_optimizer)

    def event_handler(event):
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
            result = trainer.test(paddle.batch(reader=test_reader, batch_size=128), feeding=feeding)
            print("\nPass: %s COST: %s" % (event.pass_id, result.cost))

    trainer.train(reader=train_reader, event_handler=event_handler, feeding=feeding, num_passes=epoches)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--windows_size', type=int, default=10, help='Window Size')
    parser.add_argument('--epoches', type=int, default=10, help='training epoches')

    args = parser.parse_args()

    train(args.window_size, args.epoches)

