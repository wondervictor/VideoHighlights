# -*- coding: utf-8 -*-

import paddle.v2 as paddle


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


input_data = paddle.layer.data(name='data', type=paddle.data_type.dense_vector_sequence(2048))
output = model(input_data)

