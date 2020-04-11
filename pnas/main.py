from nasbench import api

from pnas.pnas import PNAS


def main():
    nasbench = api.NASBench('/Users/hua/Documents/datasets/nasbench_full.tfrecord')

    max_num_blocks = 5
    ops = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
    num_ops = len(ops)
    top_k = 256

    lstm_units = 100
    embed_size = 100

    pnas = PNAS(
        nasbench,
        max_num_blocks,
        ops,
        top_k,
        lstm_units,
        embed_size,
        epochs=50,
        batch_size=16
    )

    pnas.train()

    return pnas.get_top_architectures(num_blocks=5, top_k=10)
