import numpy as np

from pnas.predictor import get_predictor
from pnas.utils import (
    get_accuracy, generate_architectures as _generate_architectures
)


def generate_architectures(*args, **kwargs):
    '''
    Remove architecutres where all 5 nodes are connected to the input node
    '''
    architectures = _generate_architectures(*args, **kwargs)
    for arc in architectures:
        if len(arc) == 5 and sum(x[0] for x in arc) == 0:
            continue
        else:
            yield arc


class PNAS():
    def __init__(
        self,
        nasbench,
        max_num_blocks=5,
        ops=None,
        top_k=256,
        lstm_units=100,
        embed_size=100,
        epochs=50,
        batch_size=16
    ):
        self.nasbench = nasbench
        self.max_num_blocks = max_num_blocks
        if ops:
            self.ops = ops
        else:
            self.ops =['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
        self.num_ops = len(self.ops)
        self.top_k = top_k
        
        self.predictor = get_predictor(max_num_blocks, self.num_ops, lstm_units, embed_size)
        self.predictor.compile(optimizer='adam', loss='mean_squared_error')
        
        self.epochs = epochs
        self.batch_size = batch_size
        
    def get_top_architectures(self, num_blocks, top_k, keep_accuracy=True):
        architectures = np.array([*generate_architectures(num_blocks, self.num_ops)])
        pred_accuracies = self.predictor.predict(architectures)
        pairs = sorted(zip(pred_accuracies, architectures), key=lambda x: x[0][0], reverse=True)
        
        if keep_accuracy:
            return pairs[:top_k]
        else:
            architectures = np.array([*zip(*pairs[:top_k])][1])
            
            return architectures
        
    def train_n_blocks(self, num_blocks, use_top_k=False):
        if use_top_k:
            architectures = self.get_top_architectures(num_blocks, self.top_k, keep_accuracy=False)
        else:
            architectures = np.array([*generate_architectures(num_blocks, self.num_ops)])
            
        accuracies = np.array([get_accuracy(self.nasbench, arc, self.ops) for arc in architectures])
        
        self.predictor.fit(architectures, accuracies, epochs=self.epochs, batch_size=self.batch_size)
        
    def train(self):
        for i in range(self.max_num_blocks):
            print(f'Start to train the predictor with architectures having {i + 1} blocks')
            self.train_n_blocks(i + 1, use_top_k=i != 0)
            
    def predict(self, *args, **kwargs):
        return self.predictor.predict(*args, **kwargs)
    