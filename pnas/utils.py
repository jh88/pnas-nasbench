from nasbench import api


def generate_architectures(num_blocks, num_ops):
    if num_blocks < 1:
        yield []
    else:
        for idx in range(num_blocks):
            for op in range(num_ops):
                for arc in generate_architectures(num_blocks - 1, num_ops):
                    yield arc + [(idx, op)]


def get_accuracy(nasbench, arc, ops, key='test_accuracy'):
    matrix = [[0 for _ in range(len(arc) + 2)] for _ in range(len(arc) + 2)]
    _ops = ['input']
    for i, (idx, op) in enumerate(arc):
        matrix[idx][i + 1] = 1
        _ops.append(ops[op])
    
    _ops.append('output')
    
    for row in matrix[:-1]:
        if not sum(row):
            row[-1] = 1
    
    model_spec = api.ModelSpec(matrix=matrix, ops=_ops)
    data = nasbench.query(model_spec)
    
    return data[key]
