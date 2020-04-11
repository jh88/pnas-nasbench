from keras import Model, layers


def get_predictor(max_num_blocks, num_ops, lstm_units=100, embed_size=100):
    inputs = layers.Input(shape=(None, 2), name='inputs')

    idx_inputs = layers.Lambda(lambda x: x[:, :, 0], name='idx_inputs')(inputs)
    op_inputs = layers.Lambda(lambda x: x[:, :, 1], name='op_inputs')(inputs)

    idx_x = layers.Embedding(
        max_num_blocks, embed_size, name='idx_embed'
    )(idx_inputs)
    op_x = layers.Embedding(num_ops, embed_size, name='op_embed')(op_inputs)

    x = layers.concatenate([idx_x, op_x], axis=-1)
    x = layers.LSTM(lstm_units, name='lstm')(x)
    x = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=x, name='predictor')

    return model


# class Predictor(Model):
#     def __init__(
#         self, lstm_units=100, embed_size=100, num_blocks=5, num_ops=3, **kwargs
#     ):
#         super().__init__(**kwargs)

#         self.idx_embed = layers.Embedding(
#             num_blocks, embed_size, name='idx_embed'
#         )
#         self.op_embed = layers.Embedding(num_ops, embed_size, name='op_embed')

#         self.lstm = layers.LSTM(lstm_units, name='lstm')
#         self.dense = layers.Dense(1, activation='sigmoid', name='output')

#     def call(self, inputs):
#         idx_inputs = inputs[:, :, 0]
#         op_inputs = inputs[:, :, 1]

#         embed_idx = self.idx_embed(idx_inputs)
#         embed_op = self.op_embed(op_inputs)

#         x = layers.concatenate([embed_idx, embed_op], axis=-1)
#         x = self.lstm(x)
#         x = self.dense(x)
 
#         return x
