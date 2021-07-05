from keras.models import Model
from keras.layers import Input
from keras.layers import concatenate
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.wrappers import Bidirectional
from learning.layers.embeddings import ElmoEmbedding
from learning.layers.crf import CRF
from learning.losses import crf_losses
from keras_self_attention import SeqSelfAttention


"""
    HERMIT network structure for DialogAct+FramePrediction+SRL.
"""


def hermit(hyper_params, embedding_matrices, labels):
    inputs = []
    embeddings_layers = []

    input_layer = Input(shape=(None,), dtype='string', name='word_input')
    inputs.append(input_layer)
    embedding_layer = ElmoEmbedding(output_dim=1024,
                                    trainable=False,
                                    name='word_embedding')(input_layer)
    embeddings_layers.append(embedding_layer)

    for embedding in embedding_matrices:
        input_layer = Input(shape=(None,), dtype='int32', name=embedding + '_input')
        inputs.append(input_layer)
        embedding_layer = Embedding(input_dim=embedding_matrices[embedding].shape[0],
                                    output_dim=embedding_matrices[embedding].shape[1],
                                    weights=[embedding_matrices[embedding]],
                                    trainable=False,
                                    mask_zero=True,
                                    name=embedding + '_embedding')(input_layer)
        embeddings_layers.append(embedding_layer)

    if len(embeddings_layers) > 1:
        input_embeddings = concatenate(embeddings_layers)
    else:
        input_embeddings = embeddings_layers[0]

    shared_network = Bidirectional(LSTM(units=hyper_params['units'],
                                        activation='tanh',
                                        return_sequences=True))(input_embeddings)

    shared_network, dialog_att_w = SeqSelfAttention(attention_activation=hyper_params['attention_activation'],
                                                    attention_width=hyper_params['attention_width'],
                                                    attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                                                    return_attention=True)(shared_network)

    shared_network1 = Dropout(rate=hyper_params['dropout'])(shared_network)
    output1 = TimeDistributed(Dense(units=labels['domain'],
                                    activation='relu'))(shared_network1)
    crf_domain = CRF(labels['domain'], name='domain')
    output1 = crf_domain(output1)

    shared_network = concatenate([input_embeddings, shared_network])

    shared_network = Bidirectional(LSTM(units=hyper_params['units'],
                                        activation='tanh',
                                        return_sequences=True))(shared_network)

    shared_network, dialog_att_w = SeqSelfAttention(attention_activation=hyper_params['attention_activation'],
                                                    attention_width=hyper_params['attention_width'],
                                                    attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                                                    return_attention=True)(shared_network)

    shared_network2 = Dropout(rate=hyper_params['dropout'])(shared_network)
    output2 = TimeDistributed(Dense(units=labels['frame'],
                                    activation='relu'))(shared_network2)
    crf_fr = CRF(labels['frame'], name='frame')
    output2 = crf_fr(output2)

    shared_network = concatenate([input_embeddings, shared_network])

    shared_network = Bidirectional(LSTM(units=hyper_params['units'],
                                        activation='tanh',
                                        return_sequences=True))(shared_network)

    shared_network3 = Dropout(rate=hyper_params['dropout'])(shared_network)
    output3 = TimeDistributed(Dense(units=labels['frame_element'],
                                    activation='relu'))(shared_network3)
    crf_fe = CRF(labels['frame_element'], name='frame_element')
    output3 = crf_fe(output3)
    model = Model(inputs=inputs, outputs=[output1, output2, output3])
    model.compile(optimizer=hyper_params['optimizer'], loss=crf_losses.crf_loss)
    return model
