# -*- coding: UTF-8 -*-
import warnings
import numpy as np
import flask
from flask_cors import CORS
import argparse
import spacy
import time
import os
import json
import utils.converters
from collections import OrderedDict


nlp = spacy.load('en')

parser = argparse.ArgumentParser(description='Mummer NLU server')
parser.add_argument('-o', '--output', type=str, default='json', help='Output format [json|token|hrc]')
parser.add_argument('-n', '--network', type=str, default='hermit',
                    help='Network to load')
parser.add_argument('-d', '--debug', action='store_true', help='Store the output as *.hrc2 for debugging')
parser.add_argument('-p', '--port', type=int, default=7785, help='Listening port')
parser.add_argument('--run-folder', type=str, help='Run folder path')
parser.add_argument('-s', '--syntax-only', action='store_true', help='Tag only syntax')
parser.add_argument('--gpu', action='store_true', help='Use GPU')
args = parser.parse_args()

if not args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
if not args.syntax_only:
    from learning.network import KerasNetwork
    import utils.loaders
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

if args.debug:
    debug_folder = os.path.join('debug', str(int(time.time())))
    if not os.path.exists(debug_folder):
        os.makedirs(debug_folder)
    with open(os.path.join(debug_folder, 'config.txt'), 'w') as f:
        f.write(str(args))
        f.close()

network = args.network

run_folder = args.run_folder

port = args.port

nets = OrderedDict()
label_encoders = OrderedDict()

labels = ['dialogue_act', 'frame', 'frame_element']

if not args.syntax_only:
    for label in labels:
        label_encoders[label] = utils.loaders.load_label_encoder(os.path.join(run_folder,
                                                                              'encoders',
                                                                              label + '_labels.npy'))
    net = KerasNetwork(network=network, label_encoders=label_encoders)
    net.load_model(os.path.join(run_folder, 'models', network, 'architecture.json'),
                   os.path.join(run_folder, 'models', network, 'weights.h5'))

app = flask.Flask(__name__)
CORS(app)


@app.route('/ack', methods=['GET', 'POST'])
def ack():
    request = dict()
    if flask.request.method == 'POST':
        request = flask.request.get_json()
        request['status'] = 'OK'
    return flask.jsonify(request)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    annotation = dict()
    if flask.request.method == 'POST':
        json_request = flask.request.get_json()
        sentence = json_request['sentence']
        print(sentence)
        doc = nlp(sentence)
        annotation['sentence'] = sentence
        annotation['tokens'] = []
        predictions = dict()
        if not args.syntax_only:
            sentence_array, feature_vector = utils.converters.spacy_to_features(doc)
            predictions = dict()
            prediction = net.model.predict(feature_vector)
            for i, label in enumerate(labels):
                prediction_idx = np.argmax(prediction[i], -1)[0]
                prediction_labels = label_encoders[label].inverse_transform(prediction_idx)
                predictions[label] = prediction_labels.tolist()
        for i, t in enumerate(doc):
            token = dict()
            token['index'] = i + 1
            token['word'] = t.text
            token['lemma'] = t.lemma_
            token['pos'] = t.pos_
            if not args.syntax_only:
                for label in predictions:
                    token[label] = predictions.get(label)[i]
            if t.ent_type_:
                token['ner'] = t.ent_iob_ + '-' + t.ent_type_
            else:
                token['ner'] = 'O'
            annotation['tokens'].append(token)
        print(annotation)

    example_id = str(int(time.time()))
    if args.debug:
        hrc = utils.converters.annotation_to_hrc(example_id, annotation)
        with open(os.path.join(debug_folder, example_id + '.hrc2'), 'w') as f:
            f.write(hrc)
            f.close()
    if args.output == 'json':
        annotation = utils.converters.annotation_to_json(annotation)
    elif args.output == 'token':
        annotation = utils.converters.annotation_to_token(annotation)
    elif args.output == 'hrc':
        annotation = utils.converters.annotation_to_hrc(example_id, annotation)
    return json.dumps(annotation)


if __name__ == '__main__':
    print('* Loading Keras models and starting Flask server... please wait until server has fully started')
    app.run(host='0.0.0.0', port=port)
