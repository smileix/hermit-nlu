# -*- coding: UTF-8 -*-
import xml.etree.ElementTree as et
import os
import numpy as np
from sklearn import preprocessing
from collections import OrderedDict
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from progress.bar import Bar


class Dataset(object):

    dataset = []
    train_set = []
    test_set = []
    embeddings = OrderedDict()
    labels = ['domain', 'frame', 'frame_element']
    feature_encoders = OrderedDict()
    label_encoders = OrderedDict()
    encoders_path = None
    max_sentence_length = 0
    bag_of_pos = ['AAA_PADDING', 'ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART',
                  'PRON',
                  'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', 'SPACE', 'O']
    bag_of_ner = ['AAA_PADDING', 'B-PERSON', 'I-PERSON', 'B-NORP', 'I-NORP', 'B-FAC', 'I-FAC', 'B-ORG', 'I-ORG', 'B-GPE',
                  'I-GPE', 'B-LOC', 'I-LOC', 'B-PRODUCT', 'I-PRODUCT', 'B-EVENT', 'I-EVENT', 'B-WORK_OF_ART',
                  'I-WORK_OF_ART', 'B-LAW', 'I-LAW', 'B-LANGUAGE', 'I-LANGUAGE', 'B-DATE', 'I-DATE', 'B-TIME',
                  'I-TIME', 'B-PERCENT', 'I-PERCENT', 'B-MONEY', 'I-MONEY', 'B-QUANTITY', 'I-QUANTITY', 'B-ORDINAL',
                  'I-ORDINAL', 'B-CARDINAL', 'I-CARDINAL', 'O']

    def __init__(self, dataset_path=None, train_set_path=None, test_set_path=None):
        if dataset_path is not None:
            for root, directories, file_names in os.walk(dataset_path):
                file_names = [fi for fi in file_names if fi.endswith(".hrc2")]
                if len(file_names) > 0:
                    bar = Bar('Loading {} dataset: '.format(os.path.basename(root)), max=len(file_names))
                    for filename in file_names:
                        bar.next()
                        huric_example = self.__import_example(os.path.join(root, filename))
                        if huric_example is not None:
                            self.dataset.append(huric_example)
                    del bar
                print('')
            print('\nDataset size: {} examples'.format(len(self.dataset)))
        else:
            for root, directories, file_names in os.walk(train_set_path):
                file_names = [fi for fi in file_names if fi.endswith(".hrc2")]
                if len(file_names) > 0:
                    bar = Bar('Loading {} dataset: '.format(os.path.basename(root)), max=len(file_names))
                    for filename in file_names:
                        bar.next()
                        huric_example = self.__import_example(os.path.join(root, filename))
                        if huric_example is not None:
                            self.train_set.append(huric_example)
                    del bar
                print('')
            for root, directories, file_names in os.walk(test_set_path):
                file_names = [fi for fi in file_names if fi.endswith(".hrc2")]
                if len(file_names) > 0:
                    bar = Bar('Loading {} dataset: '.format(os.path.basename(root)), max=len(file_names))
                    for filename in file_names:
                        bar.next()
                        huric_example = self.__import_example(os.path.join(root, filename))
                        if huric_example is not None:
                            self.test_set.append(huric_example)
                    del bar
                print('')
            print('\nDataset size:\n\tTrain set: {} examples\n\tTest set: {} examples'.format(len(self.train_set),
                                                                                            len(self.test_set)))

    def __import_example(self, input_file):
        try:
            huric_example_xml = et.parse(input_file)
        except et.ParseError:
            print('Problems on file: {}'.format(input_file))
            return None
        root = huric_example_xml.getroot()
        huric_example_id = root.attrib['id']
        huric_example = dict()
        huric_example['id'] = huric_example_id
        for sentence in root.findall('sentence'):
            huric_example['sentence'] = sentence.text.encode('utf-8')
        ids_array = []
        lemmas_array = []
        pos_array = []
        sentence = []
        for token in root.findall('./tokens/token'):
            token_id = token.attrib['id']
            ids_array.append(token_id)
            lemma = token.attrib['lemma']
            lemmas_array.append(lemma.encode('utf-8'))
            pos = token.attrib['pos']
            pos_array.append(pos)
            surface = token.attrib['surface']
            sentence.append(surface.encode('utf-8'))
        huric_example['index'] = np.asarray(ids_array)
        huric_example['lemma'] = np.asarray(lemmas_array)
        huric_example['pos'] = np.asarray(pos_array)
        huric_example['tokens'] = np.asarray(sentence)
        sentence_length = len(sentence)
        if sentence_length > self.max_sentence_length:
            self.max_sentence_length = sentence_length
        huric_example['sentence_length'] = sentence_length
        ner_annotations = np.full(sentence_length, fill_value='O', dtype='object')
        domain_annotations = np.full(sentence_length, fill_value='O', dtype='object')
        frame_annotations = np.full(sentence_length, fill_value='O', dtype='object')
        frame_element_annotations = np.full(sentence_length, fill_value='O', dtype='object')
        for domain in root.findall('./semantics/domain/token'):
            domain_annotations[int(domain.attrib['id']) - 1] = domain.attrib['value']
        for ner in root.findall('./semantics/ner/token'):
            ner_annotations[int(ner.attrib['id']) - 1] = ner.attrib['value']
        for frame in root.findall('./semantics/frame/token'):
            frame_annotations[int(frame.attrib['id']) - 1] = frame.attrib['value']
        for frame_element in root.findall('./semantics/frame/frameElement/token'):
            frame_element_annotations[int(frame_element.attrib['id']) - 1] = frame_element.attrib['value']
        huric_example['ner'] = ner_annotations
        huric_example['domain'] = domain_annotations
        huric_example['frame'] = frame_annotations
        huric_example['frame_element'] = frame_element_annotations
        return huric_example

    def generate_training_data(self, feature_spaces, run_folder):
        examples = []
        train_examples = []
        test_examples = []

        self.encoders_path = os.path.join('resources', run_folder, 'encoders')
        if not os.path.exists(self.encoders_path):
            os.makedirs(self.encoders_path)

        self.generate_feature_encoders(feature_spaces, save_encoders=True)
        self.generate_label_encoders(save_encoders=True)

        for sentence in self.dataset:
            example = []
            features = []
            label_vector = []
            sentence_array = sentence['tokens']

            tokens = np.full(self.max_sentence_length, fill_value='__PAD__', dtype='object')
            sentence_array = sentence_array.tolist()
            sentence_array.reverse()
            for i in range(self.max_sentence_length - len(sentence_array), self.max_sentence_length):
                tokens[i] = sentence_array.pop()
            tokens = np.asarray(tokens)
            features.append(tokens)

            for feature_space in feature_spaces:
                features.append(pad_sequences([self.feature_encoders[feature_space].transform(sentence[feature_space])],
                                              maxlen=self.max_sentence_length)[0])
            example.append(np.asarray(features))

            for label in self.labels:
                label_vector.append(to_categorical(pad_sequences([self.label_encoders[label].transform(sentence[
                                                                                                           label])],
                                                                 maxlen=self.max_sentence_length)[0],
                                                   num_classes=len(self.label_encoders[label].classes_)))
            example.append(label_vector)
            examples.append(example)

        for sentence in self.train_set:
            example = []
            features = []
            label_vector = []
            sentence_array = sentence['tokens']

            tokens = np.full(self.max_sentence_length, fill_value='__PAD__', dtype='object')
            sentence_array = sentence_array.tolist()
            sentence_array.reverse()
            for i in range(self.max_sentence_length - len(sentence_array), self.max_sentence_length):
                tokens[i] = sentence_array.pop()
            tokens = np.asarray(tokens)
            features.append(tokens)

            for feature_space in feature_spaces:
                features.append(pad_sequences([self.feature_encoders[feature_space].transform(sentence[feature_space])],
                                              maxlen=self.max_sentence_length)[0])
            example.append(np.asarray(features))

            for label in self.labels:
                label_vector.append(to_categorical(pad_sequences([self.label_encoders[label].transform(sentence[
                                                                                                           label])],
                                                                 maxlen=self.max_sentence_length)[0],
                                                   num_classes=len(self.label_encoders[label].classes_)))
            example.append(label_vector)
            train_examples.append(example)

        for sentence in self.test_set:
            example = []
            features = []
            label_vector = []
            sentence_array = sentence['tokens']

            tokens = np.full(self.max_sentence_length, fill_value='__PAD__', dtype='object')
            sentence_array = sentence_array.tolist()
            sentence_array.reverse()
            for i in range(self.max_sentence_length - len(sentence_array), self.max_sentence_length):
                tokens[i] = sentence_array.pop()
            tokens = np.asarray(tokens)
            features.append(tokens)

            for feature_space in feature_spaces:
                features.append(pad_sequences([self.feature_encoders[feature_space].transform(sentence[feature_space])],
                                              maxlen=self.max_sentence_length)[0])
            example.append(np.asarray(features))

            for label in self.labels:
                label_vector.append(to_categorical(pad_sequences([self.label_encoders[label].transform(sentence[
                                                                                                           label])],
                                                                 maxlen=self.max_sentence_length)[0],
                                                   num_classes=len(self.label_encoders[label].classes_)))
            example.append(label_vector)
            test_examples.append(example)

        if not self.dataset:
            return train_examples, test_examples, self.label_encoders
        else:
            return examples, self.label_encoders

    def print_sentences(self, out_file=None, include_id=False):
        if out_file:
            with open(out_file, 'w') as f:
                bar = Bar('Printing dataset: ', max=len(self.dataset))

                for i, example in enumerate(self.dataset):
                    bar.next()
                    if include_id:
                        f.write(example['id'] + '\t')
                    f.write(example['sentence'] + '\n')
        else:
            bar = Bar('Printing dataset: ', max=len(self.dataset))

            for i, example in enumerate(self.dataset):
                bar.next()
                if include_id:
                    print(example['id'] + '\t'),
                print(example['sentence'])

    def generate_feature_encoders(self, feature_spaces, save_encoders=False):
        for feature_space in feature_spaces:
            feature_encoder = preprocessing.LabelEncoder()

            if feature_space == 'pos':
                feature_encoder.fit(self.bag_of_pos)
            elif feature_space == 'ner':
                feature_encoder.fit(self.bag_of_ner)
            else:
                bag_of_features = set()
                for sentence in self.dataset:
                    for label in sentence[feature_space]:
                        bag_of_features.add(label)
                for sentence in self.train_set:
                    for label in sentence[feature_space]:
                        bag_of_features.add(label)
                for sentence in self.test_set:
                    for label in sentence[feature_space]:
                        bag_of_features.add(label)
                bag_of_features = list(bag_of_features)
                feature_encoder.fit(bag_of_features)

            self.feature_encoders[feature_space] = feature_encoder
            embedding = np.identity(len(feature_encoder.classes_), dtype='float32')
            self.embeddings[feature_space] = embedding
            if save_encoders:
                np.save(os.path.join(self.encoders_path, feature_space + '_labels.npy'), feature_encoder.classes_)

    def generate_label_encoders(self, save_encoders=False):
        for label in self.labels:
            bag_of_labels = set()
            bag_of_labels.add('AAA_PADDING')

            for sentence in self.dataset:
                for l in sentence[label]:
                    bag_of_labels.add(l)
            for sentence in self.train_set:
                for l in sentence[label]:
                    bag_of_labels.add(l)
            for sentence in self.test_set:
                for l in sentence[label]:
                    bag_of_labels.add(l)
            bag_of_labels = list(bag_of_labels)
            label_encoder = preprocessing.LabelEncoder()
            label_encoder.fit(bag_of_labels)

            self.label_encoders[label] = label_encoder
            if save_encoders:
                np.save(os.path.join(self.encoders_path, label + '_labels.npy'), label_encoder.classes_)

    def compute_random_baseline(self):
        from learning.metrics.sequence_labeling import classification_report
        np.random.seed(42)
        distribution = self.statistics()
        self.generate_label_encoders()
        domain_labels = self.label_encoders['domain'].classes_
        frame_labels = self.label_encoders['frame'].classes_
        frame_element_labels = self.label_encoders['frame_element'].classes_

        domain_labels = np.delete(domain_labels, np.argwhere(domain_labels == 'AAA_PADDING'))
        frame_labels = np.delete(frame_labels, np.argwhere(frame_labels == 'AAA_PADDING'))
        frame_element_labels = np.delete(frame_element_labels, np.argwhere(frame_element_labels == 'AAA_PADDING'))
        domain_labels = np.delete(domain_labels, np.argwhere(domain_labels == 'O'))
        frame_labels = np.delete(frame_labels, np.argwhere(frame_labels == 'O'))
        frame_element_labels = np.delete(frame_element_labels, np.argwhere(frame_element_labels == 'O'))

        domain_labels = list(set([label.replace('B-', '').replace('I-', '') for label in domain_labels]))
        frame_labels = list(set([label.replace('B-', '').replace('I-', '') for label in frame_labels]))
        frame_element_labels = list(set([label.replace('B-', '').replace('I-', '') for label in frame_element_labels]))

        domain_distribution = []
        frame_distribution = []
        frame_element_distribution = []
        for label in domain_labels:
            print(label, distribution['domain'][label])
            domain_distribution.append(distribution['domain'][label])
        for label in frame_labels:
            frame_distribution.append(distribution['frame'][label])
        for label in frame_element_labels:
            frame_element_distribution.append(distribution['frame_element'][label])

        np.random.shuffle(self.dataset)
        folds = [self.dataset[i::10] for i in range(10)]
        for i, fold in enumerate(folds):
            print('Iteration {}'.format(i))
            y = OrderedDict()
            y_true = []
            y_pred = []
            for example in fold:
                new_example = []
                for token in example['domain']:
                    if token.startswith('B-'):
                        current_label = token.split('-')[1]
                        random_label = np.random.choice(domain_labels, 1, domain_distribution)[0]
                        new_token = token.replace(current_label, random_label)
                    else:
                        new_token = token.replace(current_label, random_label)
                    new_example.append(new_token)
                y_true.append(example['domain'].tolist())
                y_pred.append(new_example)
            y['domain'] = (y_true, y_pred)
            y_true = []
            y_pred = []
            for example in fold:
                new_example = []
                for token in example['frame']:
                    if token.startswith('B-'):
                        current_label = token.split('-')[1]
                        random_label = np.random.choice(frame_labels, 1, frame_distribution)[0]
                        new_token = token.replace(current_label, random_label)
                    else:
                        new_token = token.replace(current_label, random_label)
                    new_example.append(new_token)
                y_true.append(example['frame'].tolist())
                y_pred.append(new_example)
            y['frame'] = (y_true, y_pred)
            y_true = []
            y_pred = []

            for example in fold:
                new_example = []
                for token in example['frame_element']:
                    if token.startswith('B-'):
                        current_label = token.split('-')[1]
                        random_label = np.random.choice(frame_element_labels, 1, frame_element_distribution)[0]
                        new_token = token.replace(current_label, random_label)
                    else:
                        new_token = token.replace(current_label, random_label)
                    new_example.append(new_token)
                y_true.append(example['frame_element'].tolist())
                y_pred.append(new_example)
            y['frame_element'] = (y_true, y_pred)
            print(classification_report(y))

    def statistics(self):
        avg_length_of_sentence = .0
        total_number_of_domain = 0
        total_number_of_frame = 0
        total_number_of_frame_element = 0
        domain_number = dict()
        frame_number = dict()
        frame_element_number = dict()
        lexical_unit_distribution = dict()
        domain_predicates = 0.
        frame_predicates = 0.
        frame_element_predicates = 0.
        for example in self.dataset:
            avg_length_of_sentence += len(example['tokens'])
            for i, token in enumerate(example['tokens']):
                if example['domain'][i].startswith('B-'):
                    total_number_of_domain += 1
                    name = example['domain'][i].split('-')[1]
                    if name not in domain_number:
                        domain_number[name] = 1
                    else:
                        domain_number[name] += 1
                    domain_predicates += 1
                if example['frame'][i].startswith('B-'):
                    total_number_of_frame += 1
                    name = example['frame'][i].split('-')[1]
                    if name not in frame_number:
                        frame_number[name] = 1
                    else:
                        frame_number[name] += 1
                    frame_predicates += 1
                if example['frame_element'][i].startswith('B-'):
                    total_number_of_frame_element += 1
                    name = example['frame_element'][i].split('-')[1]
                    if name not in frame_element_number:
                        frame_element_number[name] = 1
                    else:
                        frame_element_number[name] += 1
                    frame_element_predicates += 1
                    if name == 'Lexical_unit':
                        index = example['index'][i]
                        if index not in lexical_unit_distribution:
                            lexical_unit_distribution[index] = 1
                        else:
                            lexical_unit_distribution[index] += 1
        avg_length_of_sentence = avg_length_of_sentence / len(self.dataset)
        print('Number of sentences:\t\t\t{}'.format(len(self.dataset)))
        print('Average length of sentence:\t\t{}'.format(round(avg_length_of_sentence, 2)))
        print('Dialogue act label set:\t\t\t{}'.format(len(domain_number)))
        print('Frame label set:\t\t\t\t{}'.format(len(frame_number)))
        print('Frame element label set:\t\t{}'.format(len(frame_element_number)))
        print('Total number of dialogue act:\t{}'.format(total_number_of_domain))
        print('Total number of frame:\t\t\t{}'.format(total_number_of_frame))
        print('Total number of frame element:\t{}'.format(total_number_of_frame_element))
        print('Average dialogue act/sentence:\t{}'.format(
            round(float(total_number_of_domain) / len(self.dataset), 2)))
        print('Average frame/sentence:\t\t\t{}'.format(
            round(float(total_number_of_frame) / len(self.dataset), 2)))
        print('Average frame element/sentence:\t{}'.format(
            round(float(total_number_of_frame_element) / len(self.dataset), 2)))
        print('Average frame/dialogue act:\t\t{}'.format(
            round(float(total_number_of_frame) / float(total_number_of_domain), 2)))
        print('Average frame element/frame:\t{}'.format(
            round(float(total_number_of_frame_element) / float(total_number_of_frame), 2)))
        print('Lexical unit distribution:\t\t{}'.format(sorted(lexical_unit_distribution.items(), key=lambda x: x[1],
                                                               reverse=True)))

        distribution = OrderedDict()
        domain_distribution = OrderedDict()
        frame_distribution = OrderedDict()
        frame_element_distribution = OrderedDict()
        distribution['domain'] = domain_distribution
        distribution['frame'] = frame_distribution
        distribution['frame_element'] = frame_element_distribution

        print('Dialogue act distribution:\t\t{}'.format(
            sorted(domain_number.items(), key=lambda x: x[1], reverse=True)))
        for label, number in domain_number.items():
            domain_distribution[label] = float(number) / domain_predicates
        print('Frame distribution:\t\t\t\t{}'.format(
            sorted(frame_number.items(), key=lambda x: x[1], reverse=True)))
        for label, number in frame_number.items():
            frame_distribution[label] = float(number) / frame_predicates
        print('Frame element distribution:\t\t{}'.format(
            sorted(frame_element_number.items(), key=lambda x: x[1], reverse=True)))
        for label, number in frame_element_number.items():
            frame_element_distribution[label] = float(number) / frame_element_predicates

        return distribution

    def statistics_benchmark(self):
        avg_length_of_sentence = .0
        total_number_of_domain = 0
        total_number_of_frame = 0
        total_number_of_intent = 0
        total_number_of_frame_element = 0
        domain_number = dict()
        frame_number = dict()
        frame_element_number = dict()
        intent_number = dict()
        lexical_unit_distribution = dict()
        domain_predicates = 0.
        frame_predicates = 0.
        frame_element_predicates = 0.
        intent_predicates = 0.
        for example in self.dataset:
            avg_length_of_sentence += len(example['tokens'])
            for i, token in enumerate(example['tokens']):
                domain_check = False
                frame_check = False
                if example['domain'][i].startswith('B-'):
                    domain_check = True
                    total_number_of_domain += 1
                    name = example['domain'][i][2:]
                    if name not in domain_number:
                        domain_number[name] = 1
                    else:
                        domain_number[name] += 1
                    domain_predicates += 1
                if example['frame'][i].startswith('B-'):
                    frame_check = True
                    total_number_of_frame += 1
                    name = example['frame'][i][2:]
                    if name not in frame_number:
                        frame_number[name] = 1
                    else:
                        frame_number[name] += 1
                    frame_predicates += 1
                if domain_check and frame_check:
                    name = example['domain'][i][2:] + '_' + example['frame'][i][2:]
                    total_number_of_intent += 1
                    if name not in intent_number:
                        intent_number[name] = 1
                    else:
                        intent_number[name] += 1
                    intent_predicates += 1
                if example['frame_element'][i].startswith('B-'):
                    total_number_of_frame_element += 1
                    name = example['frame_element'][i][2:]
                    if name not in frame_element_number:
                        frame_element_number[name] = 1
                    else:
                        frame_element_number[name] += 1
                    frame_element_predicates += 1
                    if name == 'Lexical_unit':
                        index = example['index'][i]
                        if index not in lexical_unit_distribution:
                            lexical_unit_distribution[index] = 1
                        else:
                            lexical_unit_distribution[index] += 1
        avg_length_of_sentence = avg_length_of_sentence / len(self.dataset)
        print('Number of sentences:\t\t\t{}'.format(len(self.dataset)))
        print('Average length of sentence:\t\t{}'.format(round(avg_length_of_sentence, 2)))
        print('Dialogue act label set:\t\t\t{}'.format(len(domain_number)))
        print('Frame label set:\t\t\t\t{}'.format(len(frame_number)))
        print('Frame element label set:\t\t{}'.format(len(frame_element_number)))
        print('Intent label set:\t\t\t\t{}'.format(len(intent_number)))
        print('Total number of dialogue act:\t{}'.format(total_number_of_domain))
        print('Total number of frame:\t\t\t{}'.format(total_number_of_frame))
        print('Total number of intent:\t\t\t{}'.format(total_number_of_intent))
        print('Total number of frame element:\t{}'.format(total_number_of_frame_element))
        print('Average dialogue act/sentence:\t{}'.format(
            round(float(total_number_of_domain) / len(self.dataset), 2)))
        print('Average frame/sentence:\t\t\t{}'.format(
            round(float(total_number_of_frame) / len(self.dataset), 2)))
        print('Average intent/sentence:\t\t\t{}'.format(
            round(float(total_number_of_intent) / len(self.dataset), 2)))
        print('Average frame element/sentence:\t{}'.format(
            round(float(total_number_of_frame_element) / len(self.dataset), 2)))
        print('Average frame/dialogue act:\t\t{}'.format(
            round(float(total_number_of_frame) / float(total_number_of_domain), 2)))
        print('Average frame element/frame:\t{}'.format(
            round(float(total_number_of_frame_element) / float(total_number_of_frame), 2)))
        print('Average frame element/intent:\t{}'.format(
            round(float(total_number_of_frame_element) / float(total_number_of_intent), 2)))
        print(
            'Lexical unit distribution:\t\t{}'.format(sorted(lexical_unit_distribution.items(), key=lambda x: x[1],
                                                             reverse=True)))

        distribution = OrderedDict()
        domain_distribution = OrderedDict()
        frame_distribution = OrderedDict()
        frame_element_distribution = OrderedDict()
        distribution['domain'] = domain_distribution
        distribution['frame'] = frame_distribution
        distribution['frame_element'] = frame_element_distribution

        print('Dialogue act distribution:\t\t{}'.format(
            sorted(domain_number.items(), key=lambda x: x[1], reverse=True)))
        for label, number in domain_number.items():
            domain_distribution[label] = float(number) / domain_predicates
        print('Frame distribution:\t\t\t\t{}'.format(
            sorted(frame_number.items(), key=lambda x: x[1], reverse=True)))
        for label, number in frame_number.items():
            frame_distribution[label] = float(number) / frame_predicates
        print('Frame element distribution:\t\t{}'.format(
            sorted(frame_element_number.items(), key=lambda x: x[1], reverse=True)))
        for label, number in frame_element_number.items():
            frame_element_distribution[label] = float(number) / frame_element_predicates

        return distribution