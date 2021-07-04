# -*- coding: UTF-8 -*-
import numpy as np
import xml.etree.ElementTree as et
import xml.dom.minidom


def annotation_to_token(annotation):
    for token in annotation['tokens']:
        token['frameElement'] = token.get('frame_element', '')
        token['dialogueAct'] = token.get('dialogue_act', '')
        token.pop('frame_element', None)
        token.pop('dialogue_act', None)
    return annotation


def annotation_to_json(annotation):
    result = dict()
    result['syntax'] = []
    result['semantics'] = []
    result['named_entities'] = []
    dialogue_act_span = []
    frame_span = []
    frame_element_span = []

    current_ner = None
    current_annotation_name = None
    for i, token in enumerate(annotation['tokens']):
        # Compile syntax section
        current_token = dict()
        current_token['word'] = token['word']
        current_token['lemma'] = token['lemma']
        current_token['pos'] = token['pos']
        result['syntax'].append(current_token)

        # Compile named_entities section
        ner = token['ner']
        if ner != 'O':
            if ner.startswith('B-'):
                current_ner = dict()
                current_annotation_name = token['ner'].replace('B-', '')
                current_ner[current_annotation_name] = [token['word']]
                result['named_entities'].append(current_ner)
            elif ner.startswith('I-'):
                ner_name = token['ner'].replace('I-', '')
                if ner_name != current_annotation_name:
                    current_annotation_name = ner_name
                    current_ner = dict()
                    current_ner[current_annotation_name] = []
                    result['named_entities'].append(current_ner)
                current_ner[current_annotation_name].append(token['word'])

        ann = token['dialogue_act']
        if ann != 'O':
            if ann.startswith('B-'):
                current_annotation = [i]
                current_annotation_name = ann.replace('B-', '')
                dialogue_act_span.append(current_annotation)
            elif ann.startswith('I-'):
                annotation_name = ann.replace('I-', '')
                if annotation_name == current_annotation_name:
                    current_annotation.append(i)
                else:
                    success = False
                    for d in dialogue_act_span[::-1]:
                        if annotation['tokens'][d[0]]['dialogue_act'].split('-')[1] == annotation_name:
                            d.append(i)
                            success = True
                            break
                    if not success:
                        current_annotation = [i]
                        current_annotation_name = ann.replace('B-', '')
                        dialogue_act_span.append(current_annotation)

    for da in dialogue_act_span:
        for i in da:
            ann = annotation['tokens'][i]['frame']
            if ann != 'O':
                if ann.startswith('B-'):
                    current_annotation = [i]
                    current_annotation_name = ann.replace('B-', '')
                    frame_span.append(current_annotation)
                elif ann.startswith('I-'):
                    annotation_name = ann.replace('I-', '')
                    if annotation_name == current_annotation_name:
                        current_annotation.append(i)
                    else:
                        success = False
                        for d in da[::-1]:
                            try:
                                if annotation['tokens'][d]['frame'].split('-')[1] == annotation_name and d != i:
                                    for fr in frame_span[::-1]:
                                        if d in fr:
                                            fr.append(i)
                                            success = True
                                            break
                                    else:
                                        continue
                                    break
                            except IndexError:
                                continue
                        if not success:
                            current_annotation = [i]
                            current_annotation_name = ann.replace('B-', '')
                            frame_span.append(current_annotation)

    for fr in frame_span:
        for i in fr:
            ann = annotation['tokens'][i]['frame_element']
            if ann != 'O':
                if ann.startswith('B-'):
                    current_annotation = [i]
                    current_annotation_name = ann.replace('B-', '')
                    frame_element_span.append(current_annotation)
                elif ann.startswith('I-'):
                    annotation_name = ann.replace('I-', '')
                    if annotation_name == current_annotation_name:
                        current_annotation.append(i)
                    else:
                        success = False
                        for d in fr[::-1]:
                            try:
                                if annotation['tokens'][d]['frame_element'].split('-')[1] == annotation_name and d != i:
                                    for fe in frame_element_span[::-1]:
                                        if d in fe:
                                            fe.append(i)
                                            success = True
                                            break
                                    else:
                                        continue
                                    break
                            except IndexError:
                                continue
                        if not success:
                            current_annotation = [i]
                            current_annotation_name = ann.replace('B-', '')
                            frame_element_span.append(current_annotation)

    for da in dialogue_act_span:
        current_dialogue_act = dict()
        current_dialogue_act_name = annotation['tokens'][da[0]]['dialogue_act'].split('-')[1]
        current_dialogue_act[current_dialogue_act_name] = []
        result['semantics'].append(current_dialogue_act)
        for fr in frame_span:
            if any(elem in da for elem in fr):
                current_frame = dict()
                current_frame_name = annotation['tokens'][fr[0]]['frame'].split('-')[1]
                current_frame[current_frame_name] = []
                current_dialogue_act[current_dialogue_act_name].append(current_frame)
                for fe in frame_element_span:
                    if any(elem in fr for elem in fe):
                        current_frame_element = dict()
                        current_frame_element_name = annotation['tokens'][fe[0]]['frame_element'].split('-')[1]
                        current_frame_element[current_frame_element_name] = []
                        for f in fe:
                            current_frame_element[current_frame_element_name].append(annotation['tokens'][f]['word'])
                        current_frame[current_frame_name].append(current_frame_element)
    return result


def annotation_to_hrc(example_id, annotation):
    example = et.Element('example')
    example.set('id', example_id)
    sentence = et.SubElement(example, 'sentence')
    sentence.text = annotation['sentence']
    tokens = et.SubElement(example, 'tokens')
    semantics = et.SubElement(example, 'semantics')
    ner = et.SubElement(semantics, 'ner')
    dialogue_act = et.SubElement(semantics, 'dialogueAct')
    frame = et.SubElement(semantics, 'frame')
    frame_element = et.SubElement(frame, 'frameElement')
    for i, t in enumerate(annotation['tokens']):
        token = et.SubElement(tokens, 'token')
        token.attrib = {'id': str(t['index']), 'lemma': t['lemma'], 'pos': t['pos'], 'surface': t['word']}
        if t['ner'] != 'O':
            token_ner = et.SubElement(ner, 'token')
            token_ner.attrib = {'id': str(t['index']), 'value': t['ner']}
        if t['dialogue_act'] != 'O':
            token_da = et.SubElement(dialogue_act, 'token')
            token_da.attrib = {'id': str(t['index']), 'value': t['dialogue_act']}
        if t['frame'] != 'O':
            token_frm = et.SubElement(frame, 'token')
            token_frm.attrib = {'id': str(t['index']), 'value': t['frame']}
        if t['frame_element'] != 'O':
            token_fe = et.SubElement(frame_element, 'token')
            token_fe.attrib = {'id': str(t['index']), 'value': t['frame_element'].replace('B-', '').replace('I-', '')}
    rough_string = et.tostring(example, 'utf-8')
    re_parsed = xml.dom.minidom.parseString(rough_string)
    return re_parsed.toprettyxml(indent='  ')


def spacy_to_features(doc):
    sentence_array = []
    for token in doc:
        sentence_array.append(token.text)
    feature_vector = list()
    feature_vector.append(np.asarray(sentence_array)[np.newaxis, :])
    return sentence_array, feature_vector
