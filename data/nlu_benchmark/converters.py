import pandas as pd
import re
import xml.etree.ElementTree as ET
import spacy
import os
import xml.dom.minidom as minidom
from progress.bar import Bar
import json
import argparse

tagger = spacy.load('en_core_web_sm', parser=False)

arg_format_pattern = r"\[\s*(?P<label>[\w']*)\s*:(?P<filler>[\s\w'\.@\-&+]+)\]"
arg_annotation_pattern = r"\[\s*[\w']*\s*:[\s\w'\.@\-&+]+\]"


def load_json_prediction_file(predictions_file):
    _, filename = os.path.split(predictions_file)
    print("Loading {}..".format(filename))
    with open(predictions_file, "r") as f:
        json_prediction = json.load(f)
        f.close()
    return json_prediction


def squeeze_prediction_span(json_prediction):
    squeezed_predictions = []
    for example in json_prediction:
        new_example = dict()
        frame_pred_set = set()
        domain_pred_set = set()
        intent_pred_set = set()
        frame_gold_set = set()
        domain_gold_set = set()
        intent_gold_set = set()
        entities_gold = []
        entities_pred = []
        current_frame_element_gold = ''
        current_frame_element_pred = ''
        for frame_token, domain_token in zip(example['frame_gold'], example['domain_gold']):
            frame_gold_set.add(frame_token[2:])
            domain_gold_set.add(domain_token[2:])
            intent_gold_set.add(domain_token[2:] + '_' + frame_token[2:])
        for frame_token, domain_token in zip(example['frame_pred'], example['domain_pred']):
            frame_pred_set.add(frame_token[2:])
            domain_pred_set.add(domain_token[2:])
            intent_pred_set.add(domain_token[2:] + '_' + frame_token[2:])
        for frame_element_token, token in zip(example['frame_element_gold'], example['tokens']):
            if frame_element_token == "O":
                continue
            if frame_element_token.startswith("B-"):
                entity_gold = dict()
                entities_gold.append(entity_gold)
                current_frame_element_gold = frame_element_token[2:]
                entity_gold[current_frame_element_gold] = [token]
            else:
                if frame_element_token[2:] == current_frame_element_gold:
                    entity_gold[current_frame_element_gold].append(token)
                else:
                    entity_gold = dict()
                    entities_gold.append(entity_gold)
                    current_frame_element_gold = frame_element_token[2:]
                    entity_gold[current_frame_element_gold] = [token]
        for frame_element_token, token in zip(example['frame_element_pred'], example['tokens']):
            if frame_element_token == "O":
                continue
            if frame_element_token.startswith("B-"):
                entity_pred = dict()
                entities_pred.append(entity_pred)
                current_frame_element_pred = frame_element_token[2:]
                entity_pred[current_frame_element_pred] = [token]
            else:
                if frame_element_token[2:] == current_frame_element_pred:
                    entity_pred[current_frame_element_pred].append(token)
                else:
                    entity_pred = dict()
                    entities_pred.append(entity_pred)
                    current_frame_element_pred = frame_element_token[2:]
                    entity_pred[current_frame_element_pred] = [token]

        new_example['tokens'] = example['tokens']
        new_example['domain_gold'] = list(domain_gold_set)
        new_example['domain_pred'] = list(domain_pred_set)
        new_example['frame_gold'] = list(frame_gold_set)
        new_example['frame_pred'] = list(frame_pred_set)
        new_example['intent_gold'] = list(intent_gold_set)
        new_example['intent_pred'] = list(intent_pred_set)
        new_example['entities_gold'] = entities_gold
        new_example['entities_pred'] = entities_pred
        squeezed_predictions.append(new_example)
    return squeezed_predictions


def nlub_to_hrc2(nlub_path, output_dir, verbose=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data = pd.read_csv(nlub_path, sep=";")
    if verbose:
        bar = Bar('Loading from {}'.format(nlub_path), max=data.shape[0])
    skipped = 0
    for index, row in data.iterrows():
        if verbose:
            bar.next()
        if str(row["answer_annotation"]) != "nan":
            scenario = row["scenario"]
            intent = row["intent"]
            sentence = row["answer_annotation"]
            frame_elements, plain_sentence = extrapolate_frame_elements(sentence)
            if '[' in plain_sentence or ']' in plain_sentence:
                print("\n\n" + plain_sentence + "\n\n")
            if "userid" in row:
                ann_id = "{}_u{}_a{}".format(index, row["userid"], row["answerid"])
            else:
                ann_id = "{}_a{}".format(index, row["answerid"])

            generate_annotation_file(plain_sentence, scenario, intent,
                                     frame_elements, ann_id, output_dir)
        else:
            skipped += 1
    if verbose:
        bar.finish()
    print("{} skipped sentences".format(skipped))


def extrapolate_frame_elements(sentence):
    annotations = re.findall(arg_annotation_pattern, sentence)
    replaced_sentence = sentence
    frame_elements = []
    if "[" in sentence:
        for annotation in annotations:
            argument = re.search(arg_format_pattern, annotation)
            label = argument.groupdict()["label"].strip()
            filler = argument.groupdict()["filler"].strip()
            index = replaced_sentence.index(annotation)
            filler_length = len(filler)
            span = [index, index+filler_length]
            replaced_sentence = replaced_sentence.replace(annotation, filler, 1)
            frame_elements.append({"frame_element": label, "filler": filler, "span": span})

    return frame_elements, replaced_sentence


def generate_annotation_file(sentence, domain, frame,
                             frame_elements, ann_id, output_dir):

    def set_all_attrib_from_dict(xml_element, attr_dict):
        for k, v in attr_dict.iteritems():
            xml_element.set(k, v)

    example_xml = ET.Element("example")
    set_all_attrib_from_dict(example_xml, {"id": ann_id})
    sent_xml = ET.SubElement(example_xml, "sentence")
    sent_xml.text = sentence
    tokens_xml = ET.SubElement(example_xml, "tokens")
    semantics_xml = ET.SubElement(example_xml, "semantics")
    ners_xml = ET.SubElement(semantics_xml, "ner")
    dialogacts_xml = ET.SubElement(semantics_xml, "domain")
    frames_xml = ET.SubElement(semantics_xml, "frame")
    frame_elements_xml = ET.SubElement(frames_xml, "frameElement")

    sentence_ann = tagger(unicode(sentence, "utf-8"))
    tokens = []

    for i, token in enumerate(sentence_ann):
        tokens.append(token.text)

        # grammar tokens
        token_attrib = {"surface": token.text, "id": str(i+1), "lemma": token.lemma_, "pos": token.pos_}
        token_xml = ET.SubElement(tokens_xml, "token")
        set_all_attrib_from_dict(token_xml, token_attrib)

        # named entities
        if token.ent_type_:
            ner_ann = "{}-{}".format(token.ent_iob_, token.ent_type_)
            ner_token_xml = ET.SubElement(ners_xml, "token")
            set_all_attrib_from_dict(ner_token_xml, {"id": str(i+1), "value": ner_ann})

        # dialogue act
        iob = "B-" if i == 0 else "I-"
        domain_ann = "{}{}".format(iob, domain)
        domain_token_xml = ET.SubElement(dialogacts_xml, "token")
        set_all_attrib_from_dict(domain_token_xml, {"id": str(i+1), "value": domain_ann})

        # frames
        frame_ann = "{}{}".format(iob, frame)
        frame_token_xml = ET.SubElement(frames_xml, "token")
        set_all_attrib_from_dict(frame_token_xml, {"id": str(i+1), "value": frame_ann})

    for fe in frame_elements:
        span = fe["span"]
        token_span = get_tokens_from_indexes(sentence, tokens, span[0], span[1])
        for i, t in enumerate(token_span):
            iob = "B-" if i == 0 else "I-"
            fe_ann = "{}{}".format(iob, fe["frame_element"])
            fe_token_xml = ET.SubElement(frame_elements_xml, "token")
            set_all_attrib_from_dict(fe_token_xml, {"id": str(t+1), "value": fe_ann})

    final_path = os.path.join(output_dir, ann_id+".hrc2")
    xmlstr = minidom.parseString(ET.tostring(example_xml)).toprettyxml(indent="   ", encoding='UTF-8')
    with open(final_path, "w") as f:
        f.write(xmlstr)
        f.close()
        del example_xml


def get_tokens_from_indexes(sent, tokens, start, end):
    """ it works only for frame elements, as there are
        some errors in tagging POS-tags"""
    count = 0
    token_span = []

    for i in range(0, len(tokens)):
        cur_token = tokens[i]
        token_start = count
        token_end = token_start + len(cur_token)

        if token_start == start:
            token_span.append(i)
        elif token_end == end:
            token_span.append(i)
        elif token_start > start and token_end < end:
            token_span.append(i)
        elif start < token_start < end < token_end:
            token_span.append(i)
        elif token_start < start and token_start < end < token_end:
            token_span.append(i)
        count += len(cur_token)
        if sent[token_end:token_end + 1] == " ":
            count += 1
    return token_span


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NLU Benchmark to hrc2 converter')
    parser.add_argument('-i', '--input', type=str, default=None,
                        help='Input file')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output dir')
    args = parser.parse_args()
    nlub_to_hrc2(args.input, args.output)
