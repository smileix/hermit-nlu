import xml.etree.ElementTree as Et
from collections import defaultdict
import os
import json
import ntpath
from progress.bar import Bar
import spacy

FRAMENET_NAMESPACES = {"framenet":"http://framenet.icsi.berkeley.edu"}
NON_POS_TAGGED_TOKENS = ["`", "\""]
POS_SPECIAL = "$#$"
SECTIONS = ["huric", "mummer", "alexa", "nlu-benchmark"]

class SSPLoader:
    """
        loads the data from the Diafram dataset for frame prediction only
        frame labels are assigned to the whole span
        covered by the frame
    """
    # TODO this may generate problems when we use the multi-frame approach
    def __init__(self):
        self.stats = {
            "# sentences": 0,
            "# words": 0,
            "# frames": 0,
            "# arguments": 0,
            "# named entities": 0,
            "# dialogue acts": 0,
            "# frame types": 0,
            "# argument types": 0,
            "# dialogue act types": 0,
            "# named enetity types": 0
        }
        super(SSPLoader, self).__init__()
        self.tagger = spacy.load('en_core_web_sm', parser=False, entity=False)
        self.dataset = []

    def get_frame_types(self):
        frames = set()
        for example in self.dataset:
            for frame in example["frame_semantics"]:
                frames.add(frame)
        return list(frames)

    def get_arg_types(self):
        args = set()
        for example in self.dataset:
            for frame in example["frame_semantics"]:
                for arg in frame["frame_elements"]:
                    args.add(arg["frame_element"])
        return list(args)

    def get_ner_types(self):
        nes = set()
        for example in self.dataset:
            for ne in example["ner"]:
                nes.add(ne["ne"])
        return list(nes)

    def get_da_types(self):
        das = set()
        for example in self.dataset:
            for da in example["dialogue_acts"]:
                das.add(da["dialogue_act"])
        return list(das)

    def get_stats(self):
        stats = {
            "# sentences": 0,
            "# words": 0,
            "# frames": 0,
            "# arguments": 0,
            "# named entities": 0,
            "# dialogue acts": 0,
            "# frame types": 0,
            "# argument types": 0,
            "# dialogue act types": 0,
            "# named enetity types": 0
        }

        for e in self.dataset:
            stats["# sentences"] += 1
            stats["# words"] += len(e["tokens"])
            stats["# frames"] += len(e["frame_semantics"])
            stats["# arguments"] += len([a for f in e["frame_semantics"] for a in f["frame_elements"]])
            stats["# named entities"] += len(e["ner"])
            stats["# dialogue acts"] += len(e["dialogue_acts"])

        stats["# frame types"] = len(self.get_arg_types())
        stats["# argument types"] = len(self.get_arg_types())
        stats["# dialogue act types"] = len(self.get_ner_types())
        stats["# named enetity types"] = len(self.get_da_types())

        return stats

    def print_stats(self):
        stats = self.get_stats()

        for k, v in stats.items():
            print("{}: {}".format(k, v))

    def get_advanced_stats(self):
        stats = self.get_stats()

        # todo probably with a counter it even better
        adv_stats = defaultdict(lambda: defaultdict(lambda: 0))

        for e in self.dataset:
            for f in e["frame_semantics"]:
                adv_stats["frame types"][f["frame"]] += 1
                for a in f["frame_elements"]:
                    adv_stats["arguments types"][a["frame_element"]] += 1
            for ne in e["ner"]:
                adv_stats["named entity types"][ne["ne"]] += 1
            for da in e["dialogue_acts"]:
                adv_stats["dialogue act types"][da["dialogue_act"]] += 1

        stats.update(dict(adv_stats))
        return stats

    def print_advanced_stats(self):
        stats = self.get_advanced_stats()
        for k, v in stats.items():
            if type(v) == defaultdict:
                print(k)
                for k1, v1 in v.items():
                    print("{}: {}".format(k1, v1))
            else:
                print("{}: {}".format(k, v))

    ############################################
    #              diafram section            #

    @staticmethod
    def load_hrc2(hrc2_file):
        """
        loads all the frame semantics information
        :param hrc2_file: the complete path to the hrc2 file
        :return:
        """
        print(hrc2_file)
        hrc2 = defaultdict(lambda: 0)
        #hrc2["name"] = ntpath.basename(hrc2_file)
        hrc2["name"] = hrc2_file  # saving the whole path for easy loading
        xml = Et.parse(hrc2_file)
        root = xml.getroot()
        example_id = root.attrib["id"]
        hrc2["id"] = example_id

        xml_sentences = root.findall("./sentence")
        for xml_sentence in xml_sentences:
            sentence = xml_sentence.text
            hrc2["sentence"] = sentence

        tokens = list()
        xml_tokens = root.findall("./tokens/token")
        for xml_token in xml_tokens:
            token = defaultdict()
            token["id"] = int(xml_token.attrib["id"])
            token["lemma"] = xml_token.attrib["lemma"]
            token["surface"] = xml_token.attrib["surface"]
            token["pos"] = xml_token.attrib["pos"]
            tokens.append(token)
        hrc2["tokens"] = tokens

        # named entities
        hrc2["ner"] = list()
        ne_tokens = root.findall("./semantics/ner/")
        nes = defaultdict(lambda: 0)
        # probably for named entities this is even too paranoid
        for ne_token in ne_tokens:
            cur_ne = ne_token.attrib["value"]
            cur_ne_name = cur_ne[2:]
            cur_ne_iob = cur_ne[:2]
            cur_token_id = int(ne_token.attrib["id"])

            if cur_ne_name not in nes.keys():
                nes[cur_ne_name] = list()

            cur_ne_array = nes[cur_ne_name]

            if cur_ne_iob == "B-":
                new_ne = {"ne": cur_ne_name, "tokens": [cur_token_id]}
                cur_ne_array.append(new_ne)
            elif cur_ne_iob == "I-":
                ne = cur_ne_array[-1]
                ne["tokens"].append(cur_token_id)
        hrc2["ner"].extend([n for ne in nes.values() for n in ne])

        # dialogues acts
        hrc2["dialogue_acts"] = list()
        da_tokens = root.findall("./semantics/dialogueAct/")
        das = defaultdict(lambda: 0)
        # probably for named entities this is even too paranoid
        for da_token in da_tokens:
            cur_da = da_token.attrib["value"]
            cur_da_name = cur_da[2:]
            cur_da_iob = cur_da[:2]
            cur_token_id = int(da_token.attrib["id"])

            if cur_da_name not in das.keys():
                das[cur_da_name] = list()

            cur_da_array = das[cur_da_name]

            if cur_da_iob == "B-":
                new_da = {"dialogue_act": cur_da_name, "tokens": [cur_token_id]}
                cur_da_array.append(new_da)
            elif cur_da_iob == "I-":
                da = cur_da_array[-1]
                da["tokens"].append(cur_token_id)
        hrc2["dialogue_acts"].extend([d for da in das.values() for d in da])

        # frame semantics
        hrc2["frame_semantics"] = list()
        xml_frame_tokens = root.findall("./semantics/frame/token")
        frames = defaultdict(lambda: 0)
        for xml_token in xml_frame_tokens:
            cur_frame = xml_token.attrib["value"]
            cur_frame_name = cur_frame[2:]
            cur_frame_iob = cur_frame[:2]
            cur_token_id = int(xml_token.attrib["id"])

            if cur_frame_name not in frames.keys():
                frames[cur_frame_name] = list()

            cur_frame_array = frames[cur_frame_name]

            if cur_frame_iob == "B-":
                new_frame = {"frame": cur_frame_name, "tokens": [cur_token_id], "lexical_unit": [], "frame_elements": []}
                cur_frame_array.append(new_frame)
            elif cur_frame_iob == "I-":
                frame = cur_frame_array[-1]
                frame["tokens"].append(cur_token_id)

        hrc2["frame_semantics"].extend([f for frame in frames.values() for f in frame])
        xml_frame_element_tokens = root.findall("./semantics/frame/frameElement/token")

        def find_frame_from_id(id, frames):
            for frame in frames:
                if id in frame["tokens"]:
                    return frame

        for xml_token in xml_frame_element_tokens:
            cur_token_id = int(xml_token.attrib["id"])
            cur_frame_element = xml_token.attrib["value"]
            cur_frame_element_type = cur_frame_element[2:]
            cur_frame_element_iob = cur_frame_element[:2]
            frame = find_frame_from_id(cur_token_id, hrc2["frame_semantics"])

            def get_last_fe(fe_name):
                for fe in reversed(frame["frame_elements"]):
                    if fe["frame_element"] == fe_name:
                        return fe
                return None

            if cur_frame_element_type == "Lexical_unit":
                frame["lexical_unit"].append(cur_token_id)
            else:
                if cur_frame_element_iob == "B-":
                    frame_element = {"frame_element": cur_frame_element_type, "tokens": [cur_token_id]}
                    frame["frame_elements"].append(frame_element)
                elif cur_frame_element_iob == "I-":
                    if frame_element["frame_element"] == cur_frame_element_type:
                        frame_element["tokens"].append(cur_token_id)
                    else:
                        frame_element = get_last_fe(cur_frame_element_type)
                        frame_element["tokens"].append(cur_token_id)

        return hrc2

    @staticmethod
    def load_hrc3(hrc3_file):
        """
        loads all the frame semantics information
        :param hrc3_file: the complete path to the hrc3 file
        :return:
        """
        hrc3 = defaultdict(lambda: 0)
        #hrc3["name"] = ntpath.basename(hrc3_file)
        hrc3["name"] = hrc3_file  # saving the whole path for easy loading
        xml = Et.parse(hrc3_file)
        root = xml.getroot()
        example_id = root.attrib["id"]
        hrc3["id"] = example_id

        xml_sentences = root.findall("./sentence")
        for xml_sentence in xml_sentences:
            sentence = xml_sentence.text
            hrc3["sentence"] = sentence

        tokens = list()
        xml_tokens = root.findall("./tokens/token")
        for xml_token in xml_tokens:
            token = defaultdict()
            token["id"] = int(xml_token.attrib["id"])
            token["lemma"] = xml_token.attrib["lemma"]
            token["surface"] = xml_token.attrib["surface"]
            token["pos"] = xml_token.attrib["pos"]
            tokens.append(token)
        hrc3["tokens"] = tokens

        # named entities
        hrc3["ner"] = list()
        ne_tokens = root.findall("./semantics/ner/")
        nes = defaultdict(lambda: 0)
        # probably for named entities this is even too paranoid
        for ne_token in ne_tokens:
            cur_ne = ne_token.attrib["value"]
            cur_ne_name = cur_ne[2:]
            cur_ne_iob = cur_ne[:2]
            cur_token_id = int(ne_token.attrib["id"])

            if cur_ne_name not in nes.keys():
                nes[cur_ne_name] = list()

            cur_ne_array = nes[cur_ne_name]

            if cur_ne_iob == "B-":
                new_ne = {"ne": cur_ne_name, "tokens": [cur_token_id]}
                cur_ne_array.append(new_ne)
            elif cur_ne_iob == "I-":
                ne = cur_ne_array[-1]
                ne["tokens"].append(cur_token_id)
        hrc3["ner"].extend([n for ne in nes.values() for n in ne])

        # dialogues acts
        hrc3["dialogue_acts"] = list()
        da_tokens = root.findall("./semantics/dialogueAct/")
        das = defaultdict(lambda: 0)
        # probably for named entities this is even too paranoid
        for da_token in da_tokens:
            cur_da = da_token.attrib["value"]
            cur_da_name = cur_da[2:]
            cur_da_iob = cur_da[:2]
            cur_token_id = int(da_token.attrib["id"])

            if cur_da_name not in das.keys():
                das[cur_da_name] = list()

            cur_da_array = das[cur_da_name]

            if cur_da_iob == "B-":
                new_da = {"dialogue_act": cur_da_name, "tokens": [cur_token_id]}
                cur_da_array.append(new_da)
            elif cur_da_iob == "I-":
                da = cur_da_array[-1]
                da["tokens"].append(cur_token_id)
        hrc3["dialogue_acts"].extend([d for da in das.values() for d in da])

        # frame semantics
        hrc3["frame_semantics"] = list()
        xml_frames = root.findall("./semantics/frames/")

        for xml_frame in xml_frames:
            frame = {"frame": "", "tokens": [], "lexical_unit":[ ], "frame_elements": []}
            # todo to ripristinate this, I need to correct the export from the DAP to add frame type
            #frame_name = xml_frame.attrib["type"]
            #frame["frame"] = frame_name
            xml_frame_tokens = xml_frame.findall("./token")

            for xml_frame_token in xml_frame_tokens:
                if xml_frame_token.attrib["value"].startswith("B-"):
                    frame["frame"] = xml_frame_token.attrib["value"][2:]
                cur_token_id = int(xml_frame_token.attrib["id"])
                frame["tokens"].append(cur_token_id)

            def get_last_fe(fe_name):
                for fe in reversed(frame["frame_elements"]):
                    if fe["frame_element"] == fe_name:
                        return fe
                return None

            xml_fe_tokens = xml_frame.findall("./frameElement/token")
            for xml_fe_token in xml_fe_tokens:
                cur_token_id = int(xml_fe_token.attrib["id"])
                cur_fe_name = xml_fe_token.attrib["value"][2:]
                cur_fe_iob = xml_fe_token.attrib["value"][:2]

                if cur_fe_name.lower() == "lexical_unit":
                    frame["lexical_unit"].append(cur_token_id)
                else:
                    if cur_fe_iob == "B-":
                        frame_element = {"frame_element": cur_fe_name, "tokens": [cur_token_id]}
                        frame["frame_elements"].append(frame_element)
                    elif cur_fe_iob == "I-":
                        if frame_element["frame_element"] == cur_fe_name:
                            frame_element["tokens"].append(cur_token_id)
                        else:
                            frame_element = get_last_fe(cur_fe_name)
                            frame_element["tokens"].append(cur_token_id)

            hrc3["frame_semantics"].append(frame)

        return hrc3

    def load_diafram_section(self, section_dir, ann_type="hrc3", extends=False):
        section_name = ntpath.basename(section_dir)
        files = [f for f in os.listdir(section_dir) if os.path.isfile(os.path.join(section_dir, f))
                 and not f.startswith(".")]
        bar = Bar('Loading from {}'.format(section_name), max=len(files))
        section = list()

        for f in files:
            file_path = os.path.join(section_dir, f)
            if os.path.isfile(file_path):
                if ann_type == "hrc3" and f.endswith(".hrc3"):
                    hrc = self.load_hrc3(file_path)
                    section.append(hrc)
                    bar.next()
                elif ann_type == "hrc2" and f.endswith(".hrc2"):
                    hrc = self.load_hrc2(file_path)
                    section.append(hrc)
                    bar.next()
        bar.finish()

        if extends:
            self.dataset.extend(section)
        else:
            self.dataset = section

        return section

    def load_diafram(self, dataset_folder, ann_type="hrc3", extends=False):
        if self.is_section(dataset_folder):
            return self.load_diafram_section(dataset_folder, ann_type, extends)
        else:
            diafram = list()
            for f in os.listdir(dataset_folder):
                subdir_path = os.path.join(dataset_folder, f)
                if os.path.isdir(subdir_path):
                    section = self.load_diafram_section(subdir_path, ann_type=ann_type, extends=extends)
                    diafram.extend(section)

            if not extends:
                self.dataset = diafram

            return diafram

    def is_section(self, dataset_folder):
        if dataset_folder.endswith("/"):
            dataset_folder = dataset_folder[:-1]
        if ntpath.basename(dataset_folder) in SECTIONS:
            return True
        else:
            return False


if __name__ == "__main__":
    """filepath = "/Users/eb70/Projects/robots/hwu/nlu/datasets/diafram_corpus/sections/alexa/1541761236865.hrc2"
    #dirpath = "/Users/eb70/Projects/robots/hwu/nlu/datasets/diafram_corpus/sections"
    dirpath = "/Users/eb70/Projects/robots/hwu/nlu/raw_sentences/hrc3"
    d = DiaframFrameSpam(dirpath, padding=True)

    for e in d.raw_dataset:
        if len(e["frame_semantics"]) > 2:
            print(json.dumps(dict(e)))"""

    #train, val, test = d.get_train_val_test()
    #load_from_fulltext("/Users/eb70/Work/HWU/Resources/Corpora/framenet/fndata-1.7/fulltext")
    #exs = load_from_fulltext_nltk()
    #exs = load_from_lu_nltk()
    #print(exs[40])
    #for e in exs[10:30]:
    #    print(e)
    #print(len(exs))

    #path = "../resources/conll05/test-wsj"
    #path = "/Users/eb70/Work/HWU/Resources/Corpora/ontonotes/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/train/data/english/annotations/bc/p2.5_c2e/00/p2.5_c2e_0001.gold_conll"
    #path = "/Users/eb70/Work/HWU/Resources/Corpora/ontonotes/conll-formatted-ontonotes-5.0-12/conll-formatted-ontonotes-5.0/data/development/"
    #load_conll12(path)

    diafram_hrc3 = "/Users/eb70/Projects/robots/hwu/nlu/raw_sentences/hrc3"
    diafram_hrc2 = "/Users/eb70/Projects/robots/hwu/nlu/datasets/diafram_corpus/sections"
    loader = SSPLoader()
    loader.load_diafram(diafram_hrc3, 'hrc3')
    print(json.dumps(loader.dataset[670]))
    #loader.load_conll12("/Users/eb70/Work/HWU/Resources/Corpora/ontonotes/conll-formatted-ontonotes-5.0-12/"
    #                    "conll-formatted-ontonotes-5.0/data/development/")
    #loader.load_conll05("../resources/conll05/train-set")
    #loader.print_advanced_stats()
    #loader.load_from_fulltext_nltk()
    #loader.load_diafram(diafram_hrc2, ann_type="hrc2")
    #loader.print_stats()
    #loader.load_diafram_section(diafram_hrc3+"/mummer_hrc3", ann_type="hrc3")
    #loader.print_advanced_stats()
    #loader.load_diafram(diafram_hrc3, ann_type="hrc3")
    #loader.print_stats()
