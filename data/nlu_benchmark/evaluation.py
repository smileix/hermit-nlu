import converters
import argparse
from progress.bar import Bar
import os


def evaluate(predictions_file):
    json_prediction = converters.load_json_prediction_file(predictions_file=predictions_file)
    squeezed_predictions = converters.squeeze_prediction_span(json_prediction=json_prediction)

    # Initialising error variables
    intent_tp = 0.
    intent_fp = 0.
    intent_fn = 0.

    dialogue_act_tp = 0.
    dialogue_act_fp = 0.
    dialogue_act_fn = 0.

    frame_tp = 0.
    frame_fp = 0.
    frame_fn = 0.

    entity_tp = 0.
    entity_fp = 0.
    entity_fn = 0.

    combined_tp = 0.
    combined_fp = 0.
    combined_fn = 0.

    bar = Bar('Processing predictions: ', max=len(squeezed_predictions))
    for example in squeezed_predictions:
        bar.next()

        # Intent confusion matrix
        for intent_gold in example['intent_gold']:
            if intent_gold in example['intent_pred']:
                intent_tp += 1
            else:
                intent_fn += 1
        for intent_pred in example['intent_pred']:
            if intent_pred not in example['intent_gold']:
                intent_fp += 1

        # Dialogue act confusion matrix
        for dialogue_act_gold in example['dialogue_act_gold']:
            if dialogue_act_gold in example['dialogue_act_pred']:
                dialogue_act_tp += 1
            else:
                dialogue_act_fn += 1
        for dialogue_act_pred in example['dialogue_act_pred']:
            if dialogue_act_pred not in example['dialogue_act_gold']:
                dialogue_act_fp += 1

        # Frame confusion matrix
        for frame_gold in example['frame_gold']:
            if frame_gold in example['frame_pred']:
                frame_tp += 1
            else:
                frame_fn += 1
        for frame_pred in example['frame_pred']:
            if frame_pred not in example['frame_gold']:
                frame_fp += 1
        # Entity confusion matrix
        for entity_gold_temp in example['entities_gold']:
            found = False
            for entity_gold in entity_gold_temp:
                for entity_pred_temp in example['entities_pred']:
                    if entity_gold in entity_pred_temp:
                        found = not set(entity_gold_temp[entity_gold]).isdisjoint(set(entity_pred_temp[entity_gold]))
                        if found:
                            break
            if found:
                entity_tp += 1
            else:
                entity_fp += 1

        for entity_pred_temp in example['entities_pred']:
            found = False
            for entity_pred in entity_pred_temp:
                for entity_gold_temp in example['entities_gold']:
                    if entity_pred in entity_gold_temp:
                        found = not set(entity_pred_temp[entity_pred]).isdisjoint(set(entity_gold_temp[entity_pred]))
                        if found:
                            break
            if not found:
                entity_fn += 1

        combined_tp = intent_tp + entity_tp
        combined_fn = intent_fn + entity_fn
        combined_fp = intent_fp + entity_fp

    print("")

    intent_precision = intent_tp / (intent_tp + intent_fp)
    intent_recall = intent_tp / (intent_tp + intent_fn)
    intent_f1 = (2 * intent_precision * intent_recall) / (intent_precision + intent_recall)

    dialogue_act_precision = dialogue_act_tp / (dialogue_act_tp + dialogue_act_fp)
    dialogue_act_recall = dialogue_act_tp / (dialogue_act_tp + dialogue_act_fn)
    dialogue_act_f1 = (2 * dialogue_act_precision * dialogue_act_recall) / (dialogue_act_precision + dialogue_act_recall)

    frame_precision = frame_tp / (frame_tp + frame_fp)
    frame_recall = frame_tp / (frame_tp + frame_fn)
    frame_f1 = (2 * frame_precision * frame_recall) / (frame_precision + frame_recall)

    entity_precision = entity_tp / (entity_tp + entity_fp)
    entity_recall = entity_tp / (entity_tp + entity_fn)
    entity_f1 = (2 * entity_precision * entity_recall) / (entity_precision + entity_recall)

    combined_precision = combined_tp / (combined_tp + combined_fp)
    combined_recall = combined_tp / (combined_tp + combined_fn)
    combined_f1 = (2 * combined_precision * combined_recall) / (combined_precision + combined_recall)

    print('Combined scores: P: {}, R: {}, F1: {}'.format(combined_precision,
                                                         combined_recall,
                                                         combined_f1))
    print('Intent scores: P: {}, R: {}, F1: {}'.format(intent_precision,
                                                       intent_recall,
                                                       intent_f1))
    print('Dialogue act scores: P: {}, R: {}, F1: {}'.format(dialogue_act_precision,
                                                             dialogue_act_recall,
                                                             dialogue_act_f1))
    print('Frame scores: P: {}, R: {}, F1: {}'.format(frame_precision,
                                                      frame_recall,
                                                      frame_f1))
    print('Entity scores: P: {}, R: {}, F1: {}'.format(entity_precision,
                                                       entity_recall,
                                                       entity_f1))


def evaluate_folder(folder):
    for root, directories, filenames in os.walk(folder):
        filenames = [fi for fi in filenames if fi.endswith(".json")]
        for filename in filenames:
            evaluate(os.path.join(root, filename))
            print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NLU Benchmark evaluation script')
    parser.add_argument('-i', '--input', type=str, default=None,
                        help='Input file (to evaluate a single file)')
    parser.add_argument('-f', '--folder', type=str, default=None,
                        help='Input folder (to evaluate an entire folder)')
    args = parser.parse_args()
    if args.folder is not None:
        evaluate_folder(args.folder)
    else:
        evaluate(args.input)
