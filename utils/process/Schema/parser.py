import json
import os

from utils.tool import str_filter


def load_file(file):
    dialogs = []
    with open(file) as f:
        dials = json.load(f)
        for dial_dict in dials:
            dialog = []
            for ti, turn in enumerate(dial_dict["turns"]):
                if turn["speaker"] == "USER":
                    turn_usr = turn["utterance"].lower().strip()
                    if len(turn_usr) > 200:
                        break
                    dialog.append(turn_usr)
                elif turn["speaker"] == "SYSTEM":
                    turn_sys = turn["utterance"].lower().strip()
                    if len(turn_sys) > 200:
                        break
                    dialog.append(turn_sys)
            while len(dialog) > 0 and str_filter(dialog[-1]):
                dialog = dialog[:-2]
            if 0 < len(dialog) <= 25:
                if len(dialog) % 2 == 1:
                    dialogs.append(dialog[:-1])
                else:
                    dialogs.append(dialog)

    return dialogs


class Parser(object):
    @staticmethod
    def load():
        onlyfiles_trn = ['data/Schema/train/{}'.format(f) for f in os.listdir('data/Schema/train') if "dialogues" in f]
        onlyfiles_dev = ['data/Schema/dev/{}'.format(f) for f in os.listdir('data/Schema/dev') if "dialogues" in f]
        onlyfiles_test = ['data/Schema/test/{}'.format(f) for f in os.listdir('data/Schema/test') if "dialogues" in f]
        dialogs = []
        for file in onlyfiles_trn:
            train_data = load_file(file)
            dialogs.extend(train_data)

        for file in onlyfiles_dev:
            dev_data = load_file(file)
            dialogs.extend(dev_data)

        for file in onlyfiles_test:
            test_data = load_file(file)
            dialogs.extend(test_data)

        return dialogs
