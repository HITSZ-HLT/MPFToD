import json
import os

from utils.tool import str_filter


def load_file(file):
    dialogs = []
    with open(file) as f:
        dials = json.load(f)
        for dial_list in dials:
            dialog = []
            for ti, turn in enumerate(dial_list):
                text = turn["raw_text"].lower().strip()
                if len(text) > 200:
                    break
                dialog.append(text)
            while len(dialog) > 0 and str_filter(dialog[-1]):
                dialog = dialog[:-2]
            if 1 < len(dialog) <= 26:
                if len(dialog) % 2 == 1:
                    dialogs.append(dialog[:-1])
                else:
                    dialogs.append(dialog)

    return dialogs


class Parser(object):
    @staticmethod
    def load():
        train_path = 'data/universal_dialog_act/sim_joint/train.json'
        dev_path = 'data/universal_dialog_act/sim_joint/valid.json'
        test_path = 'data/universal_dialog_act/sim_joint/test.json'
        print(("Reading from {},{},{} for dialogs".format(train_path, dev_path, test_path)))
        dialogs = []
        train_data = load_file(train_path)
        dialogs.extend(train_data)
        dev_data = load_file(dev_path)
        dialogs.extend(dev_data)
        test_data = load_file(test_path)
        dialogs.extend(test_data)

        return dialogs
