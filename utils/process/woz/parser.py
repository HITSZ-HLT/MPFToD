import json
import os

from utils.tool import str_filter


def load_file(file):
    dialogs = []
    with open(file) as f:
        dials = json.load(f)
        for dial_list in dials:
            dialog = []
            for ti, turn in enumerate(dial_list["dialogue"]):
                turn_usr = turn["transcript"].lower().strip()
                turn_sys = turn["system_transcript"].lower().strip()
                if len(turn_usr) > 200 or len(turn_sys) > 200:
                    break
                if turn_sys != '':
                    dialog.append(turn_sys)
                dialog.append(turn_usr)
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
        train_path = 'data/woz/woz_train_en.json'
        dev_path = 'data/woz/woz_validate_en.json'
        test_path = 'data/woz/woz_test_en.json'
        print(("Reading from {},{},{} for dialogs".format(train_path, dev_path, test_path)))
        dialogs = []
        train_data = load_file(train_path)
        dialogs.extend(train_data)
        dev_data = load_file(dev_path)
        dialogs.extend(dev_data)
        test_data = load_file(test_path)
        dialogs.extend(test_data)

        return dialogs
