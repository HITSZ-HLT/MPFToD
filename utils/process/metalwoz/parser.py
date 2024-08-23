import json
import os

from utils.tool import str_filter


def load_file(file):
    dialogs = []
    with open(file) as f:
        dials = f.readlines()
        for dial in dials:
            dial_dict = json.loads(dial)
            # Reading data
            dialog = dial_dict["turns"]
            for idx, turn in enumerate(dialog):
                dialog[idx] = turn.lower().strip()
            if max([len(sentence) for sentence in dialog]) > 200:
                continue
            while len(dialog) > 0 and str_filter(dialog[-1]):
                dialog = dialog[:-2]
            if 1 < len(dialog) <= 26:
                dialogs.append(dialog[1:])

    return dialogs


class Parser(object):
    @staticmethod
    def load():
        onlyfiles = ['data/metalwoz/dialogues/{}'.format(f) for f in os.listdir('data/metalwoz/dialogues') if
                     ".txt" in f]
        dialogs = []
        for file in onlyfiles:
            train_data = load_file(file)
            dialogs.extend(train_data)

        return dialogs
