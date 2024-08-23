import json

from utils.tool import str_filter


def load_file(file):
    dialogs = []
    with open(file) as f:
        dials = json.load(f)
        for dial_dict in dials:
            dialog = []
            for ti, turn in enumerate(dial_dict["turns"]):
                if turn["author"] == "user":
                    turn_usr = turn["text"].lower().strip()
                    if len(turn_usr) > 200:
                        break
                    dialog.append(turn_usr)
                elif turn["author"] == "wizard":
                    turn_sys = turn["text"].lower().strip()
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
        train_path = 'data/frames.json'
        dialogs = []
        train_data = load_file(train_path)
        dialogs.extend(train_data)
        return dialogs
