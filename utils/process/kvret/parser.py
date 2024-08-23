import json

from utils.tool import str_filter


def load_file(file):
    dialogs = []
    with open(file) as f:
        dials = json.load(f)
        for dial_dict in dials:
            # Reading data
            dialog = []
            for ti, turn in enumerate(dial_dict["dialogue"]):
                if turn["turn"] == "driver":
                    turn_usr = turn["data"]["utterance"].lower().strip()
                    if len(turn_usr) > 200:
                        break
                    dialog.append(turn_usr)
                elif turn["turn"] == "assistant":
                    turn_sys = turn["data"]["utterance"].lower().strip()
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
        train_path = 'data/kvret/kvret_train_public.json'
        dev_path = 'data/kvret/kvret_dev_public.json'
        test_path = 'data/kvret/kvret_test_public.json'
        print(("Reading from {},{},{} for dialogs".format(train_path, dev_path, test_path)))
        dialogs = []
        train_data = load_file(train_path)
        dialogs.extend(train_data)
        dev_data = load_file(dev_path)
        dialogs.extend(dev_data)
        test_data = load_file(test_path)
        dialogs.extend(test_data)

        return dialogs
