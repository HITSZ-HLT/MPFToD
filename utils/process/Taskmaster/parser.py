import json
import os

from utils.tool import str_filter


def load_data(dials):
    dialogs = []
    for dial in dials:
        dialog = []
        for ti, turn in enumerate(dial["utterances"]):
            if turn["speaker"] == "USER":
                turn_usr = turn["text"].lower().strip()
                if len(turn_usr) > 200:
                    break
                dialog.append(turn_usr)
            elif turn["speaker"] == "ASSISTANT":
                turn_sys = turn["text"].lower().strip()
                if len(turn_sys) > 200:
                    break
                dialog.append(turn_sys)
        while len(dialog) > 0 and str_filter(dialog[-1]):
            dialog = dialog[:-2]
        if 1 < len(dialog) <= 26:
            if len(dialog) %2 == 0:
                dialogs.append(dialog[1:-1])
            else:
                dialogs.append(dialog[1:])

    return dialogs


class Parser(object):
    @staticmethod
    def load():
        fr_data_woz = open('data/Taskmaster/TM-1-2019/woz-dialogs.json', 'r')
        fr_data_self = open('data/Taskmaster/TM-1-2019/self-dialogs.json', 'r')
        dials_all = json.load(fr_data_woz) + json.load(fr_data_self)
        dialogs = load_data(dials_all)
        return dialogs
