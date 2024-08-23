import json
import copy

from utils.tool import str_filter


def load_file(file_name):
    dialogs = []
    with open(file_name) as f:
        dials = f.readlines()
        dialog = []
        datas = dials[1:]
        for idx, dial in enumerate(datas):
            dial_split = dial.split("\t")
            session_ID, Message_ID, Message_from, Message = dial_split[0], dial_split[1], dial_split[3], dial_split[4]
            if idx >= 1 and datas[idx - 1].split("\t")[0] != session_ID:
                # Join the previous dialog
                while len(dialog) > 0 and str_filter(dialog[-1]):
                    dialog = dialog[:-2]
                if len(dialog) <= 25:
                    if len(dialog) % 2 == 1:
                        dialogs.append(copy.deepcopy(dialog[:-1]))
                    else:
                        dialogs.append(copy.deepcopy(dialog))
                dialog = []

            if Message_from == "user":
                turn_usr = Message.lower().strip()
                if len(turn_usr) > 200:
                    continue
                dialog.append(turn_usr)
            elif Message_from == "agent":
                turn_sys = Message.lower().strip()
                if len(turn_sys) > 200:
                    continue
                dialog.append(turn_sys)
        # add last turn
        while len(dialog) > 0 and str_filter(dialog[-1]):
            dialog = dialog[:-2]
        if 0< len(dialog) <= 25:
            if len(dialog) % 2 == 1:
                dialogs.append(copy.deepcopy(dialog[:-1]))
            else:
                dialogs.append(copy.deepcopy(dialog))

    return dialogs


class Parser(object):
    @staticmethod
    def load():
        file_mov = 'data/e2e_dialog_challenge/data/movie_all.tsv'
        file_rst = 'data/e2e_dialog_challenge/data/restaurant_all.tsv'
        file_tax = 'data/e2e_dialog_challenge/data/taxi_all.tsv'
        print(("Reading from {},{},{} for dialogs".format(file_mov, file_rst, file_tax)))
        dialogs = []
        mov_data = load_file(file_mov)
        dialogs.extend(mov_data)
        rst_data = load_file(file_rst)
        dialogs.extend(rst_data)
        tax_data = load_file(file_tax)
        dialogs.extend(tax_data)

        return dialogs
