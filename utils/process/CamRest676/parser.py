import json
from utils.tool import str_filter
save_response = ['goodbye.','thank you, goodbye.', 'you\'re welcome, goodbye.', 'you\'re welconme','good bye', 'thank you good bye',
                 'have a nice day', 'thank you for using our system. good bye', 'thank you  and good bye', 'enjoy your day!',
                 'goodbye and have a nice day.', 'bye.']

class Parser(object):
    @staticmethod
    def load():
        file_path = 'data/CamRest676/CamRest676.json'
        print(("Reading from {} for dialogs".format(file_path)))
        dialogs = []
        with open(file_path) as f:
            dials = json.load(f)
            for dial_dict in dials:
                dialog = []
                # Reading data
                for ti, turn in enumerate(dial_dict["dial"]):
                    assert ti == turn["turn"]
                    turn_usr = turn["usr"]["transcript"].lower().strip()
                    turn_sys = turn["sys"]["sent"].lower().strip()
                    if len(turn_usr) > 200 or len(turn_sys) > 200:
                        break
                    dialog.append(turn_usr)
                    dialog.append(turn_sys)
                while len(dialog) > 0 and str_filter(dialog[-1]):
                    dialog = dialog[:-2]
                if 0 < len(dialog) <= 25:
                    if len(dialog) % 2 == 1:
                        dialogs.append(dialog[:-1])
                    else:
                        dialogs.append(dialog)
        return dialogs
