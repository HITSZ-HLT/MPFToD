from utils.tool import str_filter


def load_file(file):
    dialogs = []
    with open(file) as f:
        dials = f.readlines()
        for dial in dials:
            seqs = dial.lower().strip().split('__eou__')
            seqs = seqs[:-1]
            for idx, turn in enumerate(seqs):
                seqs[idx] = turn.lower().strip()
            if max([len(seq) for seq in seqs]) > 200:
                continue
            while len(seqs) > 0 and str_filter(seqs[-1]):
                seqs = seqs[:-2]
            if 0 < len(seqs) <= 25:
                if len(seqs) % 2 == 1:
                    dialogs.append(seqs[:-1])
                else:
                    dialogs.append(seqs)

    return dialogs


class Parser(object):
    @staticmethod
    def load():
        path = 'data/dailydialog/dialogues_text.txt'
        print(("Reading from {}for dialogs".format(path)))
        dialogs = []
        data = load_file(path)
        dialogs.extend(data)

        return dialogs
