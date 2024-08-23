import logging
import numpy as np
import torch
from utils.tool import *
from models.CITOD_bert import MPFToD_Bert, Trainer
from utils.configue import Configure
import wandb

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def set_seed(args):
    np.random.seed(args.train.seed)
    random.seed(args.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train.seed)
        torch.cuda.manual_seed(args.train.seed)

    torch.manual_seed(args.train.seed)
    torch.random.manual_seed(args.train.seed)


def get_args_dict(args):
    ans = {}
    for x, y in args:
        if isinstance(y, (int, float, str)):
            ans[x] = y
        else:
            ans[x] = get_args_dict(y)
    return ans


def start():
    logging.basicConfig(level=logging.INFO)
    args = Configure.Get(pretrain=True)
    set_seed(args)
    x = get_args_dict(args)
    wandb.init(
        project="mpftod_bert",
        notes="mpftod",
        tags=["mpftod", "bert"],
        config=x,
        name=args.train.log_name
    )
    wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release
    model = MPFToD_Bert(args)
    if args.train.gpu:
        model.cuda()
    trainer = Trainer(args, model)
    trainer.start()


if __name__ == "__main__":
    start()
