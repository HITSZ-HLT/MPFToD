import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, AdamW, BertConfig, get_linear_schedule_with_warmup, BertForMaskedLM
import utils.tool
from tqdm import tqdm, trange
import wandb


class BERTTool(object):
    def init(args):
        BERTTool.config = BertConfig.from_pretrained(args.bert.location)
        BERTTool.bert = BertForMaskedLM.from_pretrained(args.bert.location, config=BERTTool.config)
        BERTTool.tokenizer = BertTokenizer.from_pretrained(args.bert.location)
        BERTTool.pad = BERTTool.tokenizer.pad_token
        BERTTool.sep = BERTTool.tokenizer.sep_token
        BERTTool.cls = BERTTool.tokenizer.cls_token
        BERTTool.pad_id = BERTTool.tokenizer.pad_token_id
        BERTTool.sep_id = BERTTool.tokenizer.sep_token_id
        BERTTool.cls_id = BERTTool.tokenizer.cls_token_id
        BERTTool.special_tokens = ["<table>", "</table>", "<row>", "</row>", "<header>", "</header>", "|", "[SOT]", "[USR]", "[SYS]"]


class MPFToD_Bert(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        np.random.seed(args.train.seed)
        torch.manual_seed(args.train.seed)
        random.seed(args.train.seed)
        self.args = args
        BERTTool.init(self.args)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bert = BERTTool.bert
        self.config = BERTTool.config
        self.tokenizer = BERTTool.tokenizer
        special_tokens_dict = {'additional_special_tokens': BERTTool.special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.bert.resize_token_embeddings(len(self.tokenizer))

        self.hi_linear = nn.Linear(BERTTool.config.hidden_size, 2)
        self.qi_linear = nn.Linear(BERTTool.config.hidden_size, 2)
        self.hi_criterion = nn.CrossEntropyLoss()
        self.qi_criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        task_id, input, labels = batch
        if task_id == 0:
            outputs_gen = self.bert(input_ids=input['input_ids'],
                                    token_type_ids=input['token_type_ids'],
                                    attention_mask=input['attention_mask'],
                                    output_hidden_states=True,
                                    return_dict=True
                                    )
            last_hiddens = outputs_gen.hidden_states[-1]
            utt = self.bert.bert.pooler(last_hiddens)
            logits = self.qi_linear(utt).squeeze(1)
            loss = self.qi_criterion(logits.view(-1, 2), labels.view(-1))
            if self.args.train.gradient_accumulation_steps > 1:
                loss = loss / self.args.train.gradient_accumulation_steps
            return task_id, loss

        elif task_id == 1:
            outputs_gen = self.bert(input_ids=input['input_ids'],
                                    token_type_ids=input['token_type_ids'],
                                    attention_mask=input['attention_mask'],
                                    output_hidden_states=True,
                                    return_dict=True
                                    )
            last_hiddens = outputs_gen.hidden_states[-1]
            utt = self.bert.bert.pooler(last_hiddens)
            logits = self.hi_linear(utt).squeeze(1)
            loss = self.hi_criterion(logits.view(-1, 2), labels.view(-1))
            if self.args.train.gradient_accumulation_steps > 1:
                loss = loss / self.args.train.gradient_accumulation_steps
            return task_id, loss

        elif task_id == 2:
            outputs_gen = self.bert(input_ids=input['input_ids'],
                                    token_type_ids=input['token_type_ids'],
                                    attention_mask=input['attention_mask'],
                                    labels=labels
                                    )
            loss = outputs_gen[0]
            if self.args.train.gradient_accumulation_steps > 1:
                loss = loss / self.args.train.gradient_accumulation_steps
            return task_id, loss


class Trainer(torch.nn.Module):
    def __init__(self, args, model):
        super().__init__()
        np.random.seed(args.train.seed)
        torch.manual_seed(args.train.seed)
        random.seed(args.train.seed)
        self.args = args
        self.model = model
        self.data_loader = utils.tool.get_multi_task_loader(args, self.model.tokenizer)
        self.set_optimizer()

    def set_optimizer(self):
        if self.args.train.max_steps > 0:
            t_total = self.args.train.max_steps
            self.args.train.epoch = self.args.train.max_steps // (
                    len(self.data_loader) // self.args.train.gradient_accumulation_steps) + 1
        else:
            t_total = len(self.data_loader) // self.args.train.gradient_accumulation_steps * self.args.train.epoch
        all_params = set(self.model.parameters())
        params = [{"params": list(all_params), "lr": self.args.lr.bert}]
        warmup_steps = int(self.args.train.warmup_rate * t_total)
        self.optimizer = AdamW(params, weight_decay=self.args.train.weight_decay)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=warmup_steps,
                                                         num_training_steps=t_total)

    def run_train(self):
        global_step = 0
        self.model.zero_grad()
        train_iterator = trange(0, int(self.args.train.epoch), desc="Epoch")

        # wandb.watch() automatically fetches all layer dimensions, gradients, model parameters
        # and logs them automatically to your dashboard.
        # using log="all" log histograms of parameter values in addition to gradients
        wandb.watch(self.model, log="all")

        for epoch in train_iterator:
            self.model.train()
            loss_arr = [[], [], []]
            logging.info("Starting training epoch {}".format(epoch))
            epoch_iterator = tqdm(self.data_loader)
            for step, batch in enumerate(epoch_iterator):
                task_id, loss = self.model(batch)
                loss_arr[task_id].append(loss.item())
                loss.backward()
                if task_id == 0:
                    wandb.log({"qi_loss": loss.item()})
                elif task_id == 1:
                    wandb.log({"hi_loss": loss.item()})
                elif task_id == 2:
                    wandb.log({"kbi_loss": loss.item()})

                #  Print loss
                epoch_iterator.set_description(
                    "QI_Loss:{:.4f}  HI_Loss:{:.4f}  KBI_Loss:{:.4f} ".format(np.mean(loss_arr[0]),
                                                                              np.mean(loss_arr[1]),
                                                                              np.mean(loss_arr[2])))

                if (step + 1) % self.args.train.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.train.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                if self.args.train.save_steps > 0 and global_step % self.args.train.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(self.args.train.output_dir,
                                              "{}-{}".format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = (self.model.bert.module if hasattr(self.model.bert,
                                                                       "module") else self.model.bert)
                    model_to_save.save_pretrained(output_dir)
                    self.model.tokenizer.save_pretrained(output_dir)
                    self.model.tokenizer.save_vocabulary(output_dir)

                    torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
                    logging.info("Saving model checkpoint to %s", output_dir)

                if 0 < self.args.train.max_steps < global_step:
                    epoch_iterator.close()
                    break

            logging.info(
                "At epoch: {}, step: {} ,QI_Loss:{:.4f} HI_Loss:{:.4f} KBI_Loss:{:.4f}".format(epoch, global_step,
                                                                                               np.mean(loss_arr[0]),
                                                                                               np.mean(loss_arr[1]),
                                                                                               np.mean(loss_arr[2])))
            if 0 < self.args.train.max_steps < global_step:
                train_iterator.close()
                print('up to max steps, jump out!')
                break

    def start(self):
        self.run_train()
