import pickle
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import click

from pathlib import Path
from tqdm import tqdm
import torchtext
from torchtext.vocab import Vectors
from torchtext.datasets import SST
from model import TextCNN
logging.basicConfig(
    format='%(levelname)s: %(asctime)s %(message)s', level=logging.DEBUG)


# @click.option('--split', default='train', required=True)
# @click.option('--data_path', default="data/sst", help="path of the training data", required=True)
# @click.option('--save_to', default="outputs", required=True)
# @click.option('--batch_size', default=2048, type=int)
# @click.option('--max_sent_length', default=200, type=int)
# @click.option('--char_embed_size', default=200, type=int)
# @click.option('--num_hidden_layer', default=5, type=int)
# @click.option('--channel_size', default=200, type=int)
# @click.option('--kernel_size', default=3, type=int)
# @click.option('--learning_rate', default=2e-3, type=float)
# @click.option('--max_epoch', default=1000, type=int)
# @click.option('--log_every', default=100, type=int)
# @click.option('--patience', default=20, type=int)
# @click.option('--max_num_trial', default=20, type=int)
# @click.option('--lr_decay', default=0.5, type=float)
# @click.option('--last_model_path', default=None, type=str)
# @click.option('--cuda', default=True, type=bool)
config = {
    "split": "train",
    "data_path": "data",
    "pretrained_model_dir": "/Users/runmin/nlp_data/glove.6B/",
    "pretrained_model_file": "glove.6B.300d.txt",
    "last_model_path": None,
    "save_to": "output",

    "min_freq": 5,

    "batch_size": 128,
    "max_sent_length": 100,
    "embed_dim": 300,
    "filter_num": 100,
    "filter_widths": [3, 4, 5],

    "learning_rate": 1e-3,
    "patience": 20,
    "lr_decay": 0.5,
    "max_num_trial": 20,
    "max_epoch": 200,

    "cuda": True,
    "debug": True
}


def train(config):
    try:
        split = config["split"]
        data_path = config["data_path"]
        pretrained_model_dir = config["pretrained_model_dir"]
        pretrained_model_file = config["pretrained_model_file"]
        last_model_path = config["last_model_path"]
        save_to = config["save_to"]
        min_freq = config["min_freq"]
        batch_size = config["batch_size"]
        max_sent_length = config["max_sent_length"]
        embed_dim = config["embed_dim"]
        filter_num = config["filter_num"]
        filter_widths = config["filter_widths"]
        learning_rate = config["learning_rate"]
        patience = config["patience"]
        lr_decay = config["lr_decay"]
        max_num_trial = config["max_num_trial"]
        max_epoch = config["max_epoch"]
        cuda = config["cuda"]
        debug = config["debug"]
    except KeyError:
        print("Input Parameter Error")
        exit(1)

    if not Path(save_to).exists():
        Path(save_to).mkdir()
    device = torch.device("cuda:0" if (
        torch.cuda.is_available() and cuda) else "cpu")

    # build torchtext field
    TEXT = torchtext.data.Field(tokenize='spacy')
    LABEL = torchtext.data.Field(dtype=torch.long)

    train_data, val_data, test_data = SST.splits(TEXT, LABEL, root=data_path)
    if debug is True:
        train_data, _ = train_data.split(split_ratio=0.05)
    train_iter, val_iter = torchtext.data.Iterator.splits((train_data, val_data), batch_size=batch_size, device=device)

    if (pretrained_model_file is not None) and (pretrained_model_dir is not None):
        pretrained_vector = Vectors(name=pretrained_model_file, cache=pretrained_model_dir)
        
    TEXT.build_vocab(train_data, min_freq=min_freq, vectors=pretrained_vector)
    LABEL.build_vocab(train_data)
    
    assert embed_dim == TEXT.vocab.vectors.shape[-1], "incompatiable embeddings"
    embed_num, class_num = len(TEXT.vocab), len(LABEL.vocab)


    model = TextCNN(embed_num, embed_dim, class_num, filter_num, filter_widths, from_pretrained=TEXT.vocab.vectors).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    cross_entropy = nn.CrossEntropyLoss()
    if last_model_path is not None:
        # load model
        logging.info(f'load model from  {last_model_path}')
        params = torch.load(
            last_model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(params['state_dict'])
        logging.info('restore parameters of the optimizers')
        optimizer.load_state_dict(torch.load(last_model_path + '.optim'))

    model.train()

    epoch = 0
    cur_trial = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    logging.info("begin training!")
    while True:
        epoch += 1
        train_loss = 0
        cum_cnt = 0
        step = 0
        for batch in iter(train_iter):
            feature, target = batch.text.T, batch.label.squeeze(0)
            step += 1
            optimizer.zero_grad()
            res = model(feature) 
            loss = cross_entropy(res, target)
            train_loss += loss 
            loss.backward()
            optimizer.step()
        
        val_loss = evaluate(model, val_iter, batch_size)
        print(f"epoch:{epoch} - train_loss:{train_loss}")

        # logging.info(
        #     f'epoch {epoch}\t train_loss: {train_loss}\t val_loss:{val_loss}\t  speed:{time.time()-train_time:.2f}s/epoch\t time elapsed {time.time()-begin_time:.2f}s')
        # train_time = time.time()

        # is_better = len(hist_valid_scores) == 0 or val_loss < min(
        #     hist_valid_scores)
        # hist_valid_scores.append(val_loss)

        # if epoch % log_every == 0:
        #     model.save(f"{save_to}/model_step_{epoch}")
        #     torch.save(optimizer.state_dict(),
        #                f"{save_to}/model_step_{epoch}.optim")
        # if is_better:
        #     cur_patience = 0
        #     model_save_path = f"{save_to}/model_best"
        #     print(f'save currently the best model to [{model_save_path}]')
        #     model.save(model_save_path)
        #     # also save the optimizers' state
        #     torch.save(optimizer.state_dict(), model_save_path + '.optim')
        # elif cur_patience < patience:
        #     cur_patience += 1
        #     print('hit patience %d' % cur_patience)

        #     if cur_patience == patience:
        #         cur_trial += 1
        #         print(f'hit #{cur_trial} trial')
        #         if cur_trial == max_num_trial:
        #             print('early stop!')
        #             exit(0)

        #         # decay lr, and restore from previously best checkpoint
        #         lr = optimizer.param_groups[0]['lr'] * lr_decay
        #         logging.info(
        #             f'load previously best model and decay learning rate to {lr}')

        #         # load model
        #         params = torch.load(
        #             model_save_path, map_location=lambda storage, loc: storage)
        #         model.load_state_dict(params['state_dict'])
        #         model = model.to(device)

        #         logging.info('restore parameters of the optimizers')
        #         optimizer.load_state_dict(
        #             torch.load(model_save_path + '.optim'))

        #         # set new lr
        #         for param_group in optimizer.param_groups:
        #             param_group['lr'] = lr

        #         # reset patience
        #         cur_patience = 0

        if epoch == max_epoch:
            print('reached maximum number of epochs!')
            exit(0)


if __name__ == '__main__':
    train(config)