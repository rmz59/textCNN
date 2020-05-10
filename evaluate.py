import dill
import torch
import torch.nn as nn
import torchtext

from pathlib import Path
from torchtext.datasets import IMDB
from model import TextCNN


def evaluate(model, val_iter, loss_func):
    was_training = model.training
    model.eval()
    val_loss = 0
    correct_case = 0
    example_cnt = 0
    with torch.no_grad():   
        step = 0
        for batch in iter(val_iter):
            feature, target = batch.text.T, batch.label.squeeze(0)
            step += 1
            res = model(feature) 
            loss = loss_func(res, target)
            correct_case += (res.argmax(axis=1)==target).float().sum()
            example_cnt += feature.shape[0]
            val_loss += loss
    val_loss = val_loss / step
    accuracy = correct_case / example_cnt
    if was_training:
        model.train()
    return val_loss, accuracy




def test(config):
    device = 'cuda' if config['cuda'] else 'cpu'
    model = TextCNN.load(config['model_path']).to(device)
    with open(f"{config['text_vocab']}", "rb") as f: 
        TEXT = dill.load(f)
    with open(f"{config['label_vocab']}", "rb") as f: 
        LABEL = dill.load(f)
    _, test_data = IMDB.splits(TEXT, LABEL, root=config['data_path'])
    test_iter = torchtext.data.Iterator(test_data, batch_size=config['batch_size'], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(config['class_weight'], device=device))
    val_loss, accuracy = evaluate(model, test_iter, loss_fn)
    print(f"val_loss:{val_loss} - accuracy:{accuracy}")

if __name__ == "__main__":
    config = {
        "data_path": "data",
        "pretrained_model_dir": Path.home()/"nlp_data/glove.6B/",
        "pretrained_model_file": "glove.6B.300d.txt",
        "model_path": "output/model_best",
        "text_vocab": "output/TEXT_vocab.bin",
        "label_vocab": "output/LABEL_vocab.bin",

        "batch_size": 128,
        "max_sent_length": 100,
        "class_weight": [0,0,1.0,1.0],

        "cuda": True,
    }
    test(config)



