
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import train_step, val_step
from dataset import EurDataset, collate_data
from transceiver import DeepJSOC
from torch.utils.data import DataLoader
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='data/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints', type=str)
parser.add_argument('--MAX-LENGTH', default=32, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=8, type=int)
parser.add_argument('--epochs', default=150, type=int)
parser.add_argument('--warm-start',default=-1,type=int)
parser.add_argument('--vq-dim', default=6, type=int)
parser.add_argument('--channel-in-len', default=36, type=int)
parser.add_argument('--marker-enc-size', default=44, type=int)
parser.add_argument('--safety-len', default=59, type=int)
parser.add_argument('--estimator-file',default='36bit_59bit_estimator.pth')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def validate(epoch, args, net):
    test_eur = EurDataset('val')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            ce_loss,comm_loss,gru_loss = val_step(net,epoch,args.warm_start, sents, sents, pad_idx,
                             criterion)

            total += ce_loss
            pbar.set_description(
                'Epoch: {}; Type: VAL; CELoss: {:.5f} comm_loss={}, gru_loss ={}'.format(
                    epoch + 1, ce_loss,comm_loss,gru_loss
                )
            )

    return total/len(test_iterator)


def train(epoch, args, net, mi_net=None):
    train_eur= EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)

    for sents in pbar:
        sents = sents.to(device)

        ce_loss,comm_loss,gru_loss = train_step(net,epoch,args.warm_start, sents, sents, pad_idx,
                          optimizer, criterion)
        pbar.set_description(
            'Epoch: {};  Type: Train; CELoss: {:.5f}, comm_loss={}, gru_loss ={}'.format(
                epoch + 1, ce_loss,comm_loss,gru_loss
            )
        )


if __name__ == '__main__':
    # setup_seed(10)
    args = parser.parse_args()
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]


    """ define optimizer and loss function """
    deepjsoc = DeepJSOC(args.num_layers, num_vocab, num_vocab,
                        num_vocab, num_vocab, args.d_model, args.num_heads,
                        args.dff,args.vq_dim,args.channel_in_len,args.marker_enc_size,
                        args.safety_len,args.estimator_file, 0.1).to(device)
    #checkpoint = torch.load('checkpoints/deepsc-Rayleigh/checkpoint_00.pth')
    #deepjsoc.load_state_dict(checkpoint)
    criterion = nn.CrossEntropyLoss(reduction = 'none')
    #optimizer = torch.optim.Adam(deepsc.parameters(),
    #                             lr=5e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay = 5e-4)

    optimizer = torch.optim.Adam(deepjsoc.parameters(),
                                 lr=args.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay = 5e-4)
    #initNetParams(deepsc)
    #for name, param in deepsc.named_parameters():
    #    print(f"Parameter name: {name}")
    #    print(param.shape)
    #    print(param.requires_grad)
    for epoch in range(args.epochs):
        start = time.time()
        record_acc = 10
        train(epoch, args, deepjsoc)
        avg_acc = validate(epoch, args, deepjsoc)

        if avg_acc < record_acc:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            with open(args.checkpoint_path + '/checkpoint_{}.pth'.format(str(epoch + 1).zfill(2)), 'wb') as f:
                torch.save(deepjsoc.state_dict(), f)
            record_acc = avg_acc
    record_loss = []

    

        


