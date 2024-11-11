
import os
import subprocess
import json
import torch
import argparse
import numpy as np
import sys
from dataset import EurDataset, collate_data
from transceiver import DeepJSOC
from torch.utils.data import DataLoader
from utils import BleuScore, greedy_decode, SeqtoText
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# see the details of model with print
model  = SentenceTransformer("bert-base-uncased")
from w3lib.html import remove_tags

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='data/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints', type=str)
parser.add_argument('--channel', default='AWGN', type=str)
parser.add_argument('--MAX-LENGTH', default=32, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type = int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('--epochs', default=2, type = int)
parser.add_argument('--vq-dim', default=6, type=int)
parser.add_argument('--channel-in-len', default=36, type=int)
parser.add_argument('--marker-enc-size', default=44, type=int)
parser.add_argument('--safety-len', default=59, type=int)
parser.add_argument('--estimator-file',default='36bit_59bit_estimator.pth')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def performance(args, Pd,ps, net):
    bleu_score_1gram = BleuScore(1, 0, 0, 0)

    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)

    StoT = SeqtoText(token_to_idx, end_idx)
    score = []
    score2 = []
    net.eval()
    with torch.no_grad():
        sim_score = []
        for epoch in range(args.epochs):
            Tx_word = []
            Rx_word = []
            #break
            test_p = -1
            for pd in tqdm(Pd):
                test_p +=1
                #break
                word = []
                target_word = []

                for sents in test_iterator:

                    sents = sents.to(device)
                    # src = batch.src.transpose(0, 1)[:1]
                    target = sents

                    out = greedy_decode(net, sents, pd,ps, args.MAX_LENGTH, pad_idx,
                                        start_idx, args.channel)

                    sentences = out.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, sentences))
                    word = word + result_string

                    target_sent = target.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + result_string

                Tx_word.append(word)
                Rx_word.append(target_word)
                embeddings_in = model.encode(Tx_word[test_p])
                embeddings_tx =  model.encode(Rx_word[test_p])
                similarities = cosine_similarity(embeddings_in,embeddings_tx)
                avg_sim = 0
                for i in range(len(similarities)):
                    avg_sim += similarities[i,i]
                avg_sim = avg_sim / (i+1)
                avg_sim = (avg_sim-0.4)/0.6
                if epoch == args.epochs-1:
                  sim_score.append(avg_sim)
            bleu_score = []
            
            for sent1, sent2 in zip(Tx_word, Rx_word):
                # 1-gram
                bleu_score.append(bleu_score_1gram.compute_blue_score(sent1, sent2)) # 7*num_sent
                # sim_score.append(similarity.compute_similarity(sent1, sent2)) # 7*num_sent
            bleu_score = np.array(bleu_score)
            bleu_score = np.mean(bleu_score, axis=1)
            score.append(bleu_score)
           
            # sim_score = np.array(sim_score)
            # sim_score = np.mean(sim_score, axis=1)
            # score2.append(sim_score)
            
    score1 = np.mean(np.array(score), axis=0)
    # score2 = np.mean(np.array(score2), axis=0)

    return score1,sim_score

if __name__ == '__main__':
    args = parser.parse_args()
    #SNR = [0,3,6,9,12,15,18]
    #SNR = [-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]
    #SNR = [0.00]
    #ps = 0.02
    #Pd = [0.01]
    ps = 0.03
    #Pd = [0.00]
    #Pd = [0,0.01,0.02,0.03,0.04,0.05]
    #Ps = [0.03]
    #Ps = [sys.float_info.min,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
    Pd = [sys.float_info.min,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    """ define optimizer and loss function """
    deepjsoc = DeepJSOC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff,args.vq_dim,args.channel_in_len,args.marker_enc_size,
                    args.safety_len,args.estimator_file, 0.1).to(device)


    model_paths = []
    for fn in os.listdir(args.checkpoint_path):
        if not fn.endswith('.pth'): continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])  # read the idx of image
        model_paths.append((os.path.join(args.checkpoint_path, fn), idx))

    model_paths.sort(key=lambda x: x[1])  # sort the image by the idx

    model_path, _ = model_paths[-1]
    print(model_path)
    checkpoint = torch.load(model_path)
    deepjsoc.load_state_dict(checkpoint)
    print('model load!', model_path)

    bleu_score,bert_score = performance(args, Pd,ps, deepjsoc)
    print(bleu_score,bert_score)

