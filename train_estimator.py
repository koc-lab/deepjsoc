import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os 
import time
import argparse
from training_utils_estimator import create_batch,insert_regular_markers
from models_estimator import BI_Estimator
from scipy.optimize import minimize
parser = argparse.ArgumentParser(description='IDS Channel Training')

parser.add_argument('--path', default='BI_GRU_ESTIMATOR', type=str, help='Saving directory of the trained model.')
parser.add_argument('--training_Pd', default=[0.05], type=list, help='Specify the training deletion probabilities (list)')
parser.add_argument('--training_Ps', default=[0.05], type=list, help='Specify the training substitution probabilities (list)')
parser.add_argument('--training_Pi', default=[0.05], type=list, help='Specify the training deletion probabilities (list)')
parser.add_argument('--marker_sequence', default=np.array([0, 1]).reshape(1, -1), type=np.array, help='Specify the marker sequence.')
parser.add_argument('--loss', choices=['mse', 'bce'], default='bce', type=str, help='Specify the loss function')
parser.add_argument('--epochs', default=1000, type=int, help='Number of total epochs to run')
parser.add_argument('--step', default=100, type=int, help='Number of steps per epoch')
parser.add_argument('--bs', default=16, type=int, help='Mini-batch size')
parser.add_argument('--lr', default=9e-4, type=float, help='Initial learning rate')
parser.add_argument('--d_rnn', default=128, type=int, help='Hidden size dimension of bi-rnn')
parser.add_argument('--mlp', default=[128, 32], type=list, help='Dimensions of MLP added on top of bi-rnn.')
parser.add_argument('--rnn_type', default='gru', choices=['gru', 'lstm'], type=str, help='Type of rnn.')
parser.add_argument('--n_rnn', default=3, type=int, help='Number of bi-rnn layers')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
parser.add_argument('--seed', default=1000, type=int, help='seed for initializing training.')
parser.add_argument('--channel_bits', default = 24, type=int,help='sequence length')
parser.add_argument('--Nc', default = 6, type=int,help='marker frequency')
parser.add_argument('--max_pi',default = 0.05,type=float)
parser.add_argument('--max_pd',default = 0.05,type=float)

def func_to_maximize(x, bits):
    pd, pi = x
    return -(bits * (1 - pd) / (1 - pi) + 4.89 * bits * pi * (1 - pd) / (1 - pi)**2 + bits * pd * (1 - pd))  # Negative for maximization

def get_safety_bits(pi_range, pd_range, bits):
    # Set the bounds for x1 and x2
    bounds = [(pd_range[0], pd_range[1]), (pi_range[0], pi_range[1])]

    # Initial guesses for the variables
    initial_guesses = [
        [pd_range[0], pd_range[0]], [pd_range[0], pi_range[0]],
        [pd_range[1], pi_range[1]], [pd_range[1], pi_range[0]]
    ]

    max_point = None
    max_value = -np.inf  # Start with a very low value

    # Try multiple initial guesses
    for guess in initial_guesses:
        # Pass 'bits' as an additional parameter using a lambda function
        result = minimize(lambda x: func_to_maximize(x, bits), guess, bounds=bounds, method='Powell')

        # Check if we found a better maximum
        if -result.fun > max_value:
            max_value = -result.fun  # Store the actual maximum value
            max_point = result.x  # Store the point at which it occurs

    return int(np.ceil(max_value))

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

def main():
    args = parser.parse_args()

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    def train_step(input, labels, model, loss_fn, optimizer):
        """
        One training step for training
        ------------------------------------------
        args
        input : training batch
        labels : labels batch
        """
        model.train()
        optimizer.zero_grad()
        logits = model(input)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    epochs = args.epochs
    batch_size = args.bs
    steps = args.step
    Pd = args.training_Pd
    Ps = args.training_Ps
    Pi = args.training_Pi
    max_pi = args.max_pi
    max_pd = args.max_pd
    Nc = args.Nc
    marker_sequence = args.marker_sequence
    channel_bits = args.channel_bits
    pi_range = [0,max_pi]
    pd_range = [0,max_pd]
    
    ex_seq,_ = insert_regular_markers(np.random.randint(0,2,size=(1,channel_bits)),Nc,marker_sequence)
    safety_bits = get_safety_bits(pi_range,pd_range,ex_seq.shape[1])
    # Define the estimator model
    model = BI_Estimator(input_size=safety_bits,actual_size=ex_seq.shape[1], d_rnn=args.d_rnn, d_mlp=args.mlp, num_bi_layers=args.n_rnn, rnn_type=args.rnn_type)
    if args.loss == 'mse':
        loss_fn = nn.MSELoss()
    elif args.loss == 'bce':
        loss_fn = nn.BCELoss(reduction = 'mean')
    optimizer = optim.Adam(model.parameters(), lr=args.lr,eps=1e-7)
    # Set optimizer with the lr specified!
    
    
    best_acc = 0
    best_epoch = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        start_time = time.time()
        train_loss_total = 0
        train_acc_total = 0

        # Iterate over batches
        for step in range(steps):
            # create training batch
            trainX_batch, labels, _ = create_batch(m_total=channel_bits, num_code=batch_size,safety_bits=safety_bits,
                                                        Pd=Pd, Pi=Pi, Ps=Ps, Nc=Nc, marker_sequence=marker_sequence)
            trainX_batch = torch.tensor(trainX_batch, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.float32)

            train_loss = train_step(trainX_batch, labels, model, loss_fn, optimizer)
            train_loss_total += train_loss

            with torch.no_grad():
                model.eval()
                logits = model(trainX_batch)
                predictions = (logits > 0.5).float()
                
                train_acc = (predictions == labels).float().mean().item()
                train_acc_total += train_acc


        train_loss = train_loss_total / steps
        train_acc = train_acc_total / steps

        print(f"{time.time() - start_time:.2f}s - train loss: {train_loss:.4f} - train acc: {train_acc:.4f}")
        if train_acc > best_acc:
            best_acc = train_acc
            best_epoch = epoch
            torch.save(model.state_dict(),'{}bit_{}bit_estimator.pth'.format(channel_bits,safety_bits))
        print('Best Accuracy is {} at epoch {}'.format(best_acc, best_epoch+1))
        
if __name__ == '__main__':
    main()
