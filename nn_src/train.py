
import sys
import argparse
import math
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from util import str2bool, str2ints, count_parameters
from prepare import prepare_data
from trenc import TrEnc
from cnn import CNN

def parse_args(arg_values_to_parse):
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=1) 
    parser.add_argument('--early-stopping-patience', type=int, default=0)
    parser.add_argument('--batch-size', dest='bs', type=int, default=32) 
    parser.add_argument('--lr', type=float, default=5e-5) 
    
    parser.add_argument('--model', type=str, default='TrEnc')
    parser.add_argument('--tokenizer', type=str, default='basic')
    parser.add_argument('--vocab-size',    type=int, default=10_000) 
    parser.add_argument('--max-input-len', type=int, default=64) 
    
    # model
    parser.add_argument('--dropout', dest='m_dropout',type=float, default=0.2) 
    parser.add_argument('--layers',  dest='m_layers',type=int, default=1) 
  
    # dim is used for embedings. If a TrEnc, then embed_size = dim*heads
    parser.add_argument('--dim', dest='m_d',     type=int, default=64) 
    
    # model trenc
    parser.add_argument('--heads',   dest='m_heads', type=int, default=8) 
    parser.add_argument('--ff-dim',  dest='m_ff_dim',type=float, default=4.) 
    parser.add_argument('--use-cls-token', dest='m_use_cls_token', type=str2bool, default=True)
    parser.add_argument('--use-pos-enc',   dest='m_use_pos_enc',   type=str2bool, default=False)
    parser.add_argument('--clf-scale',     dest='m_clf_scale', type=float, default=1.0)
    
    # model cnn
    parser.add_argument('--kernel-sizes', dest='m_kernel_sizes', type=str2ints, default='[1, 3]')
    parser.add_argument('--num-filters',  dest='m_num_filters',  type=str2ints, default='[48, 48]')
    parser.add_argument('--strides',      dest='m_strides',      type=str2ints, default='[1, 1]')
    
    parser.add_argument('--dummy', dest='_ignore',type=float, default=0.) 
    
    
    # FIXME: Should I add a separate pooler?
        
    return parser.parse_args(arg_values_to_parse)

def main(*arg_values_to_parse):
    
    args = parse_args([str(la) for la in arg_values_to_parse])
    print('Arguments:', args)
  
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    data = prepare_data(tokenizer_type=args.tokenizer,
                        batch_size=args.bs, 
                        vocab_size=args.vocab_size, 
                        max_input_len=args.max_input_len,
                        device=dev)
    
    ### Model
    # Instantiate model and pass all command line args that start with m_
    model_args = {
        'num_classes': 2, # neg, pos 
        'vocab_size': len(data.vocabulary.get_itos()),
        **{k[2:]: v for k, v in vars(args).items() if k.startswith('m_')}
    }
    
    model_cls = None
    if args.model == 'TrEnc':
        model_cls = TrEnc
    elif args.model == 'CNN':
        model_cls = CNN
    
    model = model_cls(**model_args).to(dev)

    print('Model instantiated:', model)

    # prints 'learnable: ddd' to stdout
    count_parameters(model)[1]
    
    ### Train Loop
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction='sum')
        
    scaler = torch.cuda.amp.GradScaler()
    
    best_valid_loss = math.inf
    best_epoch = None
    
    #FIXME: documentation.
    best_result_store = None 
    
    for epoch in range(args.epochs):
        model.train()
        train_loss_epoch = 0.
        train_count_epoch = 0
        started = time()
    
        for i, (Yb, Xb) in enumerate(data.train_dl): 
            
            with torch.cuda.amp.autocast():
                logits = model(Xb)
            
                loss = criterion(logits, Yb)
            
            scaler.scale(loss).backward()
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_epoch  += loss.item() 
            train_count_epoch += len(Yb)
            optimizer.zero_grad()

            if i % 50 == 0: 
                print(f'i: {i:4d}: batch_train_loss: {loss/len(Yb):6.4f}')
                
        model.eval()
        with torch.no_grad():
            valid_count_epoch = 0
            matched_epoch = 0
            valid_loss_epoch = 0.
            
            for i, (Yb, Xb) in enumerate(data.valid_dl):
                with torch.cuda.amp.autocast():
                    logits = model(Xb)
                    valid_loss_epoch += criterion(logits, Yb).item()
                    
                    predictions = logits.argmax(-1)
                    matched_epoch += (predictions == Yb).sum().item()
                    valid_count_epoch += len(Yb)

            acc = matched_epoch/valid_count_epoch
        
        log_message = f'ep: {epoch} train_loss: {train_loss_epoch/train_count_epoch:6.5f} valid_loss: {valid_loss_epoch/valid_count_epoch:6.5f} valid_acc: {acc:5.4f} took: {time()-started:5.3f}s'
        print(log_message)
        
        if valid_loss_epoch < best_valid_loss:
            best_valid_loss = valid_loss_epoch
            best_result_store = log_message
            best_epoch = epoch
            
            print(f'epoch {epoch} was the best epoch so far.')
            
        # early stopping 
        print(f'best epoch {best_epoch}, epoch {epoch}, diff {(epoch-best_epoch)}')
        if best_epoch and epoch >= args.early_stopping_patience-1 and (epoch-best_epoch) >= args.early_stopping_patience:
            print(f'No progress for {args.early_stopping_patience} epochs. Stopping early.')
            break
                
    print('End of training.')
    print('Re-reporting best loss epoch:')
    print(best_result_store)

if __name__ == '__main__':
    print('main', sys.argv)
    main(*sys.argv[1:])    
