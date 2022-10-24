
import random
from pathlib import Path
from collections import namedtuple

import torch
from torch.utils.data import DataLoader
import torchdata.datapipes as dp

from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from torchtext.data.functional import generate_sp_model, load_sp_model, sentencepiece_tokenizer

def _get_sp_tokenizer(train_dp):
    
    base = Path('data')
    base.mkdir(exist_ok=True)

    input_file = base/'sp_input'
    output_dir = base/'sp_out'

    if not input_file.exists():
        with open(input_file, 'w') as f:
            for _, text in train_dp:
                f.write(text+'\n')
    
    if not output_dir.exists():    
        output_dir.mkdir(exist_ok=True)

        # Outputs, among others: Updating active symbols. max_freq=705466 min_freq=168    
        generate_sp_model(str(input_file), vocab_size=4000, model_type='bpe', model_prefix=f'{output_dir}/sp_out')
        
    sp_model = load_sp_model(str(output_dir/'sp_out.model'))
    
    tokenizer_fn = sentencepiece_tokenizer(sp_model)

    # Adapt from batch to single instance, like with the basic english tokenizer
    def tokenizer_wrapper(doc):
        return next(tokenizer_fn([doc]))
    return tokenizer_wrapper

def _get_tokenizer_and_vocab(tokenizer_type, vocab_size, train_dp):
    
    if tokenizer_type.lower() == 'basic':
        tokenizer = get_tokenizer('basic_english')
    elif tokenizer_type.lower() == 'bpe': 
        tokenizer = _get_sp_tokenizer(train_dp)
    else:
        raise ValueError(f'{tokenizer_type} is not a valid tokenizer type. Valid ones are: basic, bpe.')
    
    vocab = build_vocab_from_iterator(
        map(tokenizer, (text for (_, text) in train_dp)), specials=['<UNK>', '<PAD>'], 
        max_tokens=vocab_size
    )
    vocab.set_default_index(vocab['<UNK>'])
    
    return tokenizer, vocab


def prepare_data(tokenizer_type, batch_size, vocab_size, max_input_len, device):
    
    train_dp, valid_dp = IMDB(root='imdb_data', split=('train', 'test'))
    
    tokenizer, vocab = _get_tokenizer_and_vocab(tokenizer_type, vocab_size, train_dp)
    
    print(f'Vocabulary built with {len(vocab.get_itos())} tokens.')
    
    ### Data Loaders and methods
    def vectorize(y, x):
        y = 0 if y == 'neg' else 1
        return (y, vocab(tokenizer(x)))
    
    def collate(data):
        batch_len = min(max_input_len, max([len(x) for (_, x) in data]))
        
        padding = [1] * batch_len # <PAD> has index 1
        y, x = zip(*[(y, (x+padding)[:batch_len]) for (y, x) in data])
 
        return torch.LongTensor(y).to(device), torch.LongTensor(x).to(device)
    
    train_dl = DataLoader(
        dataset=[vectorize(y, x) for (y, x) in train_dp], 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True,
        collate_fn=collate)
    del train_dp
    
    valid_ds = [vectorize(y, x) for (y, x) in valid_dp]
    random.shuffle(valid_ds)
    valid_dl = DataLoader(
        dataset=valid_ds[:7500], 
                batch_size=batch_size*2,
                collate_fn=collate)  
    
    return namedtuple('Data', 'train_dl valid_dl vocabulary')(
        train_dl, valid_dl, vocab
    )
