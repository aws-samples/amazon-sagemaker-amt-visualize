
from ast import literal_eval

def count_parameters(m, verbose=True):
    total_count = 0
    learnable_count = 0
    if verbose:
        print('Parameters (name, tunable, count):')
    for n, p in m.named_parameters():
        count = p.data.numel() 
        if verbose:
            print(f' {n:60s} {p.requires_grad:5b} {count:>9d}')
        total_count += count
        if p.requires_grad:
            learnable_count += count
    if verbose:
        print(f'Total parameters: {total_count}, thereof learnable: {learnable_count}')
    return total_count, learnable_count

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def str2ints(v):
    assert isinstance(v, str)
    
    obj = literal_eval(v)
    assert isinstance(obj, list), f'parsed value needs to be a list, not {type(obj)}'
    for elem in obj:
        assert isinstance(elem, int)
    
    return obj
