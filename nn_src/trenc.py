
import math

import torch
from torch import nn
import torch.nn.functional as F 
from torch import log

class EncoderLayer(nn.Module):
    def __init__(self,
                 d,
                 num_heads,
                 dropout,
                 ff_dim, 
                 res_connections):

        super().__init__()

        projection_size = d // num_heads

        self.WQ = nn.Parameter(torch.randn(num_heads, d, projection_size)/math.sqrt(d))
        self.WK = nn.Parameter(torch.randn(num_heads, d, projection_size)/math.sqrt(d))
        self.WV = nn.Parameter(torch.randn(num_heads, d, projection_size)/math.sqrt(d))
        self.WO = nn.Parameter(torch.randn(num_heads, projection_size, d)/math.sqrt(d))

        self.attn_norm = nn.LayerNorm(d)
        self.ff_norm   = nn.LayerNorm(d)
        self.dropout   = nn.Dropout(p=dropout)
        self.ff        = nn.Sequential(
            nn.Linear(d, int(ff_dim*d)), 
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(int(ff_dim*d), d)
        )

        self.res_connections = res_connections
        # Why just one ReLU/GELU, one dropout? Attention is all you need: 4.3, 5.4. 
        
    def forward(self, input):

        X, prv_attention_values = input

        # b batch
        # t step
        # d dimension
        # p projected dimension
        # h head
        
        normalized_X = self.attn_norm(X) # Pre-Norm

        # Creating keys, queries and values from the normalized values, while
        # at the same time down projecting from d dimensions to p dimensions
        keys    = torch.einsum('btd,hdp->bthp', normalized_X, self.WK)
        queries = torch.einsum('btd,hdp->bthp', normalized_X, self.WQ)
        values  = torch.einsum('btd,hdp->bthp', normalized_X, self.WV)

        # Attention values are per position, not per dimension etc.
        dot = torch.einsum('bkhp,bqhp->bhqk', queries, keys)
        scaled_dot = dot / math.sqrt(values.size()[-1])
        attention_values = torch.softmax(scaled_dot, -1)

        assert torch.allclose(attention_values[0, 0, 0].sum(), torch.tensor(1.))

        new_values = torch.einsum('bkhp,bhqk->bqhp', values, attention_values)
        b, t, h, p = new_values.size()

        # Consolidate the invidual head outputs by concenating them
        # and then up projecting from p to d dimensions.
        seq_summary = torch.einsum('bthp,hpd->btd', new_values, self.WO)
        
        # Alternatively: 
        # Concat outputs of the individual heads
        # Using no learnable parameters
        # seq_summary = new_values.view(b, t, -1)

        assert seq_summary.size()[-1] == h*p == X.size()[-1]

        # Residual Connection and Pre-Norm
        if self.res_connections:
            pre_ff = X + self.ff_norm(seq_summary) 
        else:
            pre_ff = self.ff_norm(seq_summary) 

        prv_attention_values.append(attention_values)

        post_ff = self.ff(self.dropout(pre_ff))
        if self.res_connections:
            return post_ff + pre_ff, prv_attention_values
        else:
            return post_ff, prv_attention_values


class TrEnc(nn.Module):
    def __init__(self, 
                 num_classes, 
                 vocab_size, 
                 layers,
                 d,
                 heads,
                 dropout,
                 use_pos_enc,
                 use_cls_token,
                 clf_scale,
                 ff_dim, 
                 **kwargs):
        super().__init__()
        
        print(f'superfluous model parameters not used: {kwargs}')

        res_connections=True
        pos_encoding = 'sin' if use_pos_enc else None
        
        embed_size = d*heads 
        self.pooling  = 'cls' if use_cls_token else 'mean'
        
        self.res_connections = res_connections == True
  
        self.embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=1)
        self.encoder_layers = \
            nn.Sequential(
                *[EncoderLayer(embed_size, heads, dropout, ff_dim, res_connections) for _ in range(layers)],)

        # Empricially it was a toss-up if the additional LN helped or not
        # It is part of the Pre-LN Transformer though
                #nn.LayerNorm(embed_size))
        
        # REVIEW: Just have one matrix for all three weight matrices?
        # REVIEW: Maybe do not project the values, but q and k only
        
        self.dropout     = nn.Dropout(dropout)
        
        if clf_scale:
            intermediate_dim = int(embed_size*clf_scale)
            self.clf = nn.Sequential(
                nn.Linear(embed_size, intermediate_dim, bias=False),
                nn.LayerNorm(intermediate_dim),
                nn.Tanh(),
                nn.Dropout(dropout/2),
                nn.Linear(intermediate_dim, num_classes)
            )
        else:
            self.clf = nn.Linear(embed_size, num_classes)
            
        self.pos_encoder = PositionalEncoding(embed_size, dropout) if use_pos_enc else None

    def extra_repr(self):
        return f'residual_connection={self.res_connections}, pooling={self.pooling}'

    def forward(self, X, output_attentions=False):
        # TODO: Unclear how helpful dropout on embeddings is here. Same p? 
        emb = self.dropout(self.embeddings(X))

        if self.pos_encoder:
            emb = self.pos_encoder(emb) 

        z, attention_values = self.encoder_layers((emb, []))
        
        if self.pooling == 'cls':
            logits = self.clf(self.dropout(z[:, 0]))
        elif self.pooling == 'mean':
            logits = self.clf(self.dropout(z.mean(1)))
        else:
            raise ValueError('Only mean and cls are valid valued for pooling.')

        if output_attentions:
            # return attention as b, l, h, q, k
            return logits, torch.stack(attention_values, 0).transpose(0, 1)
        return logits

# From: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
