import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math

## building a transformer

############Encoder################

#### 1- first is input embeddings
d = 64
input_tokens = 10
x = torch.randint(low=0, high=10, size=(1, 10))
# here num_embeddings will be our total vocabularysize and the other will be vector size for each token
embed_layer = nn.Embedding(num_embeddings=10, embedding_dim=d) 
xe = embed_layer(x)

#### 2- We will generate positional encodings
def positional_encodings(input_tokens, d):
    pe = torch.zeros(size=(input_tokens.shape[-1], d))
    positions = torch.arange(0, input_tokens.shape[-1]).unsqueeze(1)
    # so for every i (the index of 768th dimension vector) we will compute sines and cosines
    division_factor = 1000**(-(2*torch.arange(0, d//2).float()) / d)
    pe[:, 0::2] = torch.sin(positions*division_factor)
    pe[:, 1::2] = torch.cos(positions*division_factor)
    return pe

t = positional_encodings(x, d)
plt.imshow(t[:input_tokens, :].numpy(), cmap='viridis', aspect='auto')
plt.show()
# x_encoded = xe + pe


#### 3- Next stage Multi-Head attention
# first step is implementing attention

# c = attention(xe, 5, 6, 6, 6)
# c.shape

# so we will project out x into space x*3 and in this way 
# we save the computation of Q=x*w_q, K=x*w_k, V=x*w_k 
# by doing direct multiplication 

def MultiHeadAttention(num_heads, d_dim, x, mask=None, cross_attn=False, enc_context=None):

    def split_states(x, heads):
        *shape, last_dim = list(x.shape)
        shape.extend([heads, last_dim//heads])
        return torch.reshape(x, shape)

    def split_heads(x):
        # splits x with shape [b, s, c] --> [b, h, s, c]
        x = split_states(x, num_heads)
        return torch.transpose(x, 1, 2)
    
    def merge_heads(x):
        x = torch.transpose(x, 2, 1)
        *shape, n, m = list(x.shape)
        shape.extend([n*m])
        return x.reshape(shape)

    def scaled_dot_product(d_embedding, query, key, value, mask):
        qk_map = torch.matmul(query, key.mT)
        qk_map = F.softmax((qk_map/math.sqrt(d_embedding)), dim=-1) # softmax against queries
        if mask != None:
            # If you set certain values to -inf before applying softmax, 
            # those values will effectively become zero in the output 
            # probabilities because: e^-inf = 0 where as e^0 = 1
            qk_map = qk_map.masked_fill(mask==0, float('-inf'))
        context_vector = torch.matmul(qk_map, value)
        return context_vector, qk_map

    if cross_attn:
        proj1 = nn.Linear(in_features=d_dim, out_features=d_dim*2)
        proj2 = nn.Linear(in_features=d_dim, out_features=d_dim)
        kv = proj1(enc_context)
        q = proj2(x)
        k, v = map(split_heads, kv.chunk(2, dim=-1))
        attn_output, attn_map = scaled_dot_product(d_dim, q, k, v, mask)
    else:
        proj = nn.Linear(in_features=d_dim, out_features=d_dim*3)
        qkv = proj(x)
        q, k, v = map(split_heads, qkv.chunk(3, dim=-1))
        attn_output, attn_map = scaled_dot_product(d_dim, q, k, v, mask)
    return merge_heads(attn_output), attn_map
## add mask logic for causal attention

# to make it flexible so we can mask other stuff
# too we apply mask from outside
batch = 1
num_heads = 8
d_dim = 64
input_tokens = 10
mask = torch.tril(torch.ones(batch, num_heads, input_tokens, input_tokens))
z, attn_map = MultiHeadAttention(num_heads=num_heads, d_dim=d_dim, x=xe, mask=mask)

### Feed Forward layer
# it is also called conv1d as we pass each 
# matrix location through a linear transformation
def FFN(x, dff=2048):
    nx = x.shape[-1]
    lin_proj1 = nn.Linear(in_features=nx, out_features=dff)
    lin_proj2 = nn.Linear(in_features=dff, out_features=nx)
    relu = nn.ReLU()
    z = lin_proj1(x)
    z = relu(z)
    z = lin_proj2(z)
    return z
# t = FFN(xe)

### Encoder
def Encoder(x, num_heads, d):
    layer_norm = nn.LayerNorm(d) 
    z, attn_map = MultiHeadAttention(num_heads, d, x)
    z = z + x
    z = layer_norm(z)
    z = z + FFN(z)
    return z, attn_map

z, attn_map = Encoder(xe, num_heads, d)


def Decoder(y, num_heads, d, input_context):
    layer_norm = nn.LayerNorm(d)
    z, _ = MultiHeadAttention(num_heads, d, y)
    z = layer_norm(z + y)
    z2 = MultiHeadAttention(num_heads, d, y, cross_attn=True, enc_context=input_context)
    z2 = layer_norm(z2 + z)
    z3 = FFN(z2)
    z3 = z2 + z3
    return z3


def create_padding_mask(sequence, num_heads, pad_token_id=0):
    # input sequence of shape [B, S]
    # output mask shape [B, H, S, S]
    pad_mask = (sequence != pad_token_id)
    pad_mask = pad_mask.float().unsqueeze(1)
    pad_mask = pad_mask * pad_mask.transpose(1, 2)
    pad_mask = pad_mask.unsqueeze(1)
    pad_mask = pad_mask.expand(-1, num_heads, -1, -1)    
    pad_mask = pad_mask.masked_fill(pad_mask==0, 1e-9)
    return pad_mask 
    
def create_decoder_mask(sequence, num_heads, pad_token_id=0):
    # input sequence of shape [B, S]
    # output sequence mask [B, H, S, S]
    batch, sequence_length = sequence.shape
    padding_mask = create_padding_mask(sequence, num_heads, pad_token_id)
    causal_mask = torch.tril(torch.ones(batch, num_heads, sequence_length, sequence_length))
    mask = causal_mask * padding_mask
    mask = mask.masked_fill(mask==0, 1e-9)
    return mask # B, H, S, S

def create_encoder_decoder_mask(enc_sequence, dec_sequence, num_heads, pad_token_id=0):
    enc_mask = (enc_sequence != pad_token_id)
    dec_mask = (dec_sequence != pad_token_id)
    enc_mask = enc_mask.float().unsqueeze(1) # expands [B, S] to [B, S, 1]
    dec_mask = dec_mask.float().unsqueeze(1) # expands [B, S] to [B, 1, S]
    mask = enc_mask * dec_mask.transpose(1, 2)
    mask = mask.unsqueeze(1).expand(-1, num_heads, -1, -1)
    batch, _, s1, s2 = mask.shape
    causal_mask = torch.tril(torch.ones(batch, num_heads, s1, s2))
    mask = mask * causal_mask
    # mask = mask.masked_fill(mask==0, 1e-9)
    return mask

    # when encoder context is passed to decoder

# bool_mask = bool_mask.float()
# padding_mask = bool_mask.masked_fill(bool_mask==False, 1e-9)
enc_seq = torch.tensor([[2, 4, 7, 8, 7, 8, 0, 0, 0, 0]])
dec_seq = torch.tensor([[2, 4, 7, 8, 0, 0, 0, 0, 0]])
# enc_mask = (enc_seq != 0)
# dec_mask = (dec_seq != 0)
# enc_mask = enc_mask.float().unsqueeze(1)
# dec_mask = dec_mask.float().unsqueeze(1)
# # torch.matmul(enc_mask, dec_mask)
# t = dec_mask * enc_mask.transpose(1, 2)
# t.shape
mask = create_encoder_decoder_mask(enc_seq, dec_seq, num_heads=2)
create_padding_mask(enc_seq, 2)