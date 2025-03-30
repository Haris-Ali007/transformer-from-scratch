import torch
import math
import torch.nn.functional as F

"""
Next to do:
- Then create training setup
"""

network_config={
    'vocab': 50000,
    'inner_states': 512,
    'num_heads': 8,
    'layers': 6
}


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
    
    def forward(self, input_tokens):
        num_tokens = input_tokens.shape[-1]
        pe = torch.zeros(size=(num_tokens, self.d))
        positions = torch.arange(0, num_tokens).unsqueeze(1)
        # so for every i (the index of 768th dimension vector) we will compute sines and cosines
        division_factor = 1000**(-(2*torch.arange(0, self.d//2).float()) / self.d)
        pe[:, 0::2] = torch.sin(positions*division_factor)
        pe[:, 1::2] = torch.cos(positions*division_factor)
        return pe
    

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads, inner_states, cross_attn=False):
        super().__init__()
        self.num_heads = num_heads
        self.d_dim = inner_states
        self.cross_attn = cross_attn
        self.proj1 = torch.nn.Linear(self.d_dim, self.d_dim)
        self.proj2 = torch.nn.Linear(self.d_dim, self.d_dim)
        self.proj3 = torch.nn.Linear(self.d_dim, self.d_dim)

    @staticmethod
    def split_states(x, heads):
        *shape, last_dim = list(x.shape)
        shape.extend([heads, last_dim//heads])
        return torch.reshape(x, shape)

    @staticmethod    
    def merge_heads(x):
        x = torch.transpose(x, 2, 1)
        *shape, n, m = list(x.shape)
        shape.extend([n*m])
        return x.reshape(shape)

    def split_heads(self, x):
        # splits x with shape [b, s, c] --> [b, h, s, c]
        x = self.split_states(x, self.num_heads)
        return torch.transpose(x, 1, 2)

    def scaled_dot_product(self, query, key, value, mask):
        qk_map = torch.matmul(query, key.mT)
        qk_map = F.softmax((qk_map/math.sqrt(self.d_dim)), dim=-1) # softmax against queries
        # If you set certain values to -inf before applying softmax, 
        # those values will effectively become zero in the output 
        # probabilities because: e^-inf = 0 where as e^0 = 1
        qk_map = qk_map.masked_fill(mask==0, 1e-9)
        context_vector = torch.matmul(qk_map, value)
        return context_vector, qk_map
    
    def forward(self, x, mask, enc_context=None):
        if self.cross_attn:
            q = self.split_heads(self.proj1(x))
            k = self.split_heads(self.proj2(enc_context))
            v = self.split_heads(self.proj3(enc_context))
            attn_output, attn_map = self.scaled_dot_product(q, k, v, mask)
        else:
            q = self.split_heads(self.proj1(x))
            k = self.split_heads(self.proj2(x))
            v = self.split_heads(self.proj3(x))
            attn_output, attn_map = self.scaled_dot_product(q, k, v, mask)
        return self.merge_heads(attn_output), attn_map


class FeedForwardLayer(torch.nn.Module):
    def __init__(self, dff=2048, inner_states=512):
        super().__init__()   
        self.dff = dff
        self.inner_states = inner_states
        self.lin1 = torch.nn.Linear(in_features=self.inner_states, out_features=self.dff)
        self.lin2 = torch.nn.Linear(in_features=self.dff, out_features=self.inner_states)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        xproj = self.lin1(x)
        xproj = self.relu(xproj)
        xproj = self.lin2(xproj)
        return xproj

class Encoder(torch.nn.Module):
    def __init__(self, num_heads=8, inner_states=512):
        super().__init__()
        self.num_heads = num_heads
        self.inner_states = inner_states
        self.layer_norm = torch.nn.LayerNorm(inner_states)
        self.mha = MultiHeadAttention(self.num_heads, self.inner_states)
        self.ffn = FeedForwardLayer()
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x, mask):
        x_attn, attn_map = self.mha(x, mask)
        x_attn = self.dropout(x_attn) + x
        x_attn = self.layer_norm(x_attn)
        x_attn_proj = self.ffn(x_attn)
        x_attn_proj = self.dropout(x_attn_proj) + x_attn
        return x_attn_proj, attn_map


class Decoder(torch.nn.Module):
    def __init__(self, num_heads=8, inner_states=512):
        super().__init__()
        self.num_heads = num_heads
        self.inner_states = inner_states
        self.layer_norm = torch.nn.LayerNorm(inner_states)
        self.mha_self = MultiHeadAttention(self.num_heads, self.inner_states)
        self.mha_cross = MultiHeadAttention(self.num_heads, self.inner_states,
                                            cross_attn=True)
        self.dropout = torch.nn.Dropout(0.1)
        self.ffn = FeedForwardLayer()

    def forward(self, x, dec_mask, enc_dec_mask, enc_context):
        x_attn, _ = self.mha_self(x, dec_mask)
        x_attn = self.dropout(x_attn) + x
        x_attn = self.layer_norm(x_attn)
        x_attn_cross, attn_map_cross = self.mha_cross(x_attn, enc_dec_mask, enc_context)
        x_attn_cross = x_attn + self.dropout(x_attn_cross)
        x_attn_cross = self.layer_norm(x_attn_cross)
        x_attn_proj = self.ffn(x_attn_cross)
        x_attn_proj = self.dropout(x_attn_proj) + x_attn_cross
        return x_attn_proj, attn_map_cross        


class Transformer(torch.nn.Module):
    def __init__(self, vocab_src, vocab_trgt, num_heads=8, 
                 inner_states=512, layers = 6, pad_token_id=0):
        super().__init__()
        self.num_heads = num_heads
        self.pad_token_id = pad_token_id
        self.embeddings_src = torch.nn.Embedding(num_embeddings=vocab_src, 
                                            embedding_dim=inner_states)
        self.embeddings_trgt = torch.nn.Embedding(num_embeddings=vocab_trgt, 
                                    embedding_dim=inner_states)
        self.positional_enc = PositionalEncoding(d=inner_states)
        self.encoder_layers = [Encoder(num_heads, inner_states) for _ in range(layers)]
        self.decoder_layers = [Decoder(num_heads, inner_states) for _ in range(layers)]
        self.linear = torch.nn.Linear(inner_states, vocab_trgt)
        self.dropout = torch.nn.Dropout(0.1)
        self.softmax = torch.nn.Softmax(dim=-1)

    def create_padding_mask(self, sequence):
        # input sequence of shape [B, S]
        # output mask shape [B, H, S, S]
        pad_mask = (sequence != self.pad_token_id)
        pad_mask = pad_mask.float().unsqueeze(1)
        pad_mask = pad_mask * pad_mask.transpose(1, 2)
        pad_mask = pad_mask.unsqueeze(1)
        pad_mask = pad_mask.expand(-1, self.num_heads, -1, -1)    
        return pad_mask 
    
    def create_decoder_mask(self, sequence):
        # input sequence of shape [B, S]
        # output sequence mask [B, H, S, S]
        batch, sequence_length = sequence.shape
        padding_mask = self.create_padding_mask(sequence)
        causal_mask = torch.tril(torch.ones(batch, self.num_heads, sequence_length, sequence_length))
        mask = causal_mask * padding_mask
        return mask # B, H, S, S

    def create_encoder_decoder_mask(self, enc_sequence, dec_sequence):
        enc_mask = (enc_sequence != self.pad_token_id)
        dec_mask = (dec_sequence != self.pad_token_id)
        enc_mask = enc_mask.float().unsqueeze(1) # expands [B, S] to [B, S, 1]
        dec_mask = dec_mask.float().unsqueeze(1) # expands [B, S] to [B, 1, S]
        mask = enc_mask * dec_mask.transpose(1, 2)
        mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        batch, _, s1, s2 = mask.shape
        causal_mask = torch.tril(torch.ones(batch, self.num_heads, s1, s2))
        mask = mask * causal_mask
        return mask


    def forward(self, x, y):
        src_embeds = self.dropout(self.embeddings_src(x) + self.positional_enc(x))
        trgt_embeds = self.dropout(self.embeddings_trgt(y) + self.positional_enc(y))
        enc_mask = self.create_padding_mask(x)
        dec_mask = self.create_decoder_mask(y)
        enc_dec_mask = self.create_encoder_decoder_mask(x, y)
        for enc_layer in self.encoder_layers:
            src_embeds, attn_map = enc_layer(src_embeds, enc_mask)
        for dec_layer in self.decoder_layers:
            trgt_embeds, attn_map = dec_layer(trgt_embeds, dec_mask, enc_dec_mask, src_embeds)
        output = self.linear(trgt_embeds)
        return output
    
if __name__=="__main__":
    x = torch.randint(low=0, high=100, size=(1, 1024))
    y = torch.randint(low=0, high=100, size=(1, 512))

    model = Transformer(vocab=network_config['vocab'], 
                        inner_states=network_config['inner_states'],
                        num_heads=network_config['num_heads'],
                        layers=network_config['layers'])
    output = model(x, y)
    print(output.shape)

