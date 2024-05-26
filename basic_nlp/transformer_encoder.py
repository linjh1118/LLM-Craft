import torch
import torch.nn as nn
import math

class SelfAttentionLayer():
    def __init__(self, input_dim, n_head, dim_each_head):
        super(SelfAttentionLayer, self).__init__()
        self.Wq = nn.Linear(input_dim, n_head * dim_each_head)
        self.Wk = nn.Linear(input_dim, n_head * dim_each_head)
        self.Wv = nn.Linear(input_dim, n_head * dim_each_head)
        self.Wo = nn.Linear(n_head * dim_each_head, n_head * dim_each_head)

        self.input_dim, self.n_head, self,dim_each_head = \
              input_dim, n_head, dim_each_head

    def divide_heads_and_move_forward(self, query):
        new_shape = query.shape()[:-1] + [self.n_head, self.dim_each_head]
        query = query.view(new_shape)
        
        query = query.permute(0, 2, 1, 3)
        return query
    
    def merge_heads(self, value):
        bs, seq_len = value.shape()[0], value.shape()[2]
        new_shape = [bs, seq_len, self.n_head * self.dim_each_head]
        value = value.permute(0, 2, 1, 3)
        value = value.view(new_shape)
        return value


    def forward(self, x, attention_mask):
        # 1. get multi-head
        query, key, value = self.Wq(x), self.Wk(x), self.Wv(x)
        query, key, value = [self.divide_heads_and_move_forward(i) for i in [query, key, value]]
        # 2. get scores
        scores = torch.matmul(query, key.transpose(-1, -2)) # [seq_len, dim] * [dim, seq_len]
        scores = scores / math.sqrt(self.input_dim)
        scores += attention_mask
        scores = nn.functional.softmax(scores, dim=-1)
        # 3. merge value by scores
        x = torch.matmul(scores, value)
        x = self.merge_heads(x)
        x = self.Wo(x)
        return x, scores

        

class TransformerEncoderLayer():
    def __init__(self, emb_dim, hid_dim_ffn, n_head, dim_each_head) -> None:
        super(TransformerEncoderLayer, self).__init__()
        # 1. Self-Attention
        self.attention_layer = SelfAttentionLayer(emb_dim, n_head, dim_each_head)
        
        # 2. FNN
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, hid_dim_ffn),
            nn.LeakyReLU(),
            nn.Linear(hid_dim_ffn, emb_dim)
            )
    
        # 1.5 & 2.5 
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        # 1. self-attn
        x_attn = self.attention_layer(x)
        # 1.5 add & norm
        x = self.layernorm(x + x_attn)
        # 2. ffn
        x_ffn = self.ffn(x)
        # 2.5 add & norm
        x = self.layernorm(x + x_ffn)
        return x

class TransformerEmbedding():
    def __init__(self, vocab_size, max_pos, embed_dim, pad_token_id):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.pos_embedding = nn.Embedding(max_pos, embed_dim)
    
    def forward(self, input_ids, position_ids):
        inputs_embeds = self.token_embedding(input_ids)
        pos_embeds = self.pos_embedding(position_ids)
        return inputs_embeds + pos_embeds

class TransformerEncoder():
    """手撸一个Encoder"""
    def __init__(self, n_layers):
        super(TransformerEncoder, self).__init__()
        self.embedding_layer = TransformerEmbedding()
        self.layers = nn.Sequential(
            *[TransformerEncoderLayer() for _ in range(n_layers)]
        )
        self.decoding_head = None
        
    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.layers(x)
        if self.decoding_head:
            x = self.decoding_head(x)
        return x

