from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

import math
import numpy as np
import tiktoken

@dataclass
class GPTConfig:
    block_size: int=1024
    vocab_size: int=50257 # 50,000 BPE merge, 256 Byte, 1 EOS
    n_layer: int=12
    n_head: int=12
    n_embd: int=768


class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        # make sure n_embd % n_head ==0
        # the dimension include Q,K,V
        # every embeding has its q,k,v
        self.c_attn=nn.Linear(config.n_embd,config.n_embd*3)

        self.c_proj=nn.Linear(config.n_embd,config.n_embd)

        self.n_head=config.n_head
        self.n_embd=config.n_embd

        # mask, because the former words can't see later words
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size))
                             .view(1,1,config.block_size,config.block_size))
        
    def forward(self,x):
        # B: batch size, T: sequence length(time), 
        # C: number of channels(the dimension of embedding)
    
        B,T,C=x.size()
        qkv=self.c_attn(x)

        # the dim0 is batch, dim1 is length of sequence
        q,k,v=qkv.split(self.n_embd,dim=2)
        # change their shape
        Q=q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        K=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        V=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)

        # sqrt(d_k), in case big dimension lead to big digits
        # each head query and match it's corresponding K
        att=(Q@K.transpose(-2,-1)*(1.0/math.sqrt(K.size(-1))))
        
        # use mask to change future Q@V^T to -inf, ensure the softmax become zero
        att=att.masked_fill(self.bias[:,:,:T,:T]==0,float('-inf'))

        att=F.softmax(att,dim=-1)
        y=att@V

        # y:(B,head,T,d_k)-> y:(B,T,head,d_k)
        y=y.transpose(1,2).contiguous().view(B,T,C)

        # change the dimension back
        y=self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd,4*config.n_embd)
        # default parem is Cumulative Distribution Function for Gaussian Distribution.
        self.gelu=nn.GELU(approximate='tanh')
        self.c_proj=nn.Linear(config.n_embd*4,config.n_embd)

    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        return self.c_proj(x)


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embd)
        self.attn=CausalSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)

    def forward(self,x):
        x=x+self.attn(self.ln_1(x))
        x=x+self.mlp(self.ln_2(x))
        return x
    


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config=config

        self.transformer=nn.ModuleDict(dict(
            token_embd=nn.Embedding(config.vocab_size,config.n_embd),
            position_embd=nn.Embedding(config.block_size,config.n_embd),
            h=nn.ModuleList([Block(config) for i in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)


    def forward(self,idx):
        B,T=idx.size()
        assert T<=self.config.block_size, f"Sequence with the length of {T} exceed the block_size"

        pos=torch.arange(0,T,dtype=torch.long,device=idx.device)

        pos_emb=self.transformer.position_embd(pos)
        tok_emb=self.transformer.token_embd(idx)

        x=tok_emb+pos_emb

        for block in self.transformer.h:
            x=block(x)

        x=self.transformer.ln_f(x)
        logits=self.lm_head(x)

        return x

device="cpu"
if torch.cuda.is_available():
    device="cuda"
print(f"Using device: {device}.")


num_return_sequences=5
max_length=30

model=GPT(GPTConfig())
model.to(device)

prompt="I am a good man."

# call gpt2 encoder
enc=tiktoken.get_encoding('gpt2')
tokens=enc.encode(prompt)
tokens=torch.tensor(tokens,dtype=torch.long)
# copy(unsqueeze to batch size)
tokens=tokens.unsqueeze(0).repeat(num_return_sequences,1)
x=tokens.to(device)




torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1)<max_length:
    with torch.no_grad():

        logits=model(x)
        # (B,T,vocab_size) -> (B,vocab_size)
        logits=logits[:,-1,:]

        prob=F.softmax(logits,dim=-1)
        topk_probs,topk_indices=torch.topk(prob,50,dim=-1)
        # select a token fron the distribution
        # Randomly sample index 
        ix=torch.multinomial(topk_probs,1)
        # get specific token with its index 
        xcol=torch.gather(topk_indices,-1,ix)

        x=torch.cat((x,xcol),dim=1)


for i in range(num_return_sequences):
    tokens=x[i,:30].tolist()
    decoded=enc.decode(tokens)
    print(">",decoded)