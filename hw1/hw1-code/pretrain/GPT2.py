from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

import math
import numpy as np
import tiktoken
import dataloader

'''set device'''
device="cpu"
if torch.cuda.is_available():
    device="cuda"
print(f"Using device: {device}.")

class TanhGeLU(nn.Module):
    def forward(self,input):
        return 0.5*input*(1.0+torch.tanh(math.sqrt(2.0/math.pi)*(input+0.044715*torch.pow(input,3.0))))

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
        self.c_proj.NANOGPT_SCALE_INIT=1
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
        # Flash attention
        # y=F.scaled_dot_product_attention(Q,K,V,is_causal=True)
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
        # use the flag, change the initialize std
        # consider to layeres
        self.c_proj.NANOGPT_SCALE_INIT=1

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
        # share token_embd parameter to output embedding
        # a single tensor
        self.transformer.token_embd.weight=self.lm_head.weight
        self.apply(self._init_weight)


    def _init_weight(self,module):
        if isinstance(module,nn.Linear):
            std=0.02
            if hasattr(module,'NANOGPT_SCALE_INIT'):
                # there are 2*layer_num layers actually
                std*=(2*self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)


    def forward(self,idx,targets=None):
        
        B,T=idx.size()
        assert T<=self.config.block_size, f"Sequence with the length of {T} exceed the block_size"
        pos=torch.arange(0,T,dtype=torch.long,device=idx.device)
        pos_emb=self.transformer.position_embd(pos)
        tok_emb=self.transformer.token_embd(idx)
        x=tok_emb+pos_emb
        # let x go through 12 layers
        for block in self.transformer.h:
            x=block(x)

        # Layernorm
        x=self.transformer.ln_f(x)
        # classifier 
        logits=self.lm_head(x)
        loss=None
        if targets is not None:
            loss=F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1))

        return logits,loss

import time
T,B=128,16
#num_return_sequences=5
#max_length=39
epoch_num=50
model=GPT(GPTConfig(vocab_size=50304))
model.to(device)
# model=torch.compile(model)
Train_loader=dataloader.DataLoaderLite(B,T)
'''load optimizer: Adam SGD'''
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(0.9,0.95),eps=1e-8)

max_lr=3e-4
min_lr=max_lr*0.1
warmup_steps=10
max_steps=50

def get_lr(iter_time):
    if iter_time<warmup_steps:
        return max_lr*(iter_time+1)/warmup_steps
    if iter_time>max_steps:
        return min_lr
    # use cosin decay to drop down lr
    decay_ratio=(iter_time-warmup_steps)/(max_steps-warmup_steps)
    assert 0<=decay_ratio<=1
    coeff=0.5*(1.0+math.cos(math.pi*decay_ratio))
    return min_lr+coeff*(max_lr-min_lr)

def training():
    for step in range(max_steps):
        t0=time.time()
        x,y=Train_loader.next_batch()
        x=x.to(device)
        y=y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device,dtype=torch.float16):
            logits,loss=model(x,y)
            #import code; code.interact(local=locals())
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        
        lr=get_lr(step)
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group['lr']=lr


        # update parameters based on gradients 
        optimizer.step()
        torch.cuda.synchronize()# wait for gpu to finish work
        t1=time.time()
        dt=(t1-t0)*1000
        token_per_sec=(Train_loader.B*Train_loader.T)/(t1-t0)
        # .item convert tensor to a single float
        print(f"step {step} | loss: {loss.item():.6f} | norm={norm:.4f} | dt is {dt:.4f} | tokens per sec: {token_per_sec:.2f}")

training()
import sys
sys.exit(0)


# set random seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1)<max_length:
    with torch.no_grad():
        logits,loss=model(x,y)
        # (B,T,vocab_size) -> (B,vocab_size)
        logits=logits[:,-1,:]
        print(loss)


        prob=F.softmax(logits,dim=-1)
        topk_probs,topk_indices=torch.topk(prob,50,dim=-1)
        # select a token fron the distribution
        # Randomly sample index 
        ix=torch.multinomial(topk_probs,1)
        # get specific token with its index 
        xcol=torch.gather(topk_indices,-1,ix)
        x=torch.cat((x,xcol),dim=1)


for i in range(3):
    tokens=x[i,:30].tolist()
    decoded=enc.decode(tokens)
    print(">",decoded)