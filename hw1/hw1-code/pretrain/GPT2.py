from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group,destroy_process_group
from torch.nn import functional as F

import torch
import torch.distributed as dist
import torch.nn as nn
import time
import math
import numpy as np
import tiktoken
import dataloader
import inspect
import sys
import os


'''tokenizer'''
enc=tiktoken.get_encoding('gpt2')

'''
set Distributed Data Parallel
'''
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    # rank multi device
    # we want them to run different data
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    # if not ddp, we are running on a single gpu, and one process
    ddp_rank=0
    ddp_local_rank=0
    ddp_world_size=1
    master_process=True
    device="cpu"
    if torch.cuda.is_available():
        device="cuda"
    print(f"Using device: {device}.")
# instead use "python", use "torchrun"



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

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # get candidate parameters which need gradients
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        # one dimension
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9,0.95), eps=1e-8)
        print(f"using fused AdamW: {use_fused}")
        return optimizer



'''
hyperparameters:
1 epoch has 'max_steps*total_batch_size=10B' tokens to train.
'''
total_batch_size=524288 # the power of 2 is better.  0.5 Million
B=16
T=128
grad_accu_steps=total_batch_size//(B*T*ddp_world_size) # need grad_accu_steps forward times 
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"calculate accumulated gradient steps: {grad_accu_steps}")
epoch_num=4
model=GPT(GPTConfig(vocab_size=50304))
model.to(device)
model=torch.compile(model)
python_compile=True
if ddp:
    model=DDP(model,device_ids=[ddp_local_rank]) # here need local rank
raw_model=model.module if ddp else model
max_lr=6e-4
min_lr=max_lr*0.1
warmup_steps=715
max_steps=19073
log_file="file_name"
log_dir="root"



'''create data loader'''
Train_loader=dataloader.DataLoaderLite(B,T,process_rank=ddp_rank,num_processes=ddp_world_size,split="train")
val_loader=dataloader.DataLoaderLite(B,T,process_rank=ddp_rank,num_processes=ddp_world_size,split="val")
 

'''use my own defined optimizer'''
optimizer=raw_model.configure_optimizers(weight_decay=0.1,learning_rate=6e-4,device=device)

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


    ''' One step is a batch'''
    for step in range(max_steps):
        t0=time.time()
        last_step=(step==max_steps-1)
        '''
        once in a while, evaluate validation loss to see how much overfitting
        '''
        if step%100==0:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum=0.0
                val_loss_steps=20
                for _ in range(val_loss_steps):
                    x,y=val_loader.next_batch()
                    x=x.to(device)
                    y=y.to(device)
                    with torch.autocast(device_type=device,dtype=torch.float16): # change dtype to speed up
                        logits,loss=model(x,y)
                    loss=loss/val_loss_steps
                    val_loss_accum+=loss.detach() # loss_accum is only used to analyse average loss
            if ddp: # if ddp, every rank will have loss_accum: get average accumulate loss 
                dist.all_reduce(val_loss_accum,op=dist.ReduceOp.AVG)
            if master_process:
                print(f"Validation loss: {val_loss_accum.item():.4f}")
                with open(log_file,'a') as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step>0 and (step%5000==0 or last_step):
                    checkpoint_path=os.path.join(log_dir,f"model_{step:05d}.pt")
                    checkpoint={
                        'model':raw_model.state_dict(),
                        'config':raw_model.config(),
                        'step':step,
                        'val_loss':val_loss_accum.item()
                    }
                    torch.save(checkpoint,checkpoint_path)

        '''
        once in a while general samples
        '''
        
        if (not python_compile) and (last_step or step%250== 0):
            model.eval()
            num_return_seq=4
            max_length=32
            tokens=enc.encode("Hello, I love LLM and Alignment.")
            tokens=torch.tensor(tokens,dtype=torch.long)
            tokens=tokens.unsqueeze(0).repeat(num_return_seq,1)
            xgen=tokens.to(device)
            sample_rng=torch.Generator(device=device)
            # set different random seed for diff process
            sample_rng.manual_seed(42+ddp_rank)
            while xgen.size(1)<max_length:
                with torch.no_grad():
                    logits,loss=model(xgen)
                    logits=logits[:,-1,:]# (B,vocab_size)
                    prob=F.softmax(logits,dim=-1)
                    topk_probs,topk_indices=torch.topk(prob,50,dim=-1)
                    ix=torch.multinomial(topk_probs,1,sample_rng)
                    # get specific token with its index 
                    xcol=torch.gather(topk_indices,-1,ix)
                    xgen=torch.cat((xgen,xcol),dim=1)
            for i in range(num_return_seq):
                tokens=xgen[i:max_length].tolist()
                decoded=enc.decode(tokens)
                print(f"rank {ddp_rank} sample {i}: {decoded}")


        '''training loop'''
        model.train()
        loss_accum=0.0
        optimizer.zero_grad()
        for iter in range(grad_accu_steps):  # for a mini batch
            x,y=Train_loader.next_batch()
            x=x.to(device)
            y=y.to(device)
            with torch.autocast(device_type=device,dtype=torch.float16): # change dtype to speed up
                logits,loss=model(x,y)
            loss=loss/grad_accu_steps
            loss_accum+=loss.detach() # loss_accum is only used to analyse average loss
            if ddp: # synchronize gradient at last step
                model.require_backward_grad_sync(iter==grad_accu_steps-1)
            loss.backward()
        
        if ddp: # if ddp, every rank will have loss_accum: get average accumulate loss 
            dist.all_reduce(loss_accum,op=dist.ReduceOp.AVG)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        # dynamically change learning rate
        lr=get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr']=lr
        # update parameters based on gradients 
        optimizer.step()
        torch.cuda.synchronize()# wait for gpu to finish work
        t1=time.time()
        dt=(t1-t0)*1000
        token_per_sec=(Train_loader.B*Train_loader.T)*grad_accu_steps*ddp_world_size/(t1-t0)
        # .item convert tensor to a single float
        if master_process:
            print(f"step {step} | loss: {loss.item():.6f} | norm={norm:.4f} | dt is {dt:.4f} | tokens per sec: {token_per_sec:.2f}")

 
training()
if ddp:
    destroy_process_group()

sys.exit(0)