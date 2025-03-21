import tiktoken
import torch
import numpy as np
import os

def load_tokens(filename):
    npt=np.load(filename)
    ptt=torch.tensor(npt,dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self,B,T,process_rank,num_processes,split):
        self.B=B
        self.T=T
        self.process_rank=process_rank
        self.num_processes=num_processes
        assert split in {'train','val'}
        
        data_root="edu_fineweb10B"
        shards=os.listdir(data_root)


        with open('input.txt','r') as f:
            text=f.read()
        enc=tiktoken.get_encoding('gpt2')
        tokens=enc.encode(text)

        self.tokens=torch.tensor(tokens)
        print(f"loadded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batchs")
        # self.current_position=0
        # state
        self.current_position=self.B*self.T*self.process_rank

    def next_batch(self):
        B,T=self.B,self.T
        buf=self.tokens[self.current_position: self.current_position+B*T+1]

        # change shape and to device
        x=buf[:-1].view(B,T)
        y=buf[1:].view(B,T)
        # move forward
        self.current_position+=B*T*self.num_processes

        # if won to the end, reset
        if self.current_position+(B*T*self.num_processes+1)>len(self.tokens):
            self.current_position=0
        return x,y

