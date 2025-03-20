

'''
- Does not handle the regular expression splitting pattern.
- Does not handle any special tokens.

'''
import json

def save_dict_json(dictionary, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, indent=4, ensure_ascii=False)

def load_dict_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

class Tokenizer:
    def __init__(self):
        self.new_idx=0
        self.vocab={idx: bytes([idx]) for idx in range(256)}
        self.merge={}
        pass

    def get_stats(self,ids):
        # construct pairs and count frequrncy
        counts={}
        for pair in zip(ids,ids[1:]):
            counts[pair]=counts.get(pair,0)+1

        return counts

    def merge_f(self,ids,pair,idx):
        newids=[]
        i=0
        while i<len(ids):
            if i<len(ids)-1 and ids[i]==pair[0] and ids[i+1]==pair[1]:
                newids.append(idx)
                i+=2
            else:
                newids.append(ids[i])
                i+=1
        return newids

    def train(self, text, vocab_size):
        """
        Train the tokenizer using BPE algorithm.
        Params:
            text (str): string-type data used to run BPE.
            vocab_size (int): the size of final vocabulary.

        Return:
            None
        """
        tokens=text.encode("utf-8")
        tokens=list(map(int,tokens))
        new_idx=256
        while new_idx<=vocab_size:
            stats=self.get_stats(tokens) 
            top_pair=max(stats,key=stats.get)
            self.merge[top_pair]= new_idx
            # print(f"merging {top_pair} into a new token {new_idx}")
            tokens=self.merge_f(tokens,top_pair,new_idx)
            new_idx+=1

        for (p0,p1), idx in self.merge.items():
            self.vocab[idx]=self.vocab[p0]+self.vocab[p1]
        self.new_idx=new_idx

        # save_dict_json(self.vocab,"my_tokenizer.json")

    def encode(self, text):
        """
        Encode the input string into a token list.
        Params:
            text (str): data to be tokenized.

        Return:
            ids (list): list of integer-type tokens.
        """
        tokens=text.encode("utf-8")
        # in a specifuc order
        while len(tokens)>2:
            stats=self.get_stats(tokens)
            # no need to consider frequency
            # if find in self.merge: merge; or set to inf
            pair=min(stats,key=lambda p:self.merge.get(p,float("inf")))
            # find min pair number, it has lowest dependcy
            if pair not in self.merge:
                break
            idx=self.merge[pair]
            tokens=self.merge_f(tokens,pair,idx)

        return tokens


    def decode(self, ids):
        """
        Decode a token list into a string.
        Params:
            ids (list): list of integer-type tokens.

        Return:
            text (str): string-type data.
        """
        tokens=b"".join(self.vocab[idx] for idx in ids)
        text=tokens.decode("utf-8",errors='replace')
        return text


'''
use tokenizer
'''
import tiktoken
def use_tokenizer(tokenizer,str):
    enc=tiktoken.get_encoding(tokenizer)
    print(enc.encode(str))
'''
train my tokenizer and evaluate
'''



def train_mytokenizer():
    with open("manual.txt", "r", encoding="utf-8") as f:
        str = f.read()
    print("training")
    myTokenizer=Tokenizer()
    myTokenizer.train(str,1024)
    return myTokenizer
    '''
    print("encoding and decoding")
    encoded=myTokenizer.encode(str)
    decoded=myTokenizer.decode(encoded)
   

    with open("encode_decode_data.txt", "w", encoding="utf-8") as f:
        f.write(decoded)
    '''

def train_test_data(filename,save_path,tokenizer):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
    encoded_result=tokenizer.encode(text)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(str(encoded_result) + "\n\n")

tokenizer=train_mytokenizer()
train_test_data("encode_task_ch.txt","my_ch.txt",tokenizer)
train_test_data("encode_task_en.txt","my_en.txt",tokenizer)