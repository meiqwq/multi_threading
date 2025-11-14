from dataset.slimpj_dataset import get_slimpajama_6b
import numpy as np
import torch
import tqdm
import joblib
from sklearn.random_projection import GaussianRandomProjection
from torch.nn.utils import clip_grad_norm_
from sklearn.cluster import KMeans
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from torch.utils.data import Dataset,IterableDataset,DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



class Pre_Data(Dataset):
    def __init__(self,source,seq_len=1024):
        super().__init__()
        self.source=source
        self.seq_len=seq_len
        self.num_samples=(len(self.source))//seq_len
        print(self.num_samples)
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        input_ids=torch.from_numpy(self.source[idx*self.seq_len:(idx+1)*self.seq_len].astype("int32")).long()
        labels=torch.from_numpy(self.source[idx*self.seq_len:(idx+1)*self.seq_len].astype("int32")).long()
        return {
            "input_ids":input_ids,
            "labels":labels
        }


def get_data(data_path="../data/slimpajama"):
    data_src1=get_slimpajama_6b(data_path, "wikipedia", tokenizer_name="EleutherAI/pythia-160m")
    data_src2=get_slimpajama_6b(data_path, "stackexchange", tokenizer_name="EleutherAI/pythia-160m")
    data_set1=Pre_Data(data_src1['train'])
    data_loader1=DataLoader(data_set1,batch_size=8,shuffle=True,num_workers=4)

    data_set2=Pre_Data(data_src2['train'])
    data_loader2=DataLoader(data_set2,batch_size=8,shuffle=True,num_workers=4)
    return data_loader1, data_loader2
