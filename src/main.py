
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
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tool.datautil import get_data
from multi_thread_kmeans import kmeans_threading
import json
load1,load2=get_data(data_path="../data/slimpajama")
model_path="../model/pythia-160m"
model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-160m",
  revision="step100000",
  cache_dir=model_path,
)
device=torch.device("cuda:5")
model=model.to(device)

tokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-160m",cache_dir=model_path)



def work(batch):
    global model
    with torch.no_grad():
        input_ids=batch["input_ids"].to("cuda:5")
        labels = batch["labels"].to("cuda:5")
        outputs=model(input_ids=input_ids,labels=labels)
        emb=model.gpt_neox.embed_in(input_ids)
        #print("loss:",outputs.loss.item())
        return emb[:,-1,:].cpu().numpy()

ls=[]
tot=100
steps=0
progress_bar=tqdm.tqdm(range(tot))
for batch in load1:
    ls.extend(work(batch))
    steps+=1
    progress_bar.update(1)
    if steps>=tot//2:
        break
for batch in load2:
    ls.extend(work(batch))
    steps+=1
    progress_bar.update(1)
    if steps>=tot:
        break
num_k=300
labels, centers=kmeans_threading(np.vstack(ls), k=num_k, max_iter=200, n_threads=32)
print(labels[0],centers[0])
cnt=np.zeros(num_k)
for l in labels:
    cnt[l]+=1
cnt.sort()
cnt=cnt[::-1]
cnt/=sum(cnt)
# plot cnt
plt.figure(figsize=(10,6))
plt.bar(range(num_k), cnt)
plt.xlabel("Cluster Index")
plt.ylabel("Number of Points")
plt.title("Cluster Distribution")
plt.savefig("cluster_distribution.png")

example={}
for _ in range(num_k):
    example[_]=[]
for batch in load1:
    emb=work(batch)
    for i in range(emb.shape[0]):
        l=np.argmin([np.linalg.norm(emb[i]-c) for c in centers])
        example[l].append(tokenizer.decode(batch["input_ids"][i].cpu().numpy()))
    break
for batch in load2:
    emb=work(batch)
    for i in range(emb.shape[0]):
        l=np.argmin([np.linalg.norm(emb[i]-c) for c in centers])
        example[l].append(tokenizer.decode(batch["input_ids"][i].cpu().numpy()))
    break

with open("cluster_examples.json","w") as f:
    json.dump(example,f,indent=4)