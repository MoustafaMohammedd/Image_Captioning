import os
import sys  

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader ,Dataset
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator
from src.utils import pad_sentence ,prepare_data,tokenizer,trans_train,trans_test
from config.config import config_hp
from PIL import Image

data=[]
with open('D:\Image_Captioning\flickr-8k-images-with-captions\captions.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        data.append(line.strip())

data=data[1:]

img=[]
caption=[]
for i in data:
  n=i.split(",")
  img.append(n[0])
  caption.append(n[1])


df=pd.DataFrame()
df['image']=img
df['caption']=caption

dir="D:\Image_Captioning\flickr-8k-images-with-captions\images"
df["image"]=df["image"].apply(lambda x:os.path.join(dir,x))


tokens_cap=df["caption"].map(prepare_data)


x_train,x_test,y_train,y_test=train_test_split(np.array(df["image"]),np.array(df["caption"]),test_size=0.001,shuffle=True,random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


our_vocab=build_vocab_from_iterator(df["caption"].map(prepare_data),specials=config_hp["SPECIAL_TOKENS"],max_tokens=config_hp["MAX_VOCAB"])
our_vocab.set_default_index(our_vocab[config_hp["SPECIAL_TOKENS"][1]])


class OurDataSet(Dataset):
  def __init__(self,x,y,vocab,max_len,transform=None):
    self.x=x
    self.y=y
    self.vocab=vocab
    self.max_len=max_len
    self.transform=transform

  def __len__(self):
    return len(self.x)
  def __getitem__(self, index):
    
       img=Image.open(self.x[index]).convert("RGB")
       caption=self.y[index]

       caption=prepare_data(caption)

       caption=[config_hp["SPECIAL_TOKENS"][2]]+caption+[config_hp["SPECIAL_TOKENS"][3]]
       caption_input=caption[:-1]
       caption_output=caption[:]

       caption_input=pad_sentence(caption_input,self.max_len,config_hp["SPECIAL_TOKENS"][0])
       caption_output=pad_sentence(caption_output,self.max_len,config_hp["SPECIAL_TOKENS"][0])

       caption_input=caption_input[:-1]

       caption_input=self.vocab.lookup_indices(caption_input)
       caption_output=self.vocab.lookup_indices(caption_output)

       caption_input=torch.tensor(caption_input).long()
       caption_output=torch.tensor(caption_output).long()


       if self.transform is not None:
         img=self.transform(img)

      #  img=torch.tensor(np.array(img,dtype=float)).float()/255.0
      #  img=img.permute(2,0,1)

       return img, caption_input, caption_output


our_train_data_set=OurDataSet(x=x_train,y=y_train,vocab=our_vocab,max_len=config_hp["MAX_LEN"],transform=trans_train)
our_test_data_set=OurDataSet(x=x_test,y=y_test,vocab=our_vocab,max_len=config_hp["MAX_LEN"],transform=trans_test)

def get_datasets_and_loaders_for_lstm ():
   our_train_data_loader=DataLoader(our_train_data_set,batch_size=config_hp["BATCH_SIZE"],shuffle=True)
   our_test_data_loader=DataLoader(our_test_data_set,batch_size=config_hp["BATCH_SIZE"],shuffle=False)
   return our_train_data_loader,our_test_data_loader,our_train_data_set,our_test_data_set,our_vocab,x_test



if __name__ == "__main__":
  our_train_data_loader,our_test_data_loader,our_train_data_set,our_test_data_set,our_vocab,x_test=get_datasets_and_loaders_for_lstm()
  for i ,j ,k in our_train_data_loader :
    print(i.shape)
    print(j.shape)
    print(k.shape)
    break

