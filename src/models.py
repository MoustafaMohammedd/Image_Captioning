import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import torch 
import torch.nn as nn
from torchvision import models
from config.config import config_hp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet=models.resnet50(pretrained=True).to(device)
for param in resnet.parameters():
  param.requires_grad=False
  
class OurEncoder(nn.Module):
  def __init__(self,resnet,embed_size):
    super(OurEncoder,self).__init__()
    self.resnet=resnet
    self.embed_size=embed_size
    self.l1=nn.Sequential(*list(self.resnet.children())[:-1])
    self.l2=nn.Linear(2048,embed_size)
    self.relu=nn.ReLU()

    self.dropout=nn.Dropout(0.5)

  def forward(self,x):
    x=self.l1(x)
    x=x.view(x.size(0),-1)
    x=self.relu(x)
    x=self.dropout(x)
    x=self.l2(x)
    return x
  
 
class OurDecoder(nn.Module):
  def __init__(self,embed_size,hidden_size,vocab_size,n_layers):
    super(OurDecoder,self).__init__()
    self.embed_size=embed_size
    self.hidden_size=hidden_size
    self.vocab_size=vocab_size
    self.n_layers=n_layers

    self.embedding=nn.Embedding(vocab_size,embed_size,0) # 0 is the index for padding token our_vocab["<PAD>"]
    self.lstm=nn.LSTM(embed_size,hidden_size,batch_first=True,num_layers=self.n_layers)
    self.linear=nn.Linear(hidden_size,vocab_size)
    self.dropout=nn.Dropout(0.5)

  def forward(self,x,features):
    x=self.embedding(x)
    x=torch.cat((features.unsqueeze(1),x),dim=1)
    x,_=self.lstm(x)
    x=self.dropout(x)
    x=self.linear(x)

    return x
  
class OurModel(nn.Module):
  def __init__(self,embed_size,hidden_size,vocab_size,n_layers):
    super(OurModel,self).__init__()
    self.encoder=OurEncoder(resnet,embed_size)
    self.decoder=OurDecoder(embed_size,hidden_size,vocab_size,n_layers)

  def forward(self,x,y):
    features=self.encoder(x)
    x=self.decoder(y,features)
    return x
 
if __name__=="__main__": 
    our_model=OurModel(config_hp["FEATURE_EMBEDDING_SIZE"],config_hp["LSTM_HIDDEN_SIZE"],config_hp["MAX_VOCAB"],config_hp["LSTM_N_LAYERS"]).to(device)
    x = torch.randint(0, config_hp["MAX_VOCAB"], (32, 29)).to(device)
    xx = torch.rand(32,3,224,224).to(device)
    print(our_model(xx,x).shape)
