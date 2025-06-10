from models import OurModel
from data import get_datasets_and_loaders_for_lstm
from config.config import config_hp
import torch
import matplotlib.pyplot as plt
from utils import trans_test
from PIL import Image
import os


our_train_data_loader, our_test_data_loader,our_train_data_set,our_test_data_set,our_vocab,x_test=get_datasets_and_loaders_for_lstm()

device="cuda" if torch.cuda.is_available() else "cpu"


our_lstm_model=OurModel(config_hp["FEATURE_EMBEDDING_SIZE"],config_hp["LSTM_HIDDEN_SIZE"],config_hp["MAX_VOCAB"],config_hp["LSTM_N_LAYERS"]).to(device)
checkpoint = torch.load(r"D:\Image_Captioning\best_model_lstm\best_model.pth")
our_lstm_model.load_state_dict(checkpoint['model_state_dict'])



def image_captioning(img_dir, our_model, our_vocab, device):
    
    our_model.eval()
    with torch.inference_mode():
    
      #ran=torch.randint(0,40,(1,)).item()
    
      img=Image.open(img_dir).convert('RGB')
    
 
        
      #t_img=our_test[ran][0].to(device).unsqueeze(0)
        
      t_img=trans_test(img).to(device).unsqueeze(0)
      x=our_model.encoder(t_img) # 1*100
      state=None  
      final_output=[] 
       
      for _ in range (config_hp["MAX_LEN"]):

          x,state=our_model.decoder.lstm(x,state) #1*256
          x=our_model.decoder.linear(x)  #1*256 * 256*4000 >>>1*4000
          x=torch.argmax(torch.softmax(x,dim=1) ,dim=1).squeeze().cpu().detach().numpy() #1*30*1 >>> 30
          
          x=our_vocab.lookup_token(x)     #[t for t in  our_vocab.lookup_tokens(cap_pred) if t!=1 ]
          
          final_output.append(x)
          if x==config_hp["SPECIAL_TOKENS"][3]: 
              break 
          x=torch.tensor(our_vocab([x])).to(device)#.unsqueeze(0)
         # print(x.shape)
          x=our_model.decoder.embedding(x)
          
    plt.imshow(img)
    title=" ".join(final_output)
    plt.axis('off')
    plt.title(f"predicted Caption: {title}",fontsize=10)
    plt.show()   
      
    return final_output
      
    
if __name__=="__main__":
    for img_dir in os.listdir(r"D:\Image_Captioning\test_examples"):
        
        img_dir = os.path.join(r"D:\Image_Captioning\test_examples", img_dir)
    
        caption= image_captioning(img_dir, our_lstm_model, our_vocab, device)
        
        
        
