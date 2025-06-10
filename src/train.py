import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import numpy as np
from config.config import config_hp
import torch
import torch.nn as nn
from src.models import OurModel
from src.data import get_datasets_and_loaders_for_lstm
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from src.utils import EarlyStopping,save_checkpoint, plot_l_a
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO


our_train_data_loader, our_test_data_loader,our_train_data_set,our_test_data_set,our_vocab,x_test=get_datasets_and_loaders_for_lstm()

device="cuda" if torch.cuda.is_available() else "cpu"

our_lstm_model=OurModel(config_hp["FEATURE_EMBEDDING_SIZE"],config_hp["LSTM_HIDDEN_SIZE"],config_hp["MAX_VOCAB"],config_hp["LSTM_N_LAYERS"]).to(device)

early_stopping = EarlyStopping(patience=5, min_delta=0)


def our_train(epochs,our_model):
  
    our_loss=nn.CrossEntropyLoss(ignore_index=our_vocab["<PAD>"])
    our_optimizer =torch.optim.Adam(our_model.parameters(),lr=config_hp["LEARNING_RATE"])
    writer = SummaryWriter('runs\lstm')
    
    train_loss_l=[]
    
    for epoch in range(epochs): 
    
    
      train_loss_v=0.0
      
      our_model.train()  
      for imgs_train_batch , cap_train_batch, cap_target_batch in tqdm (our_train_data_loader,f"Epoch = {epoch}") : 
        
        imgs_train_batch=imgs_train_batch.to(device)
        cap_train_batch=cap_train_batch.to(device)
        cap_target_batch=cap_target_batch.to(device)
    
        cap_train_pred= our_model(imgs_train_batch,cap_train_batch)

        train_loss=our_loss(cap_train_pred.view(-1,len(our_vocab)),cap_target_batch.view(-1))

        our_optimizer.zero_grad()
        train_loss.backward()
        our_optimizer.step()
        train_loss_v +=train_loss.item()
    
         
      our_model.eval()
      with torch.inference_mode():
        
          ran=torch.randint(0,40,(1,)).item()

          img=Image.open(x_test[ran]).convert('RGB')
        
          t_img=our_test_data_set[ran][0].to(device).unsqueeze(0)
         
          t_cap_target=our_test_data_set[ran][2].to(device).unsqueeze(0)
          cap_target=t_cap_target.squeeze().cpu().detach().numpy()
          cap_target=our_vocab.lookup_tokens(cap_target)
    
         
          x=our_model.encoder(t_img) # 1*100
          state=None  
          final_output=[] 
           
          for _ in range (config_hp["MAX_LEN"]):
    
              x,state=our_model.decoder.lstm(x,state) #1*256
              x=our_model.decoder.linear(x)  #1*256 * 256*4000 >>>1*4000
              x=torch.argmax(torch.softmax(x,dim=1) ,dim=1).squeeze().cpu().detach().numpy() #1*30*1 >>> 30
              
              x=our_vocab.lookup_token(x)     #[t for t in  our_vocab.lookup_tokens(cap_pred) if t!=1 ]
              
              final_output.append(x)
              # if x==config_hp["SPECIAL_TOKENS"][3]: 
              #     break 
              x=torch.tensor(our_vocab([x])).to(device)#.unsqueeze(0)
             # print(x.shape)
              x=our_model.decoder.embedding(x)
              
          img = img.resize((224, 224))
          fig = plt.figure()
          plt.imshow(img)
          plt.title(f"cap_pred = {final_output} \n cap_target = {cap_target}")

          buf = BytesIO()
          plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
          buf.seek(0)
          img_with_title = Image.open(buf).convert('RGB')
          img_array = np.array(img_with_title)

          img_tensor = torch.from_numpy(img_array).float()
          img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)/ 255.0
          plt.close(fig)
          
          
          plt.figure(figsize=(50,15))
          
          plt.imshow(img)
          plt.title(f"cap_pred = {final_output} \n cap_target = {cap_target}")
          
          plt.show()
          
         # plt.imshow(img)
         # title=" ".join(final_output)
         # plt.axis('off')
         # plt.title(f"predicted Caption: {title} \n target cap: {cap_target}",fontsize=10)
         # plt.show()

      train_loss_l.append(train_loss_v/our_train_data_set.__len__())

      writer.add_scalar("Loss/Train", train_loss_l[-1], epoch)
      writer.add_images("Images/Test", img_tensor, epoch)  
          
      print(f"at epoch = {epoch+1} || train loss = {train_loss_l[-1]:.3f}")
      
      if (epoch + 1) % 2 == 0:
            save_checkpoint(our_model,our_optimizer,epoch + 1, train_loss_l[-1],r'D:\Image_Captioning\best_model_lstm\model_checkpoint.pth')
            print(f"Checkpoint saved at epoch {epoch + 1}.")

      if early_stopping(our_model, train_loss_l[-1],train_loss_l[-1]):
            print(early_stopping.status)
            save_checkpoint(our_model,our_optimizer,epoch + 1, train_loss_l[-1], r'D:\Image_Captioning\best_model_lstm\best_model.pth')
            print("Early stopping triggered!")
            writer.close()
            break 

    if early_stopping.counter < early_stopping.patience:
        save_checkpoint(our_model, our_optimizer, epoch+1, train_loss_l[-1], r'D:\Image_Captioning\best_model_lstm\final_model.pth')
        writer.close()
        print("Training completed without early stopping. Final model saved.")
    
    return train_loss_l


 
if __name__ == "__main__":
    train_loss_l=our_train(epochs=config_hp["EPOCHS"],our_model=our_lstm_model)


