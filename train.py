from constants import EPOCHS
import torch
import torch.nn as nn
from utils.build_model import CNN6
from preprocess import AudioDataset
import numpy as np
import random
from utils.callbacks import EarlyStopping
from torch.cuda.amp import GradScaler,autocast
from torch.utils.data import DataLoader
from tqdm import tqdm



torch.manual_seed(2021)
np.random.seed(2021)
random.seed(2021)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TOTAL_EPOCHS = 100
scaler = GradScaler()
early_stop = EarlyStopping(path='Models')




print("***** Loading the Model in {} *****".format(DEVICE))

Model = CNN6().to(DEVICE)

print("Model Shipped to {}".format(DEVICE))

train_loss = nn.CrossEntropyLoss()
val_loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(Model.parameters())





def train_loop(epoch,dataloader,model,loss_fn,optim,device=DEVICE):
    model.train()
    epoch_loss = 0
    #epoch_acc = 0
    #start_time = time.time()
    pbar = tqdm(enumerate(dataloader),total=len(dataloader))
    for i,(img,label) in pbar:
        optim.zero_grad()

        img = img.to(DEVICE).float()
        label = label.to(DEVICE).long()

        #print(label)
        
        #LOAD_TIME = time.time() - start_time

        with autocast():
            yhat = model(img)
            #Loss Calculation
            train_loss = loss_fn(input = yhat, target = label)
        
        

        scaler.scale(train_loss).backward()
        scaler.step(optim)
        scaler.update()

        
        epoch_loss += train_loss.item()
        

    train_epoch_loss = epoch_loss / len(dataloader)
    

    
        
    #print(f"Epoch:{epoch}/{TOTAL_EPOCHS} Epoch Loss:{epoch_loss / len(dataloader):.4f} Epoch Acc:{epoch_acc / len(dataloader):.4f}")
    
    return train_epoch_loss

def val_loop(epoch,dataloader,model,loss_fn,device = DEVICE):
    model.eval()
    val_epoch_loss = 0
   # val_epoch_acc = 0
    pbar = tqdm(enumerate(dataloader),total=len(dataloader))

    with torch.no_grad():
        for i,(img,label) in pbar:
            img = img.to(device).float()
            label = label.to(device).long()

            yhat = model(img)
            val_loss = loss_fn(input=yhat,target=label)

            

            val_epoch_loss += val_loss.item()
            

        val_lossd = val_epoch_loss / len(dataloader)
        
        
        

        return val_lossd






        
    


if __name__ == "__main__":

    train_data = AudioDataset(0)
    val_data = AudioDataset(0,mode='validation')

    trainloader = DataLoader(train_data,batch_size=8,shuffle=True)
    valloader = DataLoader(val_data,batch_size=8,shuffle=True)

    #print(next(iter(trainloader)))

    for e in range(EPOCHS):
        train_el = train_loop(epoch=e,dataloader=trainloader,model=Model,loss_fn=train_loss,optim=optim)
        val_el = val_loop(epoch=e,dataloader=valloader,model=Model,loss_fn=val_loss)
        early_stop(Model,val_el)

        print(f"{e}/{EPOCHS}, training_loss:{train_el}, val_loss:{val_el}")
