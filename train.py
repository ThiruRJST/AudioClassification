from unicodedata import name
from constants import EPOCHS
import torch
import torch.nn as nn
from utils.build_model import CNN14_mod
from preprocess import AudioDataset
import numpy as np
import random
from utils.callbacks import EarlyStopping
from torch.cuda.amp import GradScaler,autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


def seed_everything(seed=2021):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True




def train_loop(fold,epoch,dataloader,model,loss_fn,optim,device='cuda:0'):
    model.train()
    epoch_loss = 0
    #epoch_acc = 0
    #start_time = time.time()
    pbar = tqdm(enumerate(dataloader),total=len(dataloader))
    for i,(img,label) in pbar:
        optim.zero_grad()

        img = img.to(device).float()
        label = label.to(device).long()

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

def val_loop(fold,epoch,dataloader,model,loss_fn,device = 'cuda:0'):
    model.eval()
    val_epoch_loss = 0
    val_epoch_acc = 0
    pbar = tqdm(enumerate(dataloader),total=len(dataloader))

    with torch.no_grad():
        for i,(img,label) in pbar:
            img = img.to(device).float()
            label = label.to(device).long()

            yhat = model(img)
            val_loss = loss_fn(input=yhat,target=label)
            
            pred = torch.argmax(torch.softmax(yhat,dim=1))
            correct = (pred == label).float().sum()

            

            val_epoch_loss += val_loss.item()
            val_epoch_acc += correct.item() / img.shape[0]

            

        val_lossd = val_epoch_loss / len(dataloader)
        val_acc = val_epoch_acc / len(dataloader)
        
        
        

        return val_lossd,val_acc




if __name__ == "__main__":

    

    #print(next(iter(trainloader)))
    for f in range (5):

        seed_everything()

        DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
        scaler = GradScaler()
        early_stop = EarlyStopping(path='Models',name=f'Fold{f}')




        #print("***** Loading the Model in {} *****".format(DEVICE))

        Model = CNN14_mod().to(DEVICE)

        print("Model Shipped to {}".format(DEVICE))

        train_loss = nn.CrossEntropyLoss()
        val_loss = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(Model.parameters())


        train_data = AudioDataset(f)
        val_data = AudioDataset(f,mode='validation')

        trainloader = DataLoader(train_data,batch_size=16,shuffle=True)
        valloader = DataLoader(val_data,batch_size=16,shuffle=True)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,'min',patience=5,verbose=True)

        for e in range(EPOCHS):
            train_el = train_loop(fold=f,epoch=e,dataloader=trainloader,model=Model,loss_fn=train_loss,optim=optim,device=DEVICE)
            val_el,val_acc = val_loop(fold=f,epoch=e,dataloader=valloader,model=Model,loss_fn=val_loss,device=DEVICE)
            print(f"{e}/{EPOCHS}, training_loss:{train_el}, val_loss:{val_el}, val_acc:{val_acc}")

            scheduler.step(val_el)
            early_stop(Model,val_el)
            if early_stop.early_stop:
                break
            
