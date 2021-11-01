import torch
import wandb
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
early_stop = EarlyStopping()
wandb.init(project='deformed-darknet',entity='tensorthug',name='audio-classification-cnn14')



print("***** Loading the Model in {} *****".format(DEVICE))

Model = CNN6().to(DEVICE)

print("Model Shipped to {}".format(DEVICE))


optim = torch.optim.Adam(Model.parameters())

wandb.watch(Model)




def train_loop(epoch,dataloader,model,loss_fn,optim,device=DEVICE):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    #start_time = time.time()
    pbar = tqdm(enumerate(dataloader),total=len(dataloader))
    for i,(img,label) in pbar:
        optim.zero_grad()

        img = img.to(DEVICE).float()
        label = label.to(DEVICE).float()
        
        #LOAD_TIME = time.time() - start_time

        with autocast():
            yhat = model(img)
            #Loss Calculation
            train_loss = loss_fn(input = yhat.flatten(), target = label)
        
        out = (yhat.flatten().sigmoid() > 0.5).float()
        correct = (label == out).float().sum()

        scaler.scale(train_loss).backward()
        scaler.step(optim)
        scaler.update()

        
        epoch_loss += train_loss.item()
        epoch_acc += correct.item() / out.shape[0]

    train_epoch_loss = epoch_loss / len(dataloader)
    train_epoch_acc = epoch_acc / len(dataloader)

    wandb.log({"Training_Loss":train_epoch_loss})
    wandb.log({"Training_Acc":train_epoch_acc})
        
    #print(f"Epoch:{epoch}/{TOTAL_EPOCHS} Epoch Loss:{epoch_loss / len(dataloader):.4f} Epoch Acc:{epoch_acc / len(dataloader):.4f}")
    
    return train_epoch_loss,train_epoch_acc

def val_loop(epoch,dataloader,model,loss_fn,device = DEVICE):
    model.eval()
    val_epoch_loss = 0
    val_epoch_acc = 0
    pbar = tqdm(enumerate(dataloader),total=len(dataloader))

    with torch.no_grad():
        for i,(img,label) in pbar:
            img = img.to(device).float()
            label = label.to(device).float()

            yhat = model(img)
            val_loss = loss_fn(input=yhat.flatten(),target=label)

            out = (yhat.flatten().sigmoid()>0.5).float()
            correct = (label == out).float().sum()

            val_epoch_loss += val_loss.item()
            val_epoch_acc += correct.item() / out.shape[0]

        val_lossd = val_epoch_loss / len(dataloader)
        val_accd = val_epoch_acc / len(dataloader)
        
        wandb.log({"Val_Loss":val_lossd,"Epoch":epoch})
        wandb.log({"Val_Acc":val_accd/len(dataloader),"Epoch":epoch})

        return val_lossd,val_accd






        
    


if __name__ == "__main__":

    train_data = AudioDataset(0)
    val_data = AudioDataset(0,mode='validation')

    trainloader = DataLoader(train_data,batch_size=8,shuffle=True)
    valloader = DataLoader(val_data,batch_size=8,shuffle=True)

    print(next(iter(trainloader)))