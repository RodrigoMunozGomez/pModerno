from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
import argparse
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


from models.CSModelPlus import CSModelPlus
from dataloader.dataloader import DatasetLoader

parser = argparse.ArgumentParser(description="Configuración del entrenamiento")

parser.add_argument("--device", type=str, default="cuda:0", help="Dispositivo para entrenamiento (cpu o cuda)")
parser.add_argument("--batch",type=int,default=1)
parser.add_argument("--nEpochs", type=int, default=10, help="Número de épocas")
parser.add_argument("--nKernel", type=int, default=64, help="Tamaño de algo relacionado con la resolución")
parser.add_argument("--lr", type=float, default=0.0001, help="Tasa de aprendizaje")
parser.add_argument("--stepSize", type=int, default=1000, help="Tamaño del paso")
parser.add_argument("--gamma", type=float, default=0.8, help="Factor gamma")
parser.add_argument("--wts",type=str, default=None, help="Pesos modelo")
parser.add_argument("--modelName",type=str,default="CSDecoder")
parser.add_argument("--cRatio",type=float,default=0.1)
args = parser.parse_args()

wts = args.wts
device = args.device
device = torch.device(device if torch.cuda.is_available() else "cpu")
batch = args.batch
nEpochs = args.nEpochs
nKernel = args.nKernel
lr = args.lr
stepSize = args.stepSize
gamma = args.gamma
modelName = args.modelName
cRatio = args.cRatio

transform = transforms.Compose([ # Ajusta al tamaño deseado
    transforms.ToTensor()
])

model = CSModelPlus(output_size=(nKernel*nKernel),CR=cRatio,device=device ).to(device)

modelName = f"{modelName}"

folder_name = f"./train_results"
try:
 os.makedirs(folder_name)
except FileExistsError:
 pass

folder_name = f"./train_results/{modelName}"
try:
 os.makedirs(folder_name)
except FileExistsError:
 pass

folder_name = folder_name + f"/{nEpochs}_{lr}_{stepSize}_{gamma}/"
try:
 os.makedirs(folder_name)
except FileExistsError:
 pass


image_paths_train       = f"./dataset/parches64"
dataset_train           = DatasetLoader(image_paths_train)
data_loader_train       = DataLoader(dataset_train, batch_size=batch, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=stepSize, gamma=gamma)

writer = SummaryWriter(folder_name + "tensorboardLOG")

best_model_wts = copy.deepcopy(model.state_dict())
best_train_error = float('inf')

total_train_loss  = []
total_val_loss    = []

torch.save(model.Phi.data, f'{folder_name}phi_epoch.pt')

pbar = tqdm(range(nEpochs), desc="Training init")
for epoch in pbar:
  loss_per_epochs   = []
  # Training
  for data in data_loader_train:

    input = data.view(batch,1,-1).to(device)
    output = model(input)
    output_target = input
    
    loss_per_batch = criterion(output, output_target)
    loss_per_epochs.append((loss_per_batch.item()))
    optimizer.zero_grad()
    loss_per_batch.backward()
    optimizer.step()
    
  mean_loss_per_epochs_train = sum(loss_per_epochs)/len(loss_per_epochs)
  total_train_loss.append(mean_loss_per_epochs_train)
  
  if mean_loss_per_epochs_train < best_train_error:
    best_train_error = mean_loss_per_epochs_train
    best_model_wts = copy.deepcopy(model.state_dict())
    torch.save(best_model_wts, folder_name + 'best_model.pth')
    best_epoch = epoch
    

      
  scheduler.step()
  
  pbar.set_description(f"{modelName} | {device} | Train Loss : {mean_loss_per_epochs_train:.4} | Best train Loss : {best_train_error:.4} At Epoch : {best_epoch}")
  writer.add_scalar('Train Loss', mean_loss_per_epochs_train, epoch)


print(f"\n Model Name : {modelName}")
print(f"\n|Device : {device}      | Number of Epochs : {nEpochs}  \n|Residual Kernel : {nKernel} | Learning Rate {lr} \n|StepSize lr {stepSize}     | Gamma {gamma}\n")

writer = SummaryWriter(folder_name + "tensorboardLOG")
best_model_wts = copy.deepcopy(model.state_dict())

print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n ")
print(f"    Training Complete {nEpochs}/{nEpochs} num_epochs\n")
print(f"        train loss      :   {best_train_error:.4}")
print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n ")

## Log ##

config = {
    "Model" : modelName,
    "Device": device,
    "Batch Size": batch,
    "Number of Epochs": nEpochs,
    "Residual Kernel": nKernel,
    "Learning Rate": lr,
    "Step Size": stepSize,
    "Model Pre WTS": wts,
    "Gamma": gamma,
    "Training Loss": best_train_error,
}
log_filepath = os.path.join(folder_name, "config_log.txt")
with open(log_filepath, 'w') as log_file:
    for key, value in config.items():
        log_file.write(f"{key}: {value}\n")
        
        
# Iteracion Figure

plt.figure(figsize=(10, 5))
plt.plot(total_train_loss, label='Train Loss')
plt.title('Training and Validation Loss per Epoch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(folder_name + 'training_validation_loss.png', dpi=300)  # Guarda la gráfica como un archivo PNG con alta resolución

print("        Figure Saved")
print("\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n ")

tloss = {'train_data': total_train_loss}
scipy.io.savemat(folder_name + 'train_data.mat', tloss)
