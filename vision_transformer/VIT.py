from einops import rearrange
import numpy as np

from tqdm import tqdm
import cv2
import torch
import torch.nn  as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from torchvision.datasets.cifar import CIFAR100,CIFAR10
from torchvision.datasets.imagenet import ImageNet
from torchsummary import summary as summary
import os
# 기록을 저장할 폴더 생성 및 지정
logs_base_dir = 'logs'
os.makedirs(logs_base_dir, exist_ok = True)

exp = f"{logs_base_dir}/ex1"
writer = SummaryWriter(exp)
# 
np.random.seed(0)
torch.manual_seed(0)

class Embedding(nn.Module) :
  def __init__(self, input_size = 32, input_channel = 3, hidden_size = 8*8*3, patch_size = 4) :
    super().__init__()
    self.patch_size = patch_size
    self.hidden_size = hidden_size
    self.projection = nn.Linear((patch_size**2)*input_channel, hidden_size, bias = False)
    self.cls_token = nn.Parameter(torch.zeros(hidden_size), requires_grad= True)
    num_patches = int((input_size / patch_size) ** 2 + 1)
    self.positional = nn.Parameter(torch.zeros((num_patches, hidden_size), requires_grad= True))

  def forward(self, x) :
    x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = self.patch_size, p2 = self.patch_size)
    x = self.projection(x)
    batch_size = x.shape[0]
    x = torch.cat((self.cls_token.expand(batch_size, 1, self.hidden_size), x), axis = 1)
    x = x + self.positional
    return x

# temp = torch.arange(32*3*32*32, dtype = torch.float32).reshape(32, 3, 32, 32)
# emb = Embedding()
# print(emb(temp).shape) # torch.Size([32, 65, 192])



class MSA(nn.Module) :
  def __init__(self, hidden_dim = 8*8*3, num_heads = 6) :
    super().__init__()
    self.num_heads = num_heads
    self.D_h = (hidden_dim / num_heads) ** (1/2) 
    self.queries = nn.Linear(hidden_dim, hidden_dim)
    self.keys = nn.Linear(hidden_dim, hidden_dim)
    self.values = nn.Linear(hidden_dim, hidden_dim)
    self.softmax = nn.Softmax(dim = 1)

  def forward(self, x) :
    q = rearrange(self.queries(x), 'b n (h d) -> b h n d', h = self.num_heads)
    k = rearrange(self.keys(x), 'b n (h d) -> b h n d', h = self.num_heads)
    v = rearrange(self.values(x), 'b n (h d) -> b h n d', h = self.num_heads)
    A = torch.einsum('bhqd, bhkd -> bhqk', q, k)
    A = self.softmax(A / self.D_h)
    Ax = torch.einsum('bhan, bhnd -> bhad' ,A, v) 
    # b : batch size, h : num_heads, n : num_patches, a : num_patches, d : D_h
    return rearrange(Ax, 'b h n d -> b n (h d)')
# temp = torch.zeros(32, 65, 192)
# print(MSA()(temp).shape) # torch.Size([32, 65, 192])


class MLP(nn.Module):
  def __init__(self, input_dim = 8*8*3, hidden_dim = 8*8*3*4, output_dim = 8*8*3):
    super().__init__()

    self.feedforward = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
        nn.GELU())

  def forward(self, x) :
    return self.feedforward(x)
# temp = torch.zeros(32, 65, 192)
# print(MLP()(temp).shape) # torch.Size([32, 65, 192])

class TransformerEncoderBlock(nn.Module) :
  def __init__(self, hidden_dim = 8*8*3, num_heads = 6, mlp_size = 8*8*3*4):
    super().__init__()

    self.LN = nn.LayerNorm(normalized_shape= hidden_dim)
    self.MSA = MSA(hidden_dim = hidden_dim, num_heads = num_heads)
    self.MLP = MLP(input_dim = hidden_dim, hidden_dim = mlp_size, output_dim = hidden_dim)

  def forward(self, x) :
    x_prev = x
    x = self.LN(x)
    x = self.MSA(x)
    x = x + x_prev
    x_prev = x
    x = self.LN(x)
    x = self.MLP(x)
    return x + x_prev
# temp = torch.zeros(32, 65, 192)
# print(TransformerEncoderBlock()(temp).shape) # torch.Size([32, 65, 192])

class TransformerEncoder(nn.Module):
  def __init__(self, hidden_dim = 8*8*3, num_layers = 8) :
    super().__init__()

    self.num_layers = num_layers
    self.hidden_dim = hidden_dim

    layers = []

    for i in range(0, num_layers) :
      layers.append(TransformerEncoderBlock())

    self.blocks = nn.ModuleList(layers)

    self.LN = nn.LayerNorm(normalized_shape= hidden_dim)

  def forward(self, x):
    for i in range(0, self.num_layers) :
      x = self.blocks[i](x)

    return self.LN(x[:, 0, :])

class MLPHead(nn.Module) :
  def __init__(self, input_dim = 8*8*3, hidden_dim = 8*8*3*4, 
               num_classes = 100,dropout=0.5) :
    super().__init__()
    

    self.feedforward = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(), # GELU는  dropout,zoneout의 특징을 가진 activation이다
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
        nn.Dropout(dropout),)

  def forward(self, x) :
    return self.feedforward(x)

# temp = torch.zeros(32, 192)
# print(MLPHead()(temp).shape)

class ViT(nn.Module) :
  def __init__(self, input_size = 32, patch_size = 4, hidden_size = 8*8*3, num_layers = 8) :
    super().__init__()

    self.vit = nn.Sequential(
        Embedding(input_size = input_size, input_channel = 3, hidden_size = hidden_size, 
                  patch_size = patch_size),
        TransformerEncoder(hidden_dim = hidden_size, num_layers = num_layers),
        MLPHead(input_dim = hidden_size, hidden_dim = hidden_size*4, num_classes= 100)
    )

  def forward(self, x):
    return self.vit(x)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = ViT()
# print(summary(model.to(device),input_size=(3,32,32),batch_size=256))
import time
start_time = time.time()
writer = SummaryWriter('./logs/')
def main():
    # Loading data
    transfrom = ToTensor()
    
    train_set = CIFAR100(root='./../datasets',train=True,download=True,transform=transfrom)
    test_set = CIFAR100(root='./../datasets',train=False,download=True,transform=transfrom)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    # Defining model and training options
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ViT(input_size = 32, patch_size = 4, hidden_size = 8*8*3, num_layers = 8).to(device)
    N_EPOCHS = 1000
    LR = 1e-3

    
    # Training loop
    optimizer = Adam(model.parameters(),lr=LR)
    criterion = CrossEntropyLoss()
    for epoch in tqdm(range(N_EPOCHS),desc="Training"):
        train_loss =0.0
        for batch in tqdm(train_loader,desc=f"Epoch {epoch +1} in training",leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat,y)
            
            train_loss += loss.detach().cpu().item() / len(train_loader)
            writer.add_scalar("Loss/Epoch",loss,epoch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    
            
        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss : {train_loss:.3f}")
    writer.flush()
    
    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        
        for batch in tqdm(test_loader,desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat,y)
            test_loss += loss.detach().cpu().item() / len(test_loader)
            
            correct += torch.sum(torch.argmax(y_hat,dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss : {test_loss:.3f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")
        print('걸린 시간',time.time()-start_time)
        # img_path = 'D:\GAN\image_data/dog1.jpg'
        path = 'D:\GAN\image_data/' # 폴더 경로
        os.chdir(path) # 해당 폴더로 이동
        files = os.listdir(path) 

        jpg_img = []
        for file in files:

            if '.jpg' in file: 
                f = cv2.imread(file)
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB) # opencv는 BGR순서로 read한다.
                
                f = cv2.resize(f,dsize=(32,32),interpolation=cv2.INTER_AREA)
                f = torch.from_numpy(f).float()
                f = f.permute(2, 0, 1) # (H, W, C) -> (C, H, W) -> (1, C, H, W)
                jpg_img.append(f)
                
        out1 = model(jpg_img[0].unsqueeze(0).to(device))
        out2 = model(jpg_img[1].unsqueeze(0).to(device))
        out3 = model(jpg_img[2].unsqueeze(0).to(device))
        probabilities1 = torch.nn.functional.softmax(out1[0], dim=0)
        probabilities2 = torch.nn.functional.softmax(out2[0], dim=0)
        probabilities3 = torch.nn.functional.softmax(out3[0], dim=0)
        top5_prob1, top5_catid1 = torch.topk(probabilities1, 5)
        print('Dog')
        for i in range(top5_prob1.size(0)):
            print(categories[top5_catid1[i]], top5_prob1[i].item())
        
        top5_prob2, top5_catid2 = torch.topk(probabilities2, 5)
        print('Human')   
        for i in range(top5_prob2.size(0)):
            print(categories[top5_catid2[i]], top5_prob2[i].item())
              
        top5_prob3, top5_catid3 = torch.topk(probabilities3, 5)
        print('Rabbit')
        for i in range(top5_prob3.size(0)):
            print(categories[top5_catid3[i]], top5_prob3[i].item())
        torch.save(model, 'model3.pt')
categories = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]
        

    

                
if __name__ == '__main__':
    import os
    main()

    
    
    
    
    

