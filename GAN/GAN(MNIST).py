import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import Dataset
import pandas as pd
             
class MnistDataset(Dataset):
    def __init__(self,csv_file):
        self.data_df = pd.read_csv(csv_file,header=None)
        pass
    def __len__(self):
        return len(self.data_df)
    def __getitem__(self, index):
        # 이미지 목표(label)
        label = self.data_df.iloc[index,0]
        target = torch.zeros((10))
        target[label]= 1.0
        # 0~255의 이미지를 0~1로 정규화
        
        image_values = torch.FloatTensor(self.data_df.iloc[index,1:].values)
        # 레이블 ,이미지, 데이터 텐서, 목표 텐서 반환
        return label, image_values, target
    def plot_image(self,index):
        img = self.data_df.iloc[index,1:].values.reshape(28,28)
        plt.title("label = "+str(self.data_df.iloc[index,0]))
        plt.imshow(img, interpolation='none',cmap='Blues')
        plt.show()
        pass
    pass
mnist_dataset = MnistDataset('D:\study_data\_data\\test108\mnist_train.csv')
print(mnist_dataset.plot_image(9))        
# classifier class

class Classifier(nn.Module):
    
    def __init__(self):
        # initialise parent pytorch class
        super().__init__()
        
        # define neural network layers
        self.model = nn.Sequential(
            nn.Conv2d(1,10,kernel_size=3,stride=2),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(200),
            
            nn.Conv2d(10,10,kernel_size=3,stride=2),
            nn.LeakyReLU(0.02),
            nn.BatchNorm2d(200),
            View(250),            
            nn.Linear(250, 10),
            nn.Sigmoid()
            #nn.LeakyReLU(0.02)
        )
        
        # create loss function
        self.loss_function = nn.BCELoss()
        #self.loss_function = nn.MSELoss()

        # create optimiser, using simple stochastic gradient descent
        #self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)
        self.optimiser = torch.optim.Adam(self.parameters())

        # counter and accumulator for progress
        self.counter = 0
        self.progress = []

        pass
    
    
    def forward(self, inputs):
        # simply run model
        return self.model(inputs)
    
    
    def train(self, inputs, targets):
        # calculate the output of the network
        outputs = self.forward(inputs)
        
        # calculate loss
        loss = self.loss_function(outputs, targets)

        # increase counter and accumulate error every 10
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        if (self.counter % 10000 == 0):
            print("counter = ", self.counter)
            pass

        # zero gradients, perform a backward pass, and update the weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass
    
    
    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        pass
    
    pass   

# create neural network

C = Classifier()

# train network on MNIST data set

epochs = 3

for i in range(epochs):
    print('training epoch', i+1, "of", epochs)
    for label, image_data_tensor, target_tensor in mnist_dataset:
        C.train(image_data_tensor, target_tensor)
        pass
    pass
# test trained neural network on training data

score = 0
items = 0
mnist_test_dataset = MnistDataset('D:\study_data\_data\\test108/mnist_test.csv')
for label, image_data_tensor, target_tensor in mnist_test_dataset:
    answer = C.forward(image_data_tensor).detach().numpy()
    if (answer.argmax() == label):
        score += 1
        pass
    items += 1
    
    pass

print(score, items, score/items)



 
