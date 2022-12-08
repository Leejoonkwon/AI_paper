import torch
import torch.nn as nn
from torch.optim import Adam
import pandas 
import matplotlib.pyplot as plt
import random
import time
import numpy as np
start_time = time.time()
# funtion to generate real data
def generate_real():
    real_data = torch.FloatTensor([
        random.uniform(0.8 , 1.0),
        random.uniform(0.0 , 0.2),
        random.uniform(0.8 , 1.0),
        random.uniform(0.0 , 0.2),
        ])
    return real_data
print(generate_real()) # tensor([0.8720, 0.1870, 0.9756, 0.1758])
                                                            
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__() # pytorch 부모 클래스 초기화
        self.model = nn.Sequential( # 신경망 레이어 정의
            nn.Linear(4,3),
            nn.Sigmoid(),
            nn.Linear(3,1),
            nn.Sigmoid())
        # 손실함수 정의
        self.loss_function = nn.MSELoss()    
                                                  
        # Adam 옵티마이저 설정
        self.optimiser = Adam(self.parameters(),lr=0.01)
                                                                                                                  
        # 진행 측정을 위한 변수 초기화
        self.counter = 0
        self.progress = []
        pass
    def forward(self,inputs):
        return self.model(inputs)
                                
    def train(self,inputs,targets):
        # 신경망 출력 계산
        outputs = self.forward(inputs)
        # 손실 계산
        loss = self.loss_function(outputs,targets)
        #카운터를 증가시키고 10회마다 오차 저장
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        if (self.counter %10000 == 0):
            print("counter ",self.counter)
            pass
        # 기울기를 초기화하고 역전파 후 가중치 갱신
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        pass        
    def plot_progress(self):
        df = pandas.DataFrame(self.progress,columns=['loss'])
        df.plot(ylim=(0, 1.0),figsize=(16,8),alpha=0.1,marker = '.',
                grid=True,yticks=(0.25,0.5))
        pass
def generate_random(size):
    random_data = torch.rand(size)
    return random_data
D = Discriminator()
for i in range(10000):
    # 실제 데이터
    D.train(generate_real(),torch.FloatTensor([1.0]))
    # 생선된 데이터
    D.train(generate_random(4),torch.FloatTensor([0.0]))
    pass
# D.plot_progress()
# plt.show()    
# manually run discriminator to check it can tell real data from fake
# print(D.forward(generate_real()).item()) # 0.8601407408714294 # 진짜 데이터는 1에 가까움
# print(D.forward(generate_random(4)).item()) # 7.989050573087297e-06 # 생성 데이터는 0에 수렴함

class Generator(nn.Module):
    def __init__(self):
        super().__init__() # pytorch 부모 클래스 초기화
        # 신경망 레이어 정의
        self.model = nn.Sequential( 
            nn.Linear(1,3),
            nn.Sigmoid(),
            nn.Linear(3,4),
            nn.Sigmoid())

        # Adam 옵티마이저 설정
        self.optimiser = Adam(self.parameters(),lr=0.01)
        
        # 진행 측정을 위한 변수 초기화
        self.counter = 0
        self.progress = []
        pass
    
    def forward(self,inputs):
        return self.model(inputs) 
               
    def train(self, D, inputs,targets):
        # 신경망 출력 계산
        g_output = self.forward(inputs)
        
        # 판별기로 전달
        d_output = D.forward(g_output)
        
        # 오차 계산
        loss = D.loss_function(d_output,targets)
        # 카운터를 증가시키고 10회마다 오차 저장
        self.counter += 1
        if (self.counter% 10 == 0):
            self.progress.append(loss.item())
            pass
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        pass
    def plot_progress(self):
        df = pandas.DataFrame(self.progress,columns=['loss'])
        df.plot(ylim=(0, 1.0),figsize=(16,8),alpha=0.1,marker = '.',
                grid=True,yticks=(0.25,0.5))
G = Generator()
print(G.forward(torch.FloatTensor([0.5]))) 
# tensor([0.6723, 0.2694, 0.5441, 0.3754], grad_fn=<SigmoidBackward0>)
image_list = []

# 판별기와 생성기 훈련
for i in range(10000):
    # 1단계 : 참에 대해 판별기 훈련
    D.train(generate_real(),torch.FloatTensor([1.0]))
    # 2단계 : 거짓에 대해 판별기 훈련
    D.train(G.forward(torch.FloatTensor([0.5])).detach(),
            torch.FloatTensor([0.0]))
    # 3단계 : 생성기 훈련
    G.train(D, torch.FloatTensor([0.5]),torch.FloatTensor([1.0]))
    if (i % 1000 == 0):
        image_list.append(G.forward(torch.FloatTensor([0.5])).detach().numpy())
    
    pass

print("Done")
print("걸린 시간 :",time.time()-start_time)
# D.plot_progress()
# G.plot_progress()
# plt.show()
print(G.forward(torch.FloatTensor([0.5]))) 
# tensor([0.9973, 0.2525, 0.9960, 0.1659], grad_fn=<SigmoidBackward0>)

print(image_list)


plt.figure(figsize=(16,8))
plt.imshow(np.array(image_list).T,interpolation='none',cmap='Blues')
plt.show()

