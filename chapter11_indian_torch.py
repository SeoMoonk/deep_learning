import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = (torch.device('cuda') 
          if torch.cuda.is_available()
             else torch.device('cpu'))

dataset_numpy = np.loadtxt('data/pima-indians-diabetes3.csv', delimiter = ',', skiprows=1)

dataset = torch.from_numpy(dataset_numpy) #torch가 numpy 데이터를 tensor 데이터로 변환

X = dataset[:, :-1]
y = dataset[:, -1]

X.shape

y.shape

class Pima(nn.Module):
    def __init__(self):
        super(Pima, self).__init__()
        self.hidden_linear1 = nn.Linear(8, 12)
        self.activation = nn.ReLU()
        self.hidden_linear2 = nn.Linear(12, 8)
        self.hidden_linear3 = nn.Linear(8, 1)
        self.output_activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.hidden_linear1(x)
        x = self.activation(x)
        x = self.hidden_linear2(x)
        x = self.activation(x)
        x = self.hidden_linear3(x)
        x = self.output_activation(x)
        return x
    
    def predict(self, x):
        pred = self.forward(x)
        if pred >= 0.5:
            return 1
        else:
            return 0

model = Pima().to(device) #(설정에 따라)cpu 혹은 gpu에 만들어서 연산해라.
model

def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            if param.dim() > 1:
                print(name, ':', ' x '.join(str(x) for x in list(param.size())[::-1]), '=', num_param)
            else:
                print(name, ':', num_param)
                print('-' * 40)
            total_param += num_param
    print('total:', total_param)

count_parameters(model)

ds = TensorDataset(X, y)
dataloader = DataLoader(ds, batch_size=5)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.BCELoss()
n_epochs = 100

for epoch in range(n_epochs):
    for data, label in dataloader:
        data = data.type(torch.FloatTensor)
        out = model(data.to(device))
        loss = loss_fn(out, label.type(torch.FloatTensor).unsqueeze(1).to(device))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

train_loader = DataLoader(ds)
correct = 0

with torch.no_grad():
    for data, label in train_loader:
        predicted = model.predict(data.type(torch.FloatTensor).to(device))
        target = int(label[0])
        correct += 1 if predicted == target else 0

print("Accuracy: %f" % (correct / len(train_loader.dataset)))



