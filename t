import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

lambda_reg = 0.01

for epoch in range(100): 
    inputs = torch.randn(32, 10)  
    targets = torch.randn(32, 1)  
    
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param, 2)
    
    loss += lambda_reg * l2_reg
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')
