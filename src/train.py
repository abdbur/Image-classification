import torch
import torch.nn as nn

def train_model(net, train_loader, optimizer, criterion, epochs=10, device='cpu'):
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"[{epoch + 1}] loss: {running_loss / len(train_loader):.3f}")
