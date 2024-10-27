import torch
from sklearn.metrics import accuracy_score

def evaluate_model(net, test_loader, device='cpu'):
    net.eval()
    y_true = []
    y_predict = []
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = net(data)
            _, predicted = torch.max(outputs, 1)
            y_predict.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    accuracy = accuracy_score(y_true, y_predict)
    print(f'Accuracy: {accuracy:.4f}')
    return accuracy
