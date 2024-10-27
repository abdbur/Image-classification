import torch
from src.model import Net
from torchvision import transforms
from PIL import Image

def predict_image(image, model_path='models/model.pth', device='cpu'):
    net = Net().to(device)
    net.load_state_dict(torch.load(model_path, weights_only=True))
    net.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = net(img)
        print(output)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()