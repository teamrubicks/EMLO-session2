import torch.nn as nn
from torchvision.models import resnet18
import torch


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(pretrained=False, num_classes=10)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.maxpool = nn.Identity()

    def forward(self, x):
        return self.model(x)

    def load_weights(self, model_path):
        return self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
