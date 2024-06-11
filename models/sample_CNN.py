import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from tools import captcha_info


class my_CNN(nn.Module):

    def __init__(self):
        super(my_CNN, self).__init__()
        self.layers1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layers2 = nn.Sequential(
            nn.Linear(128 * 10 * 25, 1024), 
            nn.ReLU(),
            nn.Dropout(0.5) 
        )
        self.layers3 = nn.Sequential(
            nn.Linear(1024, captcha_info.Captcha_Len * captcha_info.Len_of_charset),
        )

    def forward(self, input):
        out = self.layers1(input)
        out = out.view(out.size(0), -1)
        out = self.layers2(out)
        out = self.layers3(out)
        return out


if __name__ == "__main__":
    my_model = my_CNN()
    input = torch.ones((64, 1, 80, 200))
    out = my_model(input)
    writer = SummaryWriter("./logs")
    writer.add_graph(my_model, input)
    writer.close()
    print(out.shape)
    torch.set_printoptions(edgeitems=100000, threshold=100000)
    print(out)
