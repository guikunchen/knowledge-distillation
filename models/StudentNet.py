import torch.nn as nn


def dwpw_conv(in_chs, out_chs, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_chs, in_chs, kernel_size, stride, padding, groups=in_chs),
        nn.Conv2d(in_chs, out_chs, 1),
    )


class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()

        self.cnn = nn.Sequential(
            dwpw_conv(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            dwpw_conv(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            dwpw_conv(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            dwpw_conv(64, 100, 3, 1, 1),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            # Here we adopt Global Average Pooling for various input size.
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(100, 11),
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
