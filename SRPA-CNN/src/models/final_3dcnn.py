# class Final3DCNN(nn.Module):
#     def __init__(self, num_bands, num_classes=2):
#         super().__init__()
#         self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
#         self.pool1 = nn.MaxPool3d((2, 2, 1))
#         self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
#         self.pool2 = nn.MaxPool3d((2, 2, 1))
#
#         self.fc = nn.Linear(16 * 12 * 12 * num_bands, num_classes)
#
#     def forward(self, x):  # x: (B, 1, 50, 50, k)
#         x = self.pool1(F.relu(self.conv1(x)))  # → (B, 8, 25, 25, k)
#         x = self.pool2(F.relu(self.conv2(x)))  # → (B, 16, 12, 12, k)
#         x = x.view(x.size(0), -1)  # flatten
#         return self.fc(x)

import torch
import torch.nn as nn


class Final3DCNN(nn.Module):
    def __init__(self, num_bands, num_classes=3):
        super(Final3DCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d((1, 1, 1))  # 🚨 This ensures output is always (B, 64, 1, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # (B, 64)
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)  # final output
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
