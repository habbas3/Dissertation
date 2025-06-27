from torch import nn
import warnings
from models.Self_Attention import SelfAttention as attention_sa
from models.Self_Attention import OutlierAttention as attention_outliers


class CNN(nn.Module):
    def __init__(self, num_classes=3, pretrained=False, in_channel=1):
        super(CNN, self).__init__()
        if pretrained:
            warnings.warn("Pretrained model is not available")

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(8)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4)
        )

        # Attention layers
        self.attn = attention_sa(128)
        self.attn_outliers = attention_outliers(128)

        # Feature extractor
        self.layer5 = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(
                nn.Linear(128 * 4, 256), n_power_iterations=1, eps=1e-12
            ),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        # Final classifier layer (3 classes)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attn(x)
        x = self.attn_outliers(x)
        x = x.view(x.size(0), -1)
        features = self.layer5(x)
        logits = self.classifier(features)
        return logits, features


class cnn_features(nn.Module):
    def __init__(self, pretrained=False, num_classes=3, in_channel=1):
        super(cnn_features, self).__init__()
        self.model_cnn = CNN(num_classes=num_classes, pretrained=pretrained, in_channel=in_channel)
        self.__in_features = 256

    def forward(self, x):
        return self.model_cnn(x)

    def output_num(self):
        return self.__in_features
