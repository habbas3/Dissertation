import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import MultiHeadAttention as attention
from models.Self_Attention import SelfAttention as attention_sa
from models.Self_Attention import OutlierAttention as attention_outliers
from scipy.stats import weibull_min
import numpy as np
import warnings

# CNN with self-attention and OpenMax
class CNN_OpenMax(nn.Module):
    def __init__(self, args, num_classes):
        super(CNN_OpenMax, self).__init__()
        self.num_classes = num_classes
        in_channels = getattr(args, "input_channels", 1)
        self.cnn = CNN(in_channel=in_channels)  # Base CNN model
        self.fc_input_size = 256  # Output of CNN
        self.fc = nn.utils.parametrizations.spectral_norm(nn.Linear(self.fc_input_size, self.num_classes))
        self.openmax = OpenMaxLayer(num_features=self.fc_input_size, num_classes=self.num_classes)

    def forward(self, x):
        x = self.cnn(x)  # Shape: [B, 256]
        features = x
        logits = self.fc(features)

        if self.openmax is not None and not self.training:
            if self.openmax.are_weibull_models_initialized():
                return self.openmax(logits, features)

        return logits

    def output_num(self):
        return self.fc.out_features


# Base CNN with attention
class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1):
        super(CNN, self).__init__()
        if pretrained:
            warnings.warn("Pretrained model is not available")

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=15),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4)
        )

        self.attn = attention_sa(128)
        self.attn_outliers = attention_outliers(128)

        self.layer5 = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(nn.Linear(128 * 4, 256)),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

    def forward(self, x):
        x = self.layer1(x)       # [B, 16, L']
        x = self.layer2(x)       # [B, 32, L']
        x = self.layer3(x)       # [B, 64, L']
        x = self.layer4(x)       # [B, 128, 4]
        x = self.attn(x)
        x = self.attn_outliers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.layer5(x)         # [B, 256]
        return x


# OpenMax Layer
class OpenMaxLayer(nn.Module):
    def __init__(self, num_features, num_classes, tail_size=70, alpha=10):
        super(OpenMaxLayer, self).__init__()
        self.num_classes = num_classes
        self.tail_size = tail_size
        self.alpha = alpha
        self.mean_vecs = torch.zeros(num_classes, num_features)
        self.weibull_models = {}

    def are_weibull_models_initialized(self):
        return all(i in self.weibull_models for i in range(self.num_classes))

    def fit_weibull(self, feature_vectors, labels):
        if isinstance(feature_vectors, np.ndarray):
            feature_vectors = torch.tensor(feature_vectors, device=self.mean_vecs.device)

        for i in range(self.num_classes):
            class_feature_vectors = feature_vectors[labels == i]
            mean_vector = torch.mean(class_feature_vectors, dim=0)
            self.mean_vecs[i] = mean_vector

            distances = torch.norm(class_feature_vectors - self.mean_vecs[i], dim=1)
            distances, _ = torch.sort(distances)
            tails = distances[-self.tail_size:]

            try:
                self.weibull_models[i] = weibull_min.fit(tails.cpu().numpy())
            except Exception as e:
                print(f"Error fitting Weibull model for class {i}: {e}")

    def compute_openmax(self, logits, features):
        features = features.view(features.size(0), -1)
        scores = torch.zeros_like(logits)

        for i in range(self.num_classes):
            distances = torch.norm(features - self.mean_vecs[i].to(features.device), dim=1)
            w_params = self.weibull_models[i]
            w_scores = weibull_min.cdf(distances.detach().cpu().numpy(), *w_params)
            w_scores = torch.tensor(w_scores, dtype=torch.float32, device=logits.device)
            scores[:, i] = logits[:, i] * (1 - w_scores**self.alpha)

        exp_scores = torch.exp(scores)
        probs = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)
        return probs

    def forward(self, logits, features):
        return self.compute_openmax(logits, features)
