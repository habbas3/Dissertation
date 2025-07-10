
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import MultiHeadAttention as attention
from models.Self_Attention import SelfAttention as attention_sa
from models.Self_Attention import OutlierAttention as attention_outliers
from scipy.stats import weibull_min
from sklearn.metrics.pairwise import euclidean_distances
from torch.autograd import Variable
import numpy as np



class OpenMaxLayer(nn.Module):
    def __init__(self, num_features, num_classes, tail_size=70, alpha=10):
        super(OpenMaxLayer, self).__init__()
        self.num_classes = num_classes
        # self.num_features = num_features
        self.tail_size = tail_size
        self.alpha = alpha
        self.mean_vecs = torch.zeros(num_classes, num_features)
        self.weibull_models = {}
        
    def are_weibull_models_initialized(self):
        return all(i in self.weibull_models for i in range(self.num_classes))
    
    def fit_weibull(self, feature_vectors, labels):
        """
        Fit the Weibull models for each class and feature based on the training set features.
        """
        # Convert feature_vectors to PyTorch tensor if it's a NumPy array
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
        """
        Compute the OpenMax probabilities given the logits and the feature vectors.
        """
        # Compute the euclidean distance from the mean activation vector
        features_processed = features.view(features.size(0), -1)  # Flatten features if necessary
        mean_vecs_processed = self.mean_vecs.view(1, self.num_classes, -1)  # Reshape for broadcasting
        dists = torch.norm(features_processed[:, None, :] - mean_vecs_processed, dim=2).flatten()

        
        # Compute the OpenMax score
        scores = torch.zeros((features.shape[0], self.num_classes))
        for i in range(self.num_classes):
            c = dists[i].detach().numpy()
            w_params = self.weibull_models[i]
            w_score = weibull_min.cdf(c, *w_params)
            scores[:, i] = logits[:, i] * (1 - torch.tensor(w_score) ** self.alpha)
        
        # Convert scores to probabilities
        exp_scores = torch.exp(scores)
        probs = exp_scores / torch.sum(exp_scores, axis=1, keepdims=True)
        return probs
    
    def forward(self, logits, features):
        """
        Forward pass for the OpenMax layer.
        """
        openmax_probs = self.compute_openmax(logits, features)
        return openmax_probs
    
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        # self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_planes)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
        #                        padding=1, bias=False)
        self.conv2 = nn.Conv1d(out_planes,out_planes,kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        # self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        #                        padding=0, bias=False) or None
        self.convShortcut = (not self.equalInOut) and nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

# class WideResNet_edited(nn.Module):
#     def __init__(self, num_classes, depth=28, widen_factor=2, drop_rate=0.3, input_size=32):
#         super(WideResNet_edited, self).__init__()
#         self.num_classes = num_classes
#         self.depth = depth
#         self.widen_factor = widen_factor
#         self.drop_rate = drop_rate
#         self.input_size = input_size

#         nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
#         assert ((depth-4) % 6 == 0), 'WideResNet depth should be 6n+4'
#         n = (depth-4) / 6
        
#         block = BasicBlock  # Assuming BasicBlock is defined elsewhere
#         self.conv1 = nn.Conv1d(1, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
#         self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate=drop_rate)
#         self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate=drop_rate)
#         self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate=drop_rate)

#         self.bn1 = nn.BatchNorm1d(nChannels[3])
#         self.relu = nn.ReLU(inplace=True)
#         self.fc = nn.Linear(nChannels[3], num_classes)
#         self.openmax = OpenMaxLayer(num_features=nChannels[3], num_classes=num_classes)
#         self.attn = attention_sa(nChannels[3])
#         self.attn_outliers = attention_outliers(nChannels[3])

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.block1(out)
#         out = self.block2(out)
#         out = self.block3(out)

#         out = self.attn(out)
#         out = self.attn_outliers(out)
#         out = self.relu(self.bn1(out))
#         out = F.avg_pool1d(out, out.size(2))
#         out = out.view(out.size(0), -1)

#         out = self.fc(out)

#         if self.openmax and not self.training:
#             out = self.openmax(out)

#         return out

#     def output_num(self):
#         return self.fc.in_features



class WideResNet_edited(nn.Module):
    def __init__(self, depth=16, widen_factor=1, drop_rate=0.3, num_classes=10, input_channels=7):
        super(WideResNet_edited, self).__init__()
        self.num_classes = num_classes
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        
        self.conv1 = nn.Conv1d(input_channels, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], BasicBlock, 1, dropRate=drop_rate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], BasicBlock, 2, dropRate=drop_rate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], BasicBlock, 2, dropRate=drop_rate)
        self.bn1 = nn.BatchNorm1d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.openmax = OpenMaxLayer(nChannels[3], num_classes)  # Initialize OpenMax layer

    def forward(self, x, return_features=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool1d(out, kernel_size=out.size()[2])
        out = out.view(out.size(0), -1)

        # Option to return features right before the final FC layer
        if return_features:
            return None, out  # Returning None as placeholder, adjust based on your specific needs

        out = self.fc(out)
        return out

    def forward_before_openmax(self, x):
        # Specifically retrieving features for OpenMax initialization
        return self.forward(x, return_features=True)

    def output_num(self):
        return self.fc.out_features
