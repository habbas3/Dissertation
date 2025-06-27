#!/usr/bin/python
# -*- coding:utf-8 -*-
from models.CNN_1 import CNN as CNN_1d
from models.cnn_1d import cnn_features as cnn_features_1d
from models.cnn_1d_selfattention import cnn_features as cnn_features_1d_sa
from models.AdversarialNet import AdversarialNet
from models.resnet18_1d import resnet18_features as resnet_features_1d
from models.Resnet1d import resnet18 as resnet_1d
from models.classifier_OSBP import ClassifierNet as classifier_OSBP
from models.AdversarialNet import AdversarialNet_auxiliary
from models.sngp import SNGP as sngp
from models.spectral_normalization import spectral_norm
from models.wideresnet import WideResNet
from models.wideresnet_self_attention import WideResNet_sa
from models.wideresnet_multihead_attention import WideResNet_mh
from models.wideresnet_edited import WideResNet_edited
from models.attention import MultiHeadAttention as attention
from models.Self_Attention import SelfAttention as attention_sa
from models.Self_Attention import OutlierAttention as attention_outliers
# from models.cnn_sa_openmax import CNN_OpenMax as cnn_openmax
from models.cnn_sa_openmax_habbas import CNN_OpenMax as cnn_openmax
from models.cnn_1d_habbas_hyperparstudy import CNN as cnn_features_1d_hyperparstudy
# from models.AttentionCNN import AttentionCNN as attention