import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


def load_base_model(base_model):
    if base_model == 'inception':
        model = models.inception_v3(pretrained=True)
    elif base_model == 'densenet161':
        model = models.densenet161(pretrained=True)
    elif base_model == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        raise TypeError(
            'The specified model is not available. The choices are "inception", "densenet161", and "vgg16".')

    return model


class Classifier(nn.Module):
    def __init__(self, features_in, hidden_units, use_dropout, dropout_ratio):
        super().__init__()
        self.features_in = features_in
        self.hidden_units = hidden_units

        self.fc1 = nn.Linear(self.features_in, self.hidden_units)
        self.fc2 = nn.Linear(self.hidden_units, 102)

        self.dropout = nn.Dropout(dropout_ratio)
        self.use_dropout = use_dropout
        self.dropout_ratio = dropout_ratio


    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = F.log_softmax(x, dim=1)
        return x


def get_classifier_inputs_number(model):
    parameters = list(model.classifier.named_parameters())
    _, first_param = parameters[0]
    dimensions = list(first_param.size())
    return int(dimensions[1])


def get_model(model, classifier):
    # Freeze the lower layers
    for param in model.parameters():
        param.require_grad = False

    model.classifier = classifier
    return model


def load_trained_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    features_in = checkpoint["features_in"]
    hidden_units = checkpoint["hidden_units"]
    arch = checkpoint["arch"]
    state_dict = checkpoint["state_dict"]
    mapping = checkpoint["mapping"]
    use_dropout = checkpoint["use_dropout"]
    dropout_ratio = checkpoint["dropout_ratio"]

    model = load_base_model(arch)
    classifier = Classifier(features_in=features_in,
                            hidden_units=hidden_units,
                            use_dropout=use_dropout,
                            dropout_ratio=dropout_ratio)

    for param in model.parameters():
        param.require_grad = False

    model.classifier = classifier
    model.load_state_dict(state_dict)

    return model, mapping


def save_model(save_directory, model, arch, classifier, optimizer, epoch, mapping):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    checkpoint = {"state_dict": model.state_dict(),
                  "optimizer_dict": optimizer.state_dict(),
                  "epoch": epoch,
                  "mapping": mapping,
                  "arch": arch,
                  "features_in": classifier.features_in,
                  "hidden_units": classifier.hidden_units,
                  "use_dropout": classifier.use_dropout,
                  "dropout_ratio": classifier.dropout_ratio
                  }

    torch.save(checkpoint, os.path.join(save_directory, "checkpoint.pth"))