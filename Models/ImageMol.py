import torch
import torchvision
import torch.nn as nn
import math
def get_support_model_names():
    return ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]


def load_model(modelname="ResNet18", imageSize=224, num_classes=2):
    assert modelname in get_support_model_names()
    if modelname == "ResNet18":
        model = torchvision.models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet34":
        model = torchvision.models.resnet34(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet50":
        model = torchvision.models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet101":
        model = torchvision.models.resnet101(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif modelname == "ResNet152":
        model = torchvision.models.resnet152(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise Exception("{} is undefined".format(modelname))
    return model


class ImageMol(nn.Module):
    def __init__(self, baseModel, jigsaw_classes, label1_classes, label2_classes, label3_classes):
        super(ImageMol, self).__init__()

        assert baseModel in get_support_model_names()

        self.baseModel = baseModel

        self.embedding_layer = nn.Sequential(*list(load_model(baseModel).children())[:-1])

        self.bn = nn.BatchNorm1d(512)

        self.jigsaw_classifier = nn.Linear(512, jigsaw_classes)
        self.class_classifier1 = nn.Linear(512, label1_classes)
        self.class_classifier2 = nn.Linear(512, label2_classes)
        self.class_classifier3 = nn.Linear(512, label3_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.embedding_layer(x)
        x = x.view(x.size(0), -1)

        x1 = self.jigsaw_classifier(x)
        x2 = self.class_classifier1(x)
        x3 = self.class_classifier2(x)
        x4 = self.class_classifier3(x)

        return x, x1, x2, x3, x4
