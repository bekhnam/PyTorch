from torch.cuda.amp import autocast
from torch import nn

class FoodClassifier(nn.Module):
    def __init__(self, baseModel, numClasses):
        super(FoodClassifier, self).__init__()
        self.baseModel = baseModel
        self.classifier = nn.Linear(baseModel.classifier.in_features, numClasses)
        self.baseModel.classifier = nn.Identity()

    # decorate the forward method with autocast to enable
    # mixed-precision training in a distributed manner
    @autocast()
    def forward(self, x):
        features = self.baseModel(x)
        logits = self.classifier(features)
        return logits