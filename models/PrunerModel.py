import torch
from torchvision import models
import torch.nn as nn
torch.autograd.set_detect_anomaly(True)
distance = nn.CrossEntropyLoss()


class PruningNetwork(nn.Module):
    """ Nothing special about the pruning model,
    it is a standard resnet predictive model. Might update it later
    """
    def __init__(self, config):
        super(PruningNetwork, self).__init__()
        self.pruning_ratio = config["pruning_ratio"]
        self.pruning_style = config["pruning_style"]
        # decoy layer to allow creation of optimizer
        self.decoy_layer = nn.Linear(10, 10)
        self.temp = 1/30

        if self.pruning_style == "network":
            self.logits = config["logits"]
            self.split_layer = config["split_layer"]
            self.model = models.resnet50(pretrained=False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(nn.Flatten(),
                                          nn.Linear(num_ftrs, self.logits))

            self.model = nn.ModuleList(self.model.children())
            self.model = nn.Sequential(*self.model)

    def prune_channels(self, z, indices=None):
        # Indexing is an inplace operation which creates problem during backprop hence z is cloned first
        z = z.clone()
        z[:, indices] = 0.
        return z

    @staticmethod
    def get_random_channels(x, ratio):
        num_channels = x.shape[1]
        num_prunable_channels = int(num_channels * ratio)
        channels_to_prune = torch.randperm(x.shape[1], device=x.device)[:num_prunable_channels]
        return channels_to_prune

    def custom_sigmoid(self, x, offset):
        exponent = (x - offset) / self.temp
        #print(exponent)
        #answer = (1 / (1 + torch.exp( - exponent / self.temp)))
        answer = nn.Sigmoid()(exponent)
        #print(answer)
        return answer

    def get_channels_from_network(self, x, ratio):
        fmap_score = self.network_forward(x)
        num_channels = x.shape[1]
        num_prunable_channels = int(num_channels * ratio)
        threshold_score = torch.sort(fmap_score)[0][:, num_prunable_channels].unsqueeze(1)
        fmap_score = self.custom_sigmoid(fmap_score, threshold_score)
        # pruning_vector = fmap_score.unsqueeze(dim=2).unsqueeze(dim=3)
        # x = x * pruning_vector
        index_array = torch.arange(num_channels).repeat(x.shape[0], 1).cuda()
        indices = index_array[fmap_score < 0.5]
        return indices

    def network_forward(self, x):
        for i, l in enumerate(self.model):
            if i < self.split_layer:
                continue
            x = l(x)
        return x

    def forward(self, x):
        if self.pruning_style == "random":
            indices = self.get_random_channels(x, self.pruning_ratio)
            x = self.prune_channels(x, indices)
        elif self.pruning_style == "nopruning":
            indices = []
        elif self.pruning_style == "network":
            indices = self.get_channels_from_network(x, self.pruning_ratio)
            x = self.prune_channels(x, indices)
        return x, indices
