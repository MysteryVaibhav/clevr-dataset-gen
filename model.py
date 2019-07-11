import torch
import torch.nn as nn
import torch.nn.functional as F


class DIFFSPOT(torch.nn.Module):
    def __init__(self, params):
        super(DIFFSPOT, self).__init__()
        self.regions_in_image = params.regions_in_image
        self.layers = nn.ModuleList([nn.Linear(in_features=params.visual_feature_dimension,
                                               out_features=params.hidden_dimension),
                                     nn.LeakyReLU(0.2),
                                     nn.Linear(in_features=params.hidden_dimension,
                                               out_features=params.hidden_dimension)])
        self.diff_layers = nn.ModuleList([nn.Linear(in_features=params.visual_feature_dimension,
                                                    out_features=params.hidden_dimension),
                                     nn.LeakyReLU(0.2),
                                     nn.Linear(in_features=params.hidden_dimension,
                                               out_features=params.hidden_dimension)])
        self.projection_layer = nn.Linear(in_features=2 * params.regions_in_image * params.hidden_dimension,
                                          out_features=params.output_dimension)

    def forward(self, img, diff_img):
        for l1, l2 in zip(self.layers, self.diff_layers):
            img = l1(img)
            diff_img = l2(diff_img)
        feat_diff = F.sigmoid(img * diff_img)
        feat_diff = torch.cat((img, feat_diff), dim=1)
        logit = self.projection_layer(feat_diff.view(len(img), -1))
        return logit