import torch.utils.data
from util import *
import h5py
import json


class ClevrDiffDataSet(torch.utils.data.TensorDataset):
    def __init__(self, diff_desc_file, image_features_file, diff_image_features_file, image_feature_dimension, num_image_regions):
        super(ClevrDiffDataSet, self).__init__()
        self.diff_json = json.load(open(diff_desc_file, 'r'))['differences']
        self.image_feat_dim = image_feature_dimension
        self.image_regions = num_image_regions
        self.image_features = h5py.File(image_features_file, 'r')
        self.diff_image_features = h5py.File(diff_image_features_file, 'r')
        self.num_of_samples = len(self.image_features)
        # Mapping from attributes to vectors
        self.shapes_label_map = {'cube': 0, 'sphere': 1, 'cylinder': 2}
        self.colors_label_map = {'gray': 0, 'red': 1, 'blue': 2, 'green': 3, 'brown': 4, 'purple': 5, 'cyan': 6,
                                 'yellow': 7}
        self.materials_label_map = {'rubber': 0, 'metal': 1}
        self.sizes_label_map = {'large': 0, 'small': 1}

    def __len__(self):
        return self.num_of_samples

    def get_diff_vector(self, attributes):
        #vec = np.zeros((3, 1))
        #vec[self.shapes_label_map[attributes['shape']]] = 1
        #vec[0] = self.shapes_label_map[attributes['shape']]
        #vec[1] = self.colors_label_map[attributes['color']]
        #vec[2] = self.materials_label_map[attributes['material']]
        #vec[3] = self.sizes_label_map[attributes['size']]
        #return vec
        return self.shapes_label_map[attributes['shape']]

    def __getitem__(self, idx):
        # image = np.random.random((self.image_regions, self.image_feat_dim))
        # diff_image = np.random.random((self.image_regions, self.image_feat_dim))
        image = self.image_features[idx].reshape(self.image_feat_dim, self.image_regions).T
        diff_image = self.diff_image_features[idx].reshape(self.image_feat_dim, self.image_regions).T
        diff = self.get_diff_vector(self.diff_json[idx]['attributes'])
        return to_tensor(image), to_tensor(diff_image), to_tensor(diff).long()


class DataLoader:
    def __init__(self, params):
        self.params = params
        kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
        self.training_data_loader = torch.utils.data.DataLoader(ClevrDiffDataSet(params.train_diff,
                                                                                 params.train_img_feats,
                                                                                 params.train_diff_img_feats,
                                                                                 params.visual_feature_dimension,
                                                                                 params.regions_in_image),
                                                                batch_size=self.params.batch_size,
                                                                shuffle=True, **kwargs)

        self.dev_data_loader = torch.utils.data.DataLoader(ClevrDiffDataSet(params.dev_diff,
                                                                            params.dev_img_feats,
                                                                            params.dev_diff_img_feats,
                                                                            params.visual_feature_dimension,
                                                                            params.regions_in_image),
                                                           batch_size=self.params.batch_size,
                                                           shuffle=False, **kwargs)

        self.test_data_loader = torch.utils.data.DataLoader(ClevrDiffDataSet(params.test_diff,
                                                                             params.test_img_feats,
                                                                             params.test_diff_img_feats,
                                                                             params.visual_feature_dimension,
                                                                             params.regions_in_image),
                                                            batch_size=self.params.batch_size,
                                                            shuffle=False, **kwargs)
