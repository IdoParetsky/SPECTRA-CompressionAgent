import os
import time
import numpy as np
import pandas as pd
import torch
from torch import optim

from .ModelClasses.LoadedModel import MissionTypes, LoadedModel
from .FeatureExtractors.ModelFeatureExtractor import FeatureExtractor
from .ModelClasses.NetX.netX import NetX


def load_checkpoint(filepath, device) -> LoadedModel:
    checkpoint = torch.load(filepath, map_location=torch.device(device))
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    # model.eval()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(checkpoint['optimizer'])

    mission_type = MissionTypes.Classification if checkpoint['mission_type'] == 'Classification' \
                                               else MissionTypes.Regression
    loss = checkpoint['loss']



    return LoadedModel(model, optimizer, mission_type, loss)

def load_model_and_data(model_path, x_path, y_path, device):
    """
    :param model_path:
    :param data_path:
    :param layer_index:
    :return:
    """
    X = pd.read_csv(x_path)
    Y = pd.read_csv(y_path)
    loaded_model = load_checkpoint(model_path, device)
    return loaded_model, X, Y

def get_fm_for_model_and_layer(model, data, layer_index):
    feature_extractor = FeatureExtractor(model, data._values)
    return feature_extractor, feature_extractor.extract_features(layer_index)

def main():
    for root, dirs, files in os.walk("../Convolutional Network Training/"):
        load_times = []
        fe_times = []

        if ('X_train.csv' not in files):
            continue

        X = pd.read_csv(root + '/X_train.csv')

        model_files = filter(lambda file_name: file_name.endswith('.pt'), files)

        for file in model_files:

            start = time.time()
            model = load_checkpoint(os.path.join(root, file))
            end = time.time()
            load_times.append(end - start)

            start = time.time()

            end = time.time()
            fe_times.append(end-start)

        print("Len dataset : ", len(X))
        print("Load Times : ", np.mean(load_times), " | Min : ", min(load_times), " | Max : ", max(load_times))
        print("Feature Extraction Times : ", np.mean(fe_times), " | Min : ", min(fe_times), " | Max : ", max(fe_times))
        print("Done ", root)

