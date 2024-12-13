# NetworkFeatureExtraction
Using this repo you can extract features from each layer from pytorch Dense network.

## Feature maps strcture
Six feature maps are extracted in each step. The feature maps contains three 'families':
1. Architectural
2. Weights
3. Activations

For each family type, 2 feature maps are constructed; one for the whole network and one for the wanted layer.
### Network feature maps:
#### Architectural feature maps:
![alt text](https://github.com/liorhirsch/NetworkFeatureExtraction/blob/master/example%20architecture%20model%20fm.png)

#### Activation & wights feature maps:
![alt text](https://github.com/liorhirsch/NetworkFeatureExtraction/blob/master/example%20activation%26weights%20fm.png)

### Network feature maps:

#### Architectural feature maps:
The layer's architectural feature map is represented similar to the network's architecture fm, but contains only one row according to the selected layer.

#### Activation & wights feature maps:
![alt text](https://github.com/liorhirsch/NetworkFeatureExtraction/blob/master/layer%20activation%20weight%20fm.png)

## Code Example
The main class is `FeatureExtractor`. This class is initlized with the model, the train data (without the target column) and the device (cpu/gpu)
The following code created a `FeatureExtractor` object and extract features from the first hidden layer.

```
feature_extractor = FeatureExtractor(self.loaded_model.model, self.X_data._values, device)
fm = feature_extractor.extract_features(0)
```

