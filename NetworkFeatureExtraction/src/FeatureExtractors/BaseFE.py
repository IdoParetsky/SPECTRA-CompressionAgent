from ..ModelWithRows import ModelWithRows


class BaseFE:
    model_with_rows: ModelWithRows

    def __init__(self, model_with_rows: ModelWithRows):
        self.model_with_rows: ModelWithRows = model_with_rows

    def extract_feature_map(self, layer_index):
        """
        Placeholder method to extract features for a specific layer.
        Should be overridden by child classes.
        """
        raise NotImplementedError("Child class must implement extract_feature_map.")

    def extract_features_all_layers(self):
        """
        Extract features for all layers in the model.
        Iterates over all layers and collects features using the `extract_feature_map` method.

        Returns:
            List[Dict]: A list of feature maps for all layers.
        """
        all_features = []
        for layer_index in range(len(self.model_with_rows.all_rows)):
            features = self.extract_feature_map(layer_index)
            all_features.append(features)
        return all_features
    