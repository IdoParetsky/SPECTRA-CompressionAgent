from ..ModelWithRows import ModelWithRows


class BaseFE:
    model_with_rows: ModelWithRows

    def __init__(self, model_with_rows: ModelWithRows):
        self.model_with_rows: ModelWithRows = model_with_rows

    def extract_feature_map(self):
        """
        Placeholder method to extract features for a specific layer.
        Should be overridden by child classes.
        """
        raise NotImplementedError("Child class must implement extract_feature_map.")
    