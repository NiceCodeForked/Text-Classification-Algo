class TwoCropTransform:
    """
    Create two crops of the same sample for contrastive learning
    
    References
    ----------
    1. https://github.com/HobbitLong/SupContrast/blob/master/util.py
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]