# /usr/bin/env python3
# -*- coding:utf-8 -*-
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted,_num_samples
import numpy as np

class MyLabelEncoder(LabelEncoder):
    def __init__(self, ignore_unknown=False,
                       unknown_original_value='unknown',
                       unknown_encoded_value=-1):
        super(MyLabelEncoder, self).__init__()

        self.ignore_unknown = ignore_unknown
        self.unknown_original_value = unknown_original_value
        self.unknown_encoded_value = unknown_encoded_value

    def transform(self, y):
        check_is_fitted(self, 'classes_')
        # y = column_or_1d(y, warn=True)

        if _num_samples(y) == 0:
            return np.array([])

        indices = np.isin(y, self.classes_)
        if not self.ignore_unknown and not np.all(indices):
            raise ValueError("y contains new labels: %s"
                                         % str(np.setdiff1d(y, self.classes_)))

        y_transformed = np.searchsorted(self.classes_, y)
        y_transformed[~indices]=self.unknown_encoded_value
        return y_transformed

    def inverse_transform(self, y):
        check_is_fitted(self, 'classes_')

        labels = np.arange(len(self.classes_))
        indices = np.isin(y, labels)
        if not self.ignore_unknown and not np.all(indices):
            raise ValueError("y contains new labels: %s"
                                         % str(np.setdiff1d(y, self.classes_)))

        y_transformed = np.asarray(self.classes_[y], dtype=object)
        y_transformed[~indices]=self.unknown_original_value
        return y_transformed