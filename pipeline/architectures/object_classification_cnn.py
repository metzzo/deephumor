import numpy

import cv2
import numpy as np
from functools import partial

from albumentations import Resize
from albumentations.augmentations import transforms
from albumentations.pytorch import ToTensor

from architectures.tuberlin_classification_cnn import TUBerlinClassificationModel
from datamanagement.object_dataset import ObjectDataset
from datamanagement.tuberlin_dataset import TUBerlinDataset
from evaluation.accuracy_evaluation import AccuracyEvaluation
from evaluation.detailed_evaluation import DetailedEvaluation
from processing.utility import edge_detection


class ObjectClassificationModel(TUBerlinClassificationModel):
    @property
    def Dataset(self):
        return ObjectDataset

    def get_validation_transformation(self):
        def to_comic(**kwargs):
            img = kwargs['image']
            img = np.array(img)
            img = img[..., ::-1]
            img = edge_detection(img, to_pil=False)

            return {
                'image': img
            }
        return [
            to_comic,
            Resize(width=225, height=225),
            ToTensor(),
        ]

    @property
    def validation_evaluations(self):
        return super(ObjectClassificationModel, self).validation_evaluations + [partial(DetailedEvaluation, model=self)]

    def get_human_readable_class(self, cl, is_predicted):
        if is_predicted:
            return TUBerlinDataset.Classes[cl]
        else:
            return ObjectDataset.Classes[cl]

