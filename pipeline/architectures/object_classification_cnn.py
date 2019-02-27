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


class ObjectClassificationModel(TUBerlinClassificationModel):
    @property
    def Dataset(self):
        return ObjectDataset

    def get_validation_transformation(self):
        def to_comic(**kwargs):
            img = kwargs['image']
            img = cv2.blur(img, (7, 7))
            newImg = np.zeros(img.shape, np.uint8)
            #ret, thresh = cv2.threshold(img, 127, 255, 0)
            thresh = cv2.Canny(img, 100, 200)

            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(newImg, contours, -1, 255, 1)
            cv2.imshow('swag', thresh)
            kwargs['image'] = newImg
            cv2.waitKey(0)
            return kwargs
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

    def load_image(self, img_name):
        return cv2.imread(img_name, 0)
