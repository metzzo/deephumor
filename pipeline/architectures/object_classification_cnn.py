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


class ObjectClassificationModel(TUBerlinClassificationModel):
    @property
    def Dataset(self):
        return ObjectDataset

    def get_validation_transformation(self):
        def to_comic(**kwargs):
            img = kwargs['image']
            img = cv2.fastNlMeansDenoising(img)
            img = cv2.fastNlMeansDenoising(img)
            img = cv2.fastNlMeansDenoising(img)
            img = cv2.blur(img, (5, 5))
            newImg = np.zeros(img.shape, np.uint8)
            #ret, thresh = cv2.threshold(img, 127, 255, 0)
            thresh = cv2.Canny(img, 100, 200)

            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # if contour near enough => combine
            """
            newContours = []
            dist_thres = max(img.shape[0] * 0.1, img.shape[1] * 0.1)
            for c1 in contours:
                p1, p2 = np.array(c1[0][0]), np.array(c1[1][0])

                for c2 in contours:
                    p3, p4 = np.array(c2[0][0]), np.array(c2[1][0])

                    if (p1 - p3 == 0).all() and (p2 - p4 == 0).all():
                        continue

                    if numpy.linalg.norm(p1-p3) < dist_thres and numpy.linalg.norm(p2-p4) < dist_thres:
                        newContours.append([
                            *(p1+p3)/2.0,
                            *(p2+p4)/2.0,
                        ])
                        break
                    if numpy.linalg.norm(p1-p4) < dist_thres and numpy.linalg.norm(p2-p3) < dist_thres:
                        newContours.append([
                            *(p1+p4)/2.0,
                            *(p2+p3)/2.0,
                        ])
                        break
            """
            cv2.drawContours(newImg, contours, -1, 255, 1)
            #cv2.imshow('swag', newImg)
            kwargs['image'] = newImg
            #cv2.waitKey(0)
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

