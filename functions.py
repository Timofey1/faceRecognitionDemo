import numpy as np
from PIL import Image
import math
from torch.nn.functional import interpolate
import torch
import cv2


def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data


def crop_resize(img, image_size):
    if isinstance(img, np.ndarray):
        out = cv2.resize(
            img,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA
        ).copy()
    elif isinstance(img, torch.Tensor):
        out = imresample(
            img.permute(2, 0, 1).unsqueeze(0).float(),
            (image_size, image_size)
        ).byte().squeeze(0).permute(1, 2, 0)
    else:
        out = img.copy().resize((image_size, image_size), Image.BILINEAR)
    return out


def compareEmbeddings(emb1, emb2, distance):
    if distance == "cosine":
        dis = findCosineDistance(emb1, emb2)
        threshold = findThreshold(distance)
    if distance == "euclidean":
        dis = findEuclideanDistance(emb1, emb2)
        threshold = findThreshold(distance)
    if distance == "euclidean_l2":
        dis = findEuclideanDistance(l2_normalize(emb1), l2_normalize(emb2))
        threshold = findThreshold(distance)
    return dis, threshold


def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def findEuclideanDistance(source_representation, test_representation):
    if type(source_representation) == list:
        source_representation = np.array(source_representation)

    if type(test_representation) == list:
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))


def findThreshold(distance_metric):
    base_threshold = {'cosine': 0.4, 'euclidean': 1, 'euclidean_l2': 0.75}
    return base_threshold[distance_metric]


def alignment_procedure(img, left_eye, right_eye):
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # find rotation direction
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    # find length of triangle edges
    a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))

    # apply cosine rule
    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation

        cos_a = (b * b + c * c - a * a) / (2 * b * c)
        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # rotate base image
        if direction == -1:
            angle = 90 - angle

        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))
    return img  # return img anyway
