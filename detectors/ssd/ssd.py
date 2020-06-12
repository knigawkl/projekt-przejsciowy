from platform import python_version
import subprocess
import os

import keras
from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt

from detectors.ssd.models.keras_ssd512 import ssd_512
from detectors.ssd.keras_loss_function.keras_ssd_loss import SSDLoss
from detectors.ssd.keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from detectors.ssd.keras_layers.keras_layer_DecodeDetections import DecodeDetections
from detectors.ssd.keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from detectors.ssd.keras_layers.keras_layer_L2Normalization import L2Normalization

from utils.logger import logger


IMG_HEIGHT = 512
IMG_WIDTH = 512
MODEL_MODE = 'inference'
CONF_THRESH = 0.01
WEIGHTS_PATH = './data/ssd512-hollywood-trainval-bs_16-lr_1e-05-scale_pascal-epoch-187-py3.6.h5'


def find_heads_ssd(img_path: str, confidence_thresh: float):
    ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

    # Clear previous models from memory.
    K.clear_session()

    if MODEL_MODE == 'inference':
        decode_layer = DecodeDetections(img_height=IMG_HEIGHT,
                                        img_width=IMG_WIDTH,
                                        confidence_thresh=CONF_THRESH,
                                        iou_threshold=0.45,
                                        top_k=200,
                                        nms_max_output_size=400)
    elif MODEL_MODE == 'inference_fast':
        decode_layer = DecodeDetectionsFast(img_height=IMG_HEIGHT,
                                            img_width=IMG_WIDTH,
                                            confidence_thresh=CONF_THRESH,
                                            iou_threshold=0.45,
                                            top_k=200,
                                            nms_max_output_size=400)

    model = load_model(WEIGHTS_PATH, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                     'L2Normalization': L2Normalization,
                                                     'DecodeDetections': decode_layer,
                                                     'compute_loss': ssd_loss.compute_loss})
    orig_images = []
    input_images = []

    orig_images.append(image.load_img(img_path))
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img = image.img_to_array(img)
    input_images.append(img)
    input_images = np.array(input_images)

    y_pred = model.predict(input_images)

    y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_thresh] for k in range(y_pred.shape[0])]
    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    logger.info("Predicted boxes:\n")
    logger.info('   class   conf xmin   ymin   xmax   ymax')
    logger.info(y_pred_thresh[0])
