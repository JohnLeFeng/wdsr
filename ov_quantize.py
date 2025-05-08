import os
import cv2
import sys
import nncf
import torch
import logging
import argparse

import numpy as np
import openvino as ov

from torchvision import datasets
from torchvision import transforms

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout) 
log = logging.getLogger()

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', 
                      help='Show this help message and exit.')
    args.add_argument('-ih', '--input_height', type = int, default = 512,
                      help='Optional. Height of input.')
    args.add_argument('-iw', '--input_width', type = int, default = 512,
                      help='Optional. Width of input.')
    args.add_argument('-mt', '--model_type', type=str, default="wdsr",
                      help='Optional. Type of wdsr model.')
    args.add_argument('-s', '--scale', type=int, default=2,
                      help='Optional. Scale factor for image super-resolution.')
    args.add_argument('-ds', '--do_static', default = True, action=argparse.BooleanOptionalAction,
                      help='Optional. Whether to execute text encoder.')

    return parser.parse_args()

def transform_fn(data_item):
    images, _ = data_item
    return images

def main():
    args = parse_args()
    
    MODEL_TYPE = args.model_type

    HEIGHT = args.input_height
    WIDTH = args.input_width
    SCALE = args.scale

    if (args.do_static):
        log.info ("Using static model...")
        ov_model_path = "./ov_models/FP16/{}_{}x_{}_{}.xml".format(MODEL_TYPE, SCALE, HEIGHT, WIDTH)
        quantized_model_path = "./ov_models/INT8/{}_{}x_{}_{}_int8.xml".format(MODEL_TYPE, SCALE, HEIGHT, WIDTH)
    else:
        log.info ("Using dynamic model...")
        ov_model_path = "./ov_models/FP16/{}_{}x_dyn.xml".format(MODEL_TYPE, SCALE)
        quantized_model_path = "./ov_models/INT8/{}_{}x_dyn.xml".format(MODEL_TYPE, SCALE)


    core = ov.Core()
    ov_model = core.read_model(ov_model_path)


    to_0_255 = transforms.Lambda(lambda x: x * 255)
    dataset = datasets.ImageFolder(
        root="./quantization_datasets/pics4q",
        transform=transforms.Compose(
            [
                transforms.Resize([HEIGHT, WIDTH]),
                transforms.ToTensor(),
                to_0_255
            ]
        ),
    )

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    calibration_dataset = nncf.Dataset(data_loader, transform_fn)

    ov_quantized_model = nncf.quantize(
                ov_model, 
                calibration_dataset,                                                                                          
                preset = nncf.QuantizationPreset.PERFORMANCE,
                subset_size = 100,
    )
    ov.save_model(ov_quantized_model, quantized_model_path)
    log.info("{} Model converted to INT8 precision and saved to {}".format(MODEL_TYPE, quantized_model_path))

if __name__ == '__main__':
    sys.exit(main())