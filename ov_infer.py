import os
import cv2
import sys
import logging
import argparse

import numpy as np
import openvino as ov

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout) 
log = logging.getLogger()

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', 
                      help='Show this help message and exit.')
    args.add_argument('-i', '--image_name', type=str, required=True, 
                      help='path of the input image (a file)')
    args.add_argument('-o', '--output_dir', type=str, default="output_imgs/",
                      help='paht for saving the predicted alpha matte (a file)')
    args.add_argument('-ih', '--input_height', type = int, default = 512,
                      help='Optional. Height of input.')
    args.add_argument('-iw', '--input_width', type = int, default = 512,
                      help='Optional. Width of input.')
    args.add_argument('-mt', '--model_type', type=str, default="wdsr",
                      help='Optional. Type of wdsr model.')
    args.add_argument('-mp', '--model_precision', type=str, default="FP16",
                      help='Optional. Precision of model.')
    args.add_argument('-s', '--scale', type=int, default=2,
                      help='Optional. Scale factor for image super-resolution.')
    args.add_argument('-ds', '--do_static', default = True, action=argparse.BooleanOptionalAction,
                      help='Optional. Whether to execute text encoder.')

    return parser.parse_args()

def main():
    args = parse_args()
    
    IMAGE_NAME = args.image_name
    MODEL_TYPE = args.model_type
    MODEL_PRECISION = args.model_precision

    HEIGHT = args.input_height
    WIDTH = args.input_width
    SCALE = args.scale

    output_img_path = (
        args.output_dir + IMAGE_NAME.split('/')[-1].split('.')[0]
    )

    if (args.do_static):
        log.info ("Using static {} model...".format(MODEL_PRECISION))
        ov_model_path = "./ov_models/{}/{}_{}x_{}_{}.xml".format(MODEL_PRECISION, MODEL_TYPE, SCALE, HEIGHT, WIDTH)
        output_img_path += "_{}_{}x_{}_{}_{}.jpg".format(MODEL_TYPE, SCALE, HEIGHT, WIDTH, MODEL_PRECISION)
    else:
        log.info ("Using dynamic {} model...".format(MODEL_PRECISION))
        ov_model_path = "./ov_models/{}/{}_{}x_dyn.xml".format(MODEL_PRECISION, MODEL_TYPE, SCALE)
        output_img_path += "_{}_{}x_dyn_{}.jpg".format(MODEL_TYPE, SCALE, MODEL_PRECISION)


    core = ov.Core()
    compiled_model = core.compile_model(ov_model_path, "GPU")

    img = cv2.imread(IMAGE_NAME)
    if (args.do_static):
        img = cv2.resize(img, (WIDTH, HEIGHT))
    result = compiled_model(np.expand_dims(np.transpose(img, (2, 0, 1)), 0))[0]
    result_image = np.transpose(result[0] * 255, (1, 2, 0))

    cv2.imwrite(output_img_path, result_image)
    log.info ("Result is saved to '%s'.", output_img_path)

if __name__ == '__main__':
    sys.exit(main())