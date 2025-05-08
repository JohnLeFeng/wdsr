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
    args.add_argument('-s', '--scale', type=int, default=2,
                      help='Optional. Scale factor for image super-resolution.')
    args.add_argument('-ds', '--do_static', default = True, action=argparse.BooleanOptionalAction,
                      help='Optional. Whether to execute text encoder.')

    return parser.parse_args()

def main():
    args = parse_args()
    
    image_name = args.image_name
    model_type = args.model_type

    HEIGHT = args.input_height
    WIDTH = args.input_width
    SCALE = args.scale

    output_img_path = (
        args.output_dir + image_name.split('/')[-1].split('.')[0]
    )

    if (args.do_static):
        log.info ("Using static model...")
        ov_model_path = "./ov_models/{}_{}x_{}_{}.xml".format(model_type, SCALE, HEIGHT, WIDTH)
        output_img_path += "_{}_{}x_{}_{}.jpg".format(model_type, SCALE, HEIGHT, WIDTH)
    else:
        log.info ("Using dynamic model...")
        ov_model_path = "./ov_models/{}_{}x_dyn.xml".format(model_type, SCALE)
        output_img_path += "_{}_{}x_dyn.jpg".format(model_type, SCALE)


    core = ov.Core()
    compiled_model = core.compile_model(ov_model_path, "GPU")

    img = cv2.imread(image_name)
    if (args.do_static):
        img = cv2.resize(img, (WIDTH, HEIGHT))
    result = compiled_model(np.expand_dims(np.transpose(img, (2, 0, 1)), 0))[0]
    result_image = np.transpose(result[0] * 255, (1, 2, 0))

    cv2.imwrite(output_img_path, result_image)
    log.info ("Result is saved to '%s'.", output_img_path)

if __name__ == '__main__':
    sys.exit(main())