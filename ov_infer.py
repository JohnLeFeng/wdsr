import os
import cv2
import sys
import logging
import argparse

import numpy as np
import openvino as ov

from pathlib import Path

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout) 
log = logging.getLogger()

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', 
                      help='Show this help message and exit.')
    args.add_argument('-i', '--input_name', type=str, required=True, 
                      help='path of the input file (a file)')
    args.add_argument('-o', '--output_dir', type=str, default="output_files/",
                      help='paht for saving the predicted alpha matte (a file)')
    args.add_argument('-ih', '--input_height', type = int, default = 512,
                      help='Optional. Height of input.')
    args.add_argument('-iw', '--input_width', type = int, default = 512,
                      help='Optional. Width of input.')
    args.add_argument('-mt', '--model_type', type=str, default="wdsr",
                      help='Optional. Type of wdsr model.')
    args.add_argument('-d', '--device', type=str, default="GPU",
                      help='Optional. Specify the target device to infer on; CPU, GPU or NPU '
                      'is acceptable. Default value is GPU.')
    args.add_argument('-mp', '--model_precision', type=str, default="FP16",
                      help='Optional. Precision of model.')
    args.add_argument('-s', '--scale', type=int, default=2,
                      help='Optional. Scale factor for image super-resolution.')
    args.add_argument('-ds', '--do_static', default = True, action=argparse.BooleanOptionalAction,
                      help='Optional. Whether to execute text encoder.')

    return parser.parse_args()

def main():
    args = parse_args()
    
    INPUT_NAME = Path(args.input_name)
    MODEL_TYPE = args.model_type
    MODEL_PRECISION = args.model_precision

    HEIGHT = args.input_height
    WIDTH = args.input_width
    DEVICE = args.device
    SCALE = args.scale

    output_file_path = (
        args.output_dir + str(INPUT_NAME).split('\\')[-1].split('.')[0]
    )

    if (args.do_static):
        log.info ("Using static {} model...".format(MODEL_PRECISION))
        ov_model_path = "./ov_models/{}/{}_{}x_{}x{}.xml".format(MODEL_PRECISION, MODEL_TYPE, SCALE, HEIGHT, WIDTH)
        output_file_path += "_{}_{}x_{}x{}_{}{}".format(MODEL_TYPE, SCALE, HEIGHT, WIDTH, MODEL_PRECISION, str(INPUT_NAME.suffix))
    else:
        log.info ("Using dynamic {} model...".format(MODEL_PRECISION))
        ov_model_path = "./ov_models/{}/{}_{}x_dyn.xml".format(MODEL_PRECISION, MODEL_TYPE, SCALE)
        output_file_path += "_{}_{}x_dyn_{}{}".format(MODEL_TYPE, SCALE, MODEL_PRECISION, str(INPUT_NAME.suffix))
    print (output_file_path)

    core = ov.Core()
    compiled_model = core.compile_model(ov_model_path, DEVICE)

    cap = cv2.VideoCapture(str(INPUT_NAME))

    if cap.isOpened():
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            VIDEO_FOURCC = int(cap.get(cv2.CAP_PROP_FOURCC))
            VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS)
            VIDEO_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * SCALE)
            VIDEO_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * SCALE)

            OUTPUT_VIDEO = cv2.VideoWriter(output_file_path, VIDEO_FOURCC, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT), isColor=True)
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    if (args.do_static):
                        frame = cv2.resize(frame, (WIDTH, HEIGHT))
                    result = compiled_model(np.expand_dims(np.transpose(frame, (2, 0, 1)), 0))[0]
                    result_frame = np.clip(np.transpose(result[0] * 255, (1, 2, 0)), 0, 255).astype(np.uint8)
                    OUTPUT_VIDEO.write(result_frame)
                else:
                    break
            OUTPUT_VIDEO.release()
        else:
            img = cv2.imread(INPUT_NAME)
            if (args.do_static):
                img = cv2.resize(img, (WIDTH, HEIGHT))
            result = compiled_model(np.expand_dims(np.transpose(img, (2, 0, 1)), 0))[0]
            result_image = np.transpose(result[0] * 255, (1, 2, 0))

            cv2.imwrite(output_file_path, result_image)
    cap.release()
    log.info ("Result is saved to '%s'.", output_file_path)

if __name__ == '__main__':
    sys.exit(main())