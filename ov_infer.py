import os
import cv2
import sys
import logging
import argparse

import numpy as np
import openvino as ov
from openvino.preprocess import ColorFormat, PrePostProcessor, ResizeAlgorithm

from pathlib import Path

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


# ---------------------------------------------------------------------------
# NV12 helper functions
# ---------------------------------------------------------------------------

def infer_layout_from_static_shape(shape):
    """Infer a common 4D image tensor layout from a fully static shape."""
    if len(shape) != 4:
        raise ValueError(f"Only 4D image tensors are supported, got shape: {shape}")
    if shape[1] in (1, 3):
        return "NCHW"
    if shape[3] in (1, 3):
        return "NHWC"
    return "NCHW"


def get_model_input_layout(port) -> str:
    """Resolve model input layout without requiring a static shape."""
    port_layout = getattr(port, "layout", None)
    if port_layout is not None:
        layout_name = str(port_layout)
        if layout_name and "?" not in layout_name and layout_name != "...":
            return layout_name

    try:
        static_shape = list(port.shape)
    except RuntimeError:
        static_shape = None

    if static_shape is not None:
        return infer_layout_from_static_shape(static_shape)

    return "NCHW"


def build_reshaped_input_shape(port, layout: str, input_height: int, input_width: int):
    """Build a concrete 4D input shape for a dynamic image model."""
    partial_shape = port.partial_shape
    if len(partial_shape) != 4:
        raise ValueError(f"Only 4D input tensors are supported for reshape, got: {partial_shape}")

    try:
        static_shape = list(port.shape)
    except RuntimeError:
        static_shape = [None] * 4

    def get_static_or_default(index: int, default_value: int) -> int:
        value = static_shape[index]
        return default_value if value is None else int(value)

    if layout == "NHWC":
        return [
            get_static_or_default(0, 1),
            input_height,
            input_width,
            get_static_or_default(3, 3),
        ]
    return [
        get_static_or_default(0, 1),
        get_static_or_default(1, 3),
        input_height,
        input_width,
    ]


def tensor_to_hwc_image(output_data: np.ndarray) -> np.ndarray:
    """Convert a model output tensor to an HWC uint8 BGR image."""
    array = np.array(output_data)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"Only batch size 1 is supported, got output shape: {array.shape}")
        if array.shape[1] in (1, 3):
            array = np.transpose(array[0], (1, 2, 0))
        elif array.shape[3] in (1, 3):
            array = array[0]
        else:
            raise ValueError(f"Cannot infer output image layout from shape: {array.shape}")
    elif array.ndim == 3:
        if array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        elif array.shape[2] not in (1, 3):
            raise ValueError(f"Unsupported output image shape: {array.shape}")
    else:
        raise ValueError(f"Unsupported output tensor rank: {array.ndim}")

    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)

    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        if array.size and np.nanmax(array) <= 1.5:
            array *= 255.0
        array = np.clip(array, 0.0, 255.0).astype(np.uint8)

    return array


def bgr_to_nv12(image_bgr: np.ndarray) -> bytes:
    """Convert an HWC BGR uint8 image to a raw NV12 frame."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected HWC BGR image with 3 channels, got shape: {image_bgr.shape}")

    height, width, _ = image_bgr.shape
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError(f"NV12 requires even width and height, got: {width}x{height}")

    image = image_bgr.astype(np.float32)
    blue  = image[:, :, 0]
    green = image[:, :, 1]
    red   = image[:, :, 2]

    y_plane = 0.114 * blue + 0.587 * green + 0.299 * red
    u_plane = 128.0 - 0.081312 * red - 0.418688 * green + 0.5 * blue
    v_plane = 128.0 + 0.5 * red - 0.331264 * green - 0.168736 * blue

    y_plane = np.clip(y_plane, 0.0, 255.0).astype(np.uint8)
    u_plane = np.clip(u_plane, 0.0, 255.0)
    v_plane = np.clip(v_plane, 0.0, 255.0)

    u_sub = (u_plane[0::2, 0::2] + u_plane[0::2, 1::2] + u_plane[1::2, 0::2] + u_plane[1::2, 1::2]) * 0.25
    v_sub = (v_plane[0::2, 0::2] + v_plane[0::2, 1::2] + v_plane[1::2, 0::2] + v_plane[1::2, 1::2]) * 0.25

    uv_plane = np.empty((height // 2, width), dtype=np.uint8)
    uv_plane[:, 0::2] = np.clip(u_sub, 0.0, 255.0).astype(np.uint8)
    uv_plane[:, 1::2] = np.clip(v_sub, 0.0, 255.0).astype(np.uint8)

    return y_plane.tobytes() + uv_plane.tobytes()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help',
                      help='Show this help message and exit.')
    args.add_argument('-i', '--input_name', type=str, required=True,
                      help='Path to the input file. Use a .nv12 file together with --nv12 for raw NV12 mode.')
    args.add_argument('-o', '--output_dir', type=str, default="output_files/",
                      help='Directory for saving the output file.')
    args.add_argument('-ih', '--input_height', type=int, default=512,
                      help='Optional. Height of input (pixels, must be even for NV12).')
    args.add_argument('-iw', '--input_width', type=int, default=512,
                      help='Optional. Width of input (pixels, must be even for NV12).')
    args.add_argument('-mt', '--model_type', type=str, default="wdsr",
                      help='Optional. Type of model.')
    args.add_argument('-d', '--device', type=str, default="GPU",
                      help='Optional. Target device: CPU, GPU or NPU. Default: GPU.')
    args.add_argument('-mp', '--model_precision', type=str, default="FP16",
                      help='Optional. Precision of model.')
    args.add_argument('-s', '--scale', type=int, default=2,
                      help='Optional. Scale factor for super-resolution.')
    args.add_argument('-ds', '--do_static', default=True, action=argparse.BooleanOptionalAction,
                      help='Optional. Use a static-shape model (default: True).')
    args.add_argument('--nv12', default=False, action=argparse.BooleanOptionalAction,
                      help='Optional. Process input as a raw NV12 binary video file.')
    args.add_argument('--num_frames', type=int, default=None,
                      help='Optional. Maximum number of NV12 frames to process (NV12 mode only).')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# NV12 inference path
# ---------------------------------------------------------------------------

def run_nv12(args, INPUT_NAME, ov_model_path, output_file_path):
    input_width  = args.input_width
    input_height = args.input_height

    if input_width % 2 != 0 or input_height % 2 != 0:
        raise ValueError("NV12 mode requires even width and height.")

    input_nv12_frame_size = input_width * input_height * 3 // 2

    if not INPUT_NAME.is_file():
        raise FileNotFoundError(f"Input file not found: {INPUT_NAME}")

    file_size    = INPUT_NAME.stat().st_size
    total_frames = file_size // input_nv12_frame_size
    if total_frames == 0:
        raise ValueError(
            f"File size ({file_size} bytes) is smaller than one NV12 frame "
            f"({input_nv12_frame_size} bytes for {input_width}x{input_height})"
        )

    frames_to_process = total_frames if args.num_frames is None else min(args.num_frames, total_frames)

    log.info("NV12 mode: %dx%d, %d frames to process.", input_width, input_height, frames_to_process)

    core  = ov.Core()
    model = core.read_model(ov_model_path)

    assert len(model.inputs)  == 1, "Sample supports models with 1 input only"
    assert len(model.outputs) == 1, "Sample supports models with 1 output only"

    input_port  = model.input()
    output_port = model.output()
    input_tensor_name  = input_port.get_any_name()
    output_tensor_name = output_port.get_any_name()
    model_input_layout = get_model_input_layout(input_port)

    if input_port.partial_shape.is_dynamic:
        reshaped = build_reshaped_input_shape(input_port, model_input_layout, input_height, input_width)
        log.info("Reshaping dynamic model input to: %s", reshaped)
        model.reshape({input_tensor_name: reshaped})
        input_port         = model.input()
        output_port        = model.output()
        input_tensor_name  = input_port.get_any_name()
        output_tensor_name = output_port.get_any_name()
        model_input_layout = get_model_input_layout(input_port)

    ppp        = PrePostProcessor(model)
    input_info = ppp.input(input_tensor_name)
    input_info.tensor() \
        .set_element_type(ov.Type.u8) \
        .set_color_format(ColorFormat.NV12_SINGLE_PLANE) \
        .set_spatial_static_shape(input_height, input_width)
    input_info.preprocess() \
        .convert_element_type(ov.Type.f32) \
        .convert_color(ColorFormat.BGR) \
        .resize(ResizeAlgorithm.RESIZE_LINEAR)
    input_info.model().set_layout(ov.Layout(model_input_layout))
    model = ppp.build()

    compiled_model = core.compile_model(model, args.device)
    infer_request  = compiled_model.create_infer_request()

    output_file_path += ".nv12"
    output_width = output_height = None

    with open(INPUT_NAME, "rb") as in_f, open(output_file_path, "wb") as out_f:
        processed = 0
        for frame_idx in range(frames_to_process):
            raw_data = in_f.read(input_nv12_frame_size)
            if len(raw_data) < input_nv12_frame_size:
                log.warning("Incomplete frame at index %d, stopping.", frame_idx)
                break

            nv12_array  = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                (1, input_height * 3 // 2, input_width, 1)
            )
            nv12_tensor = ov.Tensor(nv12_array)

            infer_request.set_tensor(input_tensor_name, nv12_tensor)
            infer_request.infer()

            output_data      = np.array(infer_request.get_tensor(output_tensor_name).data)
            output_image_bgr = tensor_to_hwc_image(output_data)
            output_height, output_width = output_image_bgr.shape[:2]

            out_f.write(bgr_to_nv12(output_image_bgr))
            processed += 1
            log.info("Frame %d: %dx%d -> %dx%d", frame_idx, input_width, input_height, output_width, output_height)

    if output_width is None or output_height is None:
        raise RuntimeError("No output frames were produced.")

    output_nv12_frame_size = output_width * output_height * 3 // 2
    log.info(
        "Processed %d frames. Output frame size: %dx%d (%d bytes/frame).",
        processed, output_width, output_height, output_nv12_frame_size,
    )
    log.info("Result saved to '%s'.", output_file_path)


# ---------------------------------------------------------------------------
# Standard (OpenCV) inference path
# ---------------------------------------------------------------------------

def run_standard(args, INPUT_NAME, ov_model_path, output_file_path):
    HEIGHT = args.input_height
    WIDTH  = args.input_width

    core           = ov.Core()
    model          = core.read_model(ov_model_path)
    compiled_model = core.compile_model(model, args.device)

    cap = cv2.VideoCapture(str(INPUT_NAME))
    if cap.isOpened():
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            VIDEO_FOURCC  = int(cap.get(cv2.CAP_PROP_FOURCC))
            VIDEO_FPS     = cap.get(cv2.CAP_PROP_FPS)
            VIDEO_WIDTH   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  * args.scale)
            VIDEO_HEIGHT  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.scale)

            OUTPUT_VIDEO = cv2.VideoWriter(
                output_file_path, VIDEO_FOURCC, VIDEO_FPS,
                (VIDEO_WIDTH, VIDEO_HEIGHT), isColor=True,
            )
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    if args.do_static:
                        frame = cv2.resize(frame, (WIDTH, HEIGHT))
                    result = compiled_model(np.expand_dims(np.transpose(frame, (2, 0, 1)), 0))[0]
                    result_frame = np.clip(np.transpose(result[0] * 255, (1, 2, 0)), 0, 255).astype(np.uint8)
                    OUTPUT_VIDEO.write(result_frame)
                else:
                    break
            OUTPUT_VIDEO.release()
        else:
            img = cv2.imread(str(INPUT_NAME))
            if args.do_static:
                img = cv2.resize(img, (WIDTH, HEIGHT))
            result       = compiled_model(np.expand_dims(np.transpose(img, (2, 0, 1)), 0))[0]
            result_image = np.transpose(result[0] * 255, (1, 2, 0))
            cv2.imwrite(output_file_path, result_image)
    cap.release()
    log.info("Result saved to '%s'.", output_file_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    INPUT_NAME      = Path(args.input_name)
    MODEL_TYPE      = args.model_type
    MODEL_PRECISION = args.model_precision
    SCALE           = args.scale

    output_stem = args.output_dir + INPUT_NAME.stem

    is_nv12 = args.nv12 or INPUT_NAME.suffix.lower() == ".nv12"

    if args.do_static:
        log.info("Using static %s model...", MODEL_PRECISION)
        ov_model_path    = "./ov_models/{}/{}_{}x_{}x{}.xml".format(
            MODEL_PRECISION, MODEL_TYPE, SCALE, args.input_height, args.input_width)
        output_file_path = "{}_{}_{}x_{}x{}_{}".format(
            output_stem, MODEL_TYPE, SCALE,
            args.input_height, args.input_width, MODEL_PRECISION,
        ) if is_nv12 else "{}_{}_{}x_{}x{}_{}{}".format(
            output_stem, MODEL_TYPE, SCALE,
            args.input_height, args.input_width, MODEL_PRECISION, INPUT_NAME.suffix,
        )
    else:
        log.info("Using dynamic %s model...", MODEL_PRECISION)
        ov_model_path    = "./ov_models/{}/{}_{}x_dyn.xml".format(MODEL_PRECISION, MODEL_TYPE, SCALE)
        output_file_path = "{}_{}_{}x_dyn_{}".format(output_stem, MODEL_TYPE, SCALE, MODEL_PRECISION) \
                           if is_nv12 else \
                           "{}_{}_{}x_dyn_{}{}".format(output_stem, MODEL_TYPE, SCALE, MODEL_PRECISION, INPUT_NAME.suffix)

    log.info("Output path: %s", output_file_path)

    if is_nv12:
        run_nv12(args, INPUT_NAME, ov_model_path, output_file_path)
    else:
        run_standard(args, INPUT_NAME, ov_model_path, output_file_path)


if __name__ == '__main__':
    sys.exit(main())