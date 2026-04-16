#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
NV12 Video Super Resolution Sample (Python)

Based on the OpenVINO C++ hello_nv12_input_classification sample,
extended to process raw NV12 video files frame-by-frame with a super-resolution model
and save the enhanced output back as a raw NV12 video file.

Usage:
    python hello_nv12_video_classification.py <path_to_model> <path_to_input_nv12_video> <input_WIDTHxHEIGHT> <path_to_output_nv12_video> <device_name> [num_frames]

Arguments:
    path_to_model            - Path to an OpenVINO IR model (.xml) or ONNX model
    path_to_input_nv12_video - Path to a raw NV12 video file
    input_WIDTHxHEIGHT       - Input frame dimensions, e.g. 320x240 (width and height must be even)
    path_to_output_nv12_video - Path to the output raw NV12 video file
    device_name              - Target device, e.g. CPU, GPU
    num_frames               - (Optional) Number of frames to process. Default: all frames in the file.

Example:
    python hello_nv12_video_classification.py model.xml input.nv12 320x240 output_sr.nv12 CPU 10
"""

import sys
import os
import numpy as np
import openvino as ov
from openvino.preprocess import ColorFormat, PrePostProcessor, ResizeAlgorithm


def parse_image_size(size_string: str):
    """Parse image size from 'WIDTHxHEIGHT' string."""
    if "x" not in size_string:
        raise ValueError(f"Incorrect format of image size parameter, expected WIDTHxHEIGHT, actual: {size_string}")

    parts = size_string.split("x")
    if len(parts) != 2:
        raise ValueError(f"Incorrect format of image size parameter, expected WIDTHxHEIGHT, actual: {size_string}")

    width, height = int(parts[0]), int(parts[1])

    if width == 0 or height == 0:
        raise ValueError("Width and height must not be equal to 0")
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError("Unsupported image size, width and height must be even numbers")

    return width, height


def infer_layout_from_static_shape(shape):
    """Infer a common 4D image tensor layout from a fully static shape."""
    if len(shape) != 4:
        raise ValueError(f"Only 4D image tensors are supported, got shape: {shape}")

    if shape[1] in (1, 3):
        return "NCHW"
    if shape[3] in (1, 3):
        return "NHWC"
    return "NCHW"


def get_port_shape_description(port) -> str:
    """Return a printable shape description for static or dynamic ports."""
    partial_shape = getattr(port, "partial_shape", None)
    if partial_shape is not None:
        return str(partial_shape)
    return str(port.shape)


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
    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]

    y_plane = 0.114 * blue + 0.587 * green + 0.299 * red
    u_plane = 128.0 - 0.081312 * red - 0.418688 * green + 0.5 * blue
    v_plane = 128.0 + 0.5 * red - 0.331264 * green - 0.168736 * blue

    y_plane = np.clip(y_plane, 0.0, 255.0).astype(np.uint8)
    u_plane = np.clip(u_plane, 0.0, 255.0)
    v_plane = np.clip(v_plane, 0.0, 255.0)

    u_subsampled = (
        u_plane[0::2, 0::2]
        + u_plane[0::2, 1::2]
        + u_plane[1::2, 0::2]
        + u_plane[1::2, 1::2]
    ) * 0.25
    v_subsampled = (
        v_plane[0::2, 0::2]
        + v_plane[0::2, 1::2]
        + v_plane[1::2, 0::2]
        + v_plane[1::2, 1::2]
    ) * 0.25

    uv_plane = np.empty((height // 2, width), dtype=np.uint8)
    uv_plane[:, 0::2] = np.clip(u_subsampled, 0.0, 255.0).astype(np.uint8)
    uv_plane[:, 1::2] = np.clip(v_subsampled, 0.0, 255.0).astype(np.uint8)

    return y_plane.tobytes() + uv_plane.tobytes()


def main():
    # -------- Parse and validate input arguments --------
    if len(sys.argv) < 6 or len(sys.argv) > 7:
        print(
            f"Usage: {sys.argv[0]} <path_to_model> <path_to_input_nv12_video> "
            f"<input_WIDTHxHEIGHT> <path_to_output_nv12_video> <device_name> [num_frames]"
        )
        return 1

    model_path = sys.argv[1]
    input_video_path = sys.argv[2]
    input_width, input_height = parse_image_size(sys.argv[3])
    output_video_path = sys.argv[4]
    device_name = sys.argv[5]
    max_frames = int(sys.argv[6]) if len(sys.argv) == 7 else None

    # NV12 frame size: width * height * 3 / 2
    input_nv12_frame_size = input_width * input_height * 3 // 2

    # Validate video file
    if not os.path.isfile(input_video_path):
        raise FileNotFoundError(f"Video file not found: {input_video_path}")

    file_size = os.path.getsize(input_video_path)
    total_frames = file_size // input_nv12_frame_size
    if total_frames == 0:
        raise ValueError(
            f"File size ({file_size} bytes) is smaller than one NV12 frame "
            f"({input_nv12_frame_size} bytes for {input_width}x{input_height})"
        )

    frames_to_process = total_frames if max_frames is None else min(max_frames, total_frames)

    print(f"OpenVINO Runtime version: {ov.get_version()}")
    print(f"Input video: {input_video_path}")
    print(f"Output video: {output_video_path}")
    print(f"Input frame size: {input_width}x{input_height}, NV12 frame bytes: {input_nv12_frame_size}")
    print(f"Total frames in file: {total_frames}, frames to process: {frames_to_process}")

    # -------- Step 1. Initialize OpenVINO Runtime Core --------
    core = ov.Core()

    # -------- Step 2. Read the model --------
    print(f"Loading model: {model_path}")
    model = core.read_model(model_path)

    assert len(model.inputs) == 1, "Sample supports models with 1 input only"
    assert len(model.outputs) == 1, "Sample supports models with 1 output only"

    input_port = model.input()
    output_port = model.output()
    input_tensor_name = input_port.get_any_name()
    output_tensor_name = output_port.get_any_name()
    model_input_shape = get_port_shape_description(input_port)
    model_input_layout = get_model_input_layout(input_port)

    if input_port.partial_shape.is_dynamic:
        reshaped_input_shape = build_reshaped_input_shape(
            input_port,
            model_input_layout,
            input_height,
            input_width,
        )
        print(f"Reshaping dynamic model input to: {reshaped_input_shape}")
        model.reshape({input_tensor_name: reshaped_input_shape})
        input_port = model.input()
        output_port = model.output()
        input_tensor_name = input_port.get_any_name()
        output_tensor_name = output_port.get_any_name()
        model_input_shape = get_port_shape_description(input_port)

    print(
        f"  Input: {input_tensor_name}, shape={model_input_shape}, layout={model_input_layout}; "
        f"Output: {output_tensor_name}"
    )

    # -------- Step 3. Configure preprocessing --------
    ppp = PrePostProcessor(model)

    # 1) Set input tensor properties
    #    - element type: u8
    #    - color format: NV12 (single plane, Y and UV interleaved)
    #    - spatial dimensions matching the raw video frame size
    input_info = ppp.input(input_tensor_name)
    input_info.tensor() \
        .set_element_type(ov.Type.u8) \
        .set_color_format(ColorFormat.NV12_SINGLE_PLANE) \
        .set_spatial_static_shape(input_height, input_width)

    # 2) Pre-processing steps:
    #    a) Convert to float for more accurate color conversion
    #    b) Convert NV12 to BGR (change to RGB if your model expects RGB)
    #    c) Resize from input frame dimensions to model's expected dimensions
    input_info.preprocess() \
        .convert_element_type(ov.Type.f32) \
        .convert_color(ColorFormat.BGR) \
        .resize(ResizeAlgorithm.RESIZE_LINEAR)

    # 3) Set model input layout
    input_info.model().set_layout(ov.Layout(model_input_layout))

    # 4) Build the preprocessing pipeline into the model
    model = ppp.build()

    # -------- Step 4. Compile the model --------
    print(f"Compiling model for device: {device_name}")
    compiled_model = core.compile_model(model, device_name)

    # -------- Step 5. Create infer request --------
    infer_request = compiled_model.create_infer_request()

    output_width = None
    output_height = None

    # -------- Step 6-9. Read frames, infer, and save results --------
    with open(input_video_path, "rb") as input_file, open(output_video_path, "wb") as output_file:
        processed_frames = 0
        for frame_idx in range(frames_to_process):
            # Read one NV12 frame
            raw_data = input_file.read(input_nv12_frame_size)
            if len(raw_data) < input_nv12_frame_size:
                print(f"Warning: Incomplete frame at index {frame_idx}, stopping.")
                break

            # Create input tensor: NV12 single-plane shape is [1, H*3/2, W, 1]
            nv12_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(
                (1, input_height * 3 // 2, input_width, 1)
            )
            nv12_tensor = ov.Tensor(nv12_array)

            # Set input tensor and run inference
            infer_request.set_tensor(input_tensor_name, nv12_tensor)
            infer_request.infer()

            # Get output, convert it to an image, and save as NV12
            output_tensor = infer_request.get_tensor(output_tensor_name)
            output_data = np.array(output_tensor.data)
            output_image_bgr = tensor_to_hwc_image(output_data)
            output_height, output_width = output_image_bgr.shape[:2]

            output_file.write(bgr_to_nv12(output_image_bgr))
            processed_frames += 1

            print(f"Processed frame {frame_idx}: {input_width}x{input_height} -> {output_width}x{output_height}")

    if output_width is None or output_height is None:
        raise RuntimeError("No output frames were produced")

    output_nv12_frame_size = output_width * output_height * 3 // 2
    print(
        f"\nProcessed {processed_frames} frames. "
        f"Output frame size: {output_width}x{output_height}, NV12 frame bytes: {output_nv12_frame_size}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
