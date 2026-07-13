"""
This script takes an existing WDSR OpenVINO IR model and bakes the
NV12 -> BGR preprocessing (color conversion + resize + scale) directly into it.

It does NOT convert from a PyTorch checkpoint. The source model must already be an
OpenVINO IR (.xml/.bin) on disk (e.g. produced by ov_convert.py). The exported
model accepts a raw NV12 (single plane) uint8 tensor as input, so no runtime
PrePostProcessor setup is required in ov_infer.py. This mirrors the runtime NV12
PPP pipeline in ov_infer.py:run_nv12().
"""
import argparse
import logging
import os
import sys

import openvino as ov
from openvino.preprocess import ColorFormat, PrePostProcessor, ResizeAlgorithm

logging.basicConfig(
    format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


def infer_layout_from_static_shape(shape):
  """Infer a common 4D image tensor layout from a fully static shape."""
  if len(shape) != 4:
    raise ValueError("Only 4D image tensors are supported, got shape: {}".format(shape))
  if shape[1] in (1, 3):
    return "NCHW"
  if shape[3] in (1, 3):
    return "NHWC"
  return "NCHW"


def build_reshaped_input_shape(port, layout, input_height, input_width):
  """Build a concrete 4D input shape for a (possibly dynamic) image model."""
  partial_shape = port.partial_shape
  if len(partial_shape) != 4:
    raise ValueError("Only 4D input tensors are supported, got: {}".format(partial_shape))

  try:
    static_shape = list(port.shape)
  except RuntimeError:
    static_shape = [None] * 4

  def static_or_default(index, default_value):
    value = static_shape[index]
    return default_value if value is None else int(value)

  if layout == "NHWC":
    return [static_or_default(0, 1), input_height, input_width, static_or_default(3, 3)]
  return [static_or_default(0, 1), static_or_default(1, 3), input_height, input_width]


def parse_args():
  parser = argparse.ArgumentParser(add_help=False)
  args = parser.add_argument_group('Options')
  args.add_argument('-h', '--help', action='help',
                    help='Show this help message and exit.')
  args.add_argument('-m', '--model_path', type=str, default=None,
                    help='Optional. Path to the source OpenVINO IR (.xml). If not set, it is '
                         'derived from --model_type/--scale/--model_precision as the dynamic model.')
  args.add_argument('-mt', '--model_type', type=str, default="wdsr",
                    help='Optional. Type of model.')
  args.add_argument('-mp', '--model_precision', type=str, default="FP16",
                    help='Optional. Precision of the source IR (FP16 or INT8).')
  args.add_argument('-s', '--scale', type=int, default=2,
                    help='Optional. Scale factor for super-resolution.')
  args.add_argument('-ih', '--input_height', type=int, default=512,
                    help='Optional. Model input height / resize target (pixels).')
  args.add_argument('-iw', '--input_width', type=int, default=512,
                    help='Optional. Model input width / resize target (pixels).')
  args.add_argument('-ivh', '--input_video_height', type=int, default=None,
                    help='Optional. Height of raw NV12 input frames (pixels, must be even). '
                         'Defaults to -ih if not set.')
  args.add_argument('-ivw', '--input_video_width', type=int, default=None,
                    help='Optional. Width of raw NV12 input frames (pixels, must be even). '
                         'Defaults to -iw if not set.')
  args.add_argument('-o', '--output_path', type=str, default=None,
                    help='Optional. Output IR path (.xml). Derived from the naming convention '
                         'if not set.')
  return parser.parse_args()


def main():
  args = parse_args()

  # Model input dimensions (resize target for the baked NV12 pipeline).
  model_height = args.input_height
  model_width = args.input_width

  # Raw NV12 frame dimensions (input tensor spatial shape). Fall back to -ih/-iw.
  video_height = args.input_video_height if args.input_video_height is not None else model_height
  video_width = args.input_video_width if args.input_video_width is not None else model_width

  if video_height % 2 != 0 or video_width % 2 != 0:
    raise ValueError(
        "NV12 requires even video frame height and width, got {}x{}.".format(
            video_height, video_width))

  # Resolve the source IR path.
  if args.model_path is not None:
    src_model_path = args.model_path
  else:
    src_model_path = "./ov_models/{}/{}_{}x_dyn.xml".format(
        args.model_precision, args.model_type, args.scale)

  if not os.path.isfile(src_model_path):
    raise FileNotFoundError("Source IR not found: {}".format(src_model_path))

  # Resolve the output IR path.
  if args.output_path is not None:
    out_model_path = args.output_path
  else:
    out_model_path = "./ov_models/{}/{}_{}x_nv12_{}x{}_{}x{}.xml".format(
        args.model_precision, args.model_type, args.scale,
        video_height, video_width, model_height, model_width)

  os.makedirs(os.path.dirname(out_model_path), exist_ok=True)

  log.info("Reading source IR: %s", src_model_path)
  core = ov.Core()
  model = core.read_model(src_model_path)

  assert len(model.inputs) == 1, "Only models with 1 input are supported."
  assert len(model.outputs) == 1, "Only models with 1 output are supported."

  input_port = model.input()
  input_tensor_name = input_port.get_any_name()

  # Resolve layout and force a static model input shape (HxW resize target) so the
  # NV12 resize has a concrete destination size to bake in.
  try:
    static_shape = list(input_port.shape)
    layout = infer_layout_from_static_shape(static_shape)
  except RuntimeError:
    layout = "NCHW"

  reshaped = build_reshaped_input_shape(input_port, layout, model_height, model_width)
  log.info("Reshaping model input to static shape: %s (layout %s)", reshaped, layout)
  model.reshape({input_tensor_name: reshaped})

  input_port = model.input()
  input_tensor_name = input_port.get_any_name()

  # Bake in the NV12 preprocessing.
  prep = PrePostProcessor(model)
  input_info = prep.input(0)
  # Declare the runtime input tensor as a raw NV12 single-plane uint8 frame.
  input_info.tensor() \
      .set_element_type(ov.Type.u8) \
      .set_color_format(ColorFormat.NV12_SINGLE_PLANE) \
      .set_spatial_static_shape(video_height, video_width)
  # NV12 -> BGR, resize to model dimensions, then normalize to [0, 1].
  input_info.preprocess() \
      .convert_element_type(ov.Type.f32) \
      .convert_color(ColorFormat.BGR) \
      .resize(ResizeAlgorithm.RESIZE_LINEAR) \
      .scale([255, 255, 255])
  input_info.model().set_layout(ov.Layout(layout))
  model = prep.build()

  ov.save_model(model, out_model_path)
  log.info("NV12 model saved to %s", out_model_path)


if __name__ == '__main__':
  sys.exit(main())
