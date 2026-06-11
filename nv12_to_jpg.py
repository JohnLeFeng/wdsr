#!/usr/bin/env python3
"""
NV12 to JPG Frame Extraction Script

Extracts a specific frame from raw NV12 video data, converts to RGB using
ITU-R BT.601 standard, and saves as JPG.

Usage:
    python nv12_to_jpg.py --input <file> --width <w> --height <h> --frame <n> [--output <jpg>] [--quality <q>]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import zoom


def parse_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract a frame from raw NV12 file and save as JPG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract frame 10 from 1920x1080 NV12
  python nv12_to_jpg.py --input video.nv12 --width 1920 --height 1080 --frame 10

  # With custom output and quality
  python nv12_to_jpg.py --input video.nv12 --width 1920 --height 1080 --frame 0 --output my_frame.jpg --quality 90
        """,
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to raw NV12 file",
    )
    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="Frame width in pixels (must be even)",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="Frame height in pixels (must be even)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        required=True,
        help="Frame number to extract (0-indexed)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JPG path (default: frame_<n>.jpg)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPG quality 1-100 (default: 95)",
    )

    return parser.parse_args()


def validate_args(args):
    """Validate command-line arguments. Raises ValueError on invalid input."""
    # Check width and height are even
    if args.width <= 0 or args.width % 2 != 0:
        raise ValueError(f"Width must be positive and even, got {args.width}")

    if args.height <= 0 or args.height % 2 != 0:
        raise ValueError(f"Height must be positive and even, got {args.height}")

    # Check frame number
    if args.frame < 0:
        raise ValueError(f"Frame number must be non-negative, got {args.frame}")

    # Check quality
    if not (1 <= args.quality <= 100):
        raise ValueError(f"Quality must be in range [1, 100], got {args.quality}")

    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        raise ValueError(f"Input file not found: {args.input}")

    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {args.input}")

    # Check file size is sufficient
    bytes_per_frame = args.width * args.height * 3 // 2
    expected_offset = args.frame * bytes_per_frame
    file_size = input_path.stat().st_size

    if expected_offset + bytes_per_frame > file_size:
        raise ValueError(
            f"File too small: frame {args.frame} requires {expected_offset + bytes_per_frame} bytes, "
            f"but file is only {file_size} bytes"
        )


def extract_nv12_frame(filepath, width, height, frame_num):
    """
    Extract Y, U, V planes from NV12 file.

    Returns:
        tuple: (Y, U, V) as numpy arrays with shapes:
            Y: (height, width)
            U: (height/2, width/2)
            V: (height/2, width/2)
    """
    bytes_per_frame = width * height * 3 // 2
    byte_offset = frame_num * bytes_per_frame

    with open(filepath, "rb") as f:
        f.seek(byte_offset)
        frame_data = f.read(bytes_per_frame)

    if len(frame_data) < bytes_per_frame:
        raise IOError(f"Failed to read {bytes_per_frame} bytes from {filepath}")

    # Extract Y plane
    y_size = width * height
    y_data = np.frombuffer(frame_data[:y_size], dtype=np.uint8)
    y = y_data.reshape((height, width))

    # Extract UV plane (interleaved UVUV...)
    uv_data = np.frombuffer(frame_data[y_size:], dtype=np.uint8)

    # De-interleave U and V
    u = uv_data[0::2].reshape((height // 2, width // 2))
    v = uv_data[1::2].reshape((height // 2, width // 2))

    return y, u, v


def upsample_uv(u, v, target_height, target_width):
    """
    Upsample U and V planes from half-resolution to full resolution.

    Args:
        u: U plane, shape (height/2, width/2)
        v: V plane, shape (height/2, width/2)
        target_height: Target height (full resolution)
        target_width: Target width (full resolution)

    Returns:
        tuple: (u_up, v_up) upsampled to (target_height, target_width)
    """
    # Zoom factor: from (H/2, W/2) to (H, W)
    scale_factor = (target_height / u.shape[0], target_width / u.shape[1])

    u_up = zoom(u, scale_factor, order=1)  # order=1 is bilinear interpolation
    v_up = zoom(v, scale_factor, order=1)

    return u_up.astype(np.float32), v_up.astype(np.float32)


def nv12_to_rgb(y, u, v):
    """
    Convert NV12 (YUV 4:2:0) to RGB using ITU-R BT.601 standard.

    Args:
        y: Y plane, shape (height, width)
        u: U plane, shape (height/2, width/2)
        v: V plane, shape (height/2, width/2)

    Returns:
        RGB array, shape (height, width, 3), dtype uint8
    """
    height, width = y.shape

    # Convert to float for calculations
    y = y.astype(np.float32)

    # Upsample U and V to full resolution
    u_up, v_up = upsample_uv(u, v, height, width)

    # ITU-R BT.601 conversion coefficients
    r = y + 1.402 * (v_up - 128.0)
    g = y - 0.344136 * (u_up - 128.0) - 0.714136 * (v_up - 128.0)
    b = y + 1.772 * (u_up - 128.0)

    # Clip to [0, 255]
    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)

    # Stack channels and convert to uint8
    rgb = np.stack([r, g, b], axis=2).astype(np.uint8)

    return rgb


def save_jpg(rgb_array, output_path, quality):
    """
    Save RGB array as JPG file.

    Args:
        rgb_array: RGB array, shape (height, width, 3), dtype uint8
        output_path: Path to output JPG file
        quality: JPG quality 1-100
    """
    image = Image.fromarray(rgb_array, mode="RGB")
    image.save(output_path, format="JPEG", quality=quality)


def main():
    """Main entry point."""
    try:
        args = parse_args()
        validate_args(args)

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            output_path = f"frame_{args.frame}.jpg"

        # Extract frame
        print(f"Extracting frame {args.frame} from {args.input}...")
        y, u, v = extract_nv12_frame(args.input, args.width, args.height, args.frame)

        # Convert to RGB
        print("Converting NV12 to RGB...")
        rgb = nv12_to_rgb(y, u, v)

        # Save as JPG
        print(f"Saving to {output_path} (quality={args.quality})...")
        save_jpg(rgb, output_path, args.quality)

        print(f"Success! Frame saved to {output_path}")
        return 0

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except IOError as e:
        print(f"IO Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
