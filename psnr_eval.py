#!/usr/bin/env python3
"""
PSNR Evaluation Script for NV12 Format Videos

Evaluates Peak Signal-to-Noise Ratio (PSNR) values for NV12 format videos
by comparing two video files frame-by-frame.
"""

import argparse
import csv
import json
import sys
import numpy as np


class NV12FrameReader:
    """Reader for NV12 format video files."""
    
    def __init__(self, filepath, width, height):
        self.filepath = filepath
        self.width = width
        self.height = height
        self.frame_size = width * height + (width * height // 2)  # Y plane + UV plane
        self.file = None
        
    def open(self):
        """Open the file for reading."""
        self.file = open(self.filepath, 'rb')
        
    def close(self):
        """Close the file."""
        if self.file:
            self.file.close()
            
    def read_frame(self, frame_index):
        """Read a specific frame and return Y, U, V planes separately."""
        self.file.seek(frame_index * self.frame_size)
        frame_data = self.file.read(self.frame_size)
        
        if len(frame_data) < self.frame_size:
            return None
        
        # NV12: Y plane (full resolution), then UV plane (half resolution, interleaved)
        y_plane = np.frombuffer(frame_data, dtype=np.uint8, count=self.width*self.height)
        y_plane = y_plane.reshape(self.height, self.width)
        
        # UV data: interleaved U and V at half resolution
        uv_data = np.frombuffer(frame_data[self.width*self.height:], dtype=np.uint8)
        u_plane = uv_data[0::2].reshape(self.height//2, self.width//2)
        v_plane = uv_data[1::2].reshape(self.height//2, self.width//2)
        
        return y_plane, u_plane, v_plane
    
    def get_frame_count(self):
        """Get total number of frames in file."""
        if not self.file:
            return 0
        self.file.seek(0, 2)  # Seek to end
        file_size = self.file.tell()
        return file_size // self.frame_size


def calculate_psnr(reference_plane, test_plane):
    """
    Calculate PSNR for a single plane.
    
    Args:
        reference_plane: numpy array (8-bit, 0-255)
        test_plane: numpy array (8-bit, 0-255)
    
    Returns:
        PSNR value in dB (float), or 100.0 if planes are identical
    """
    # Ensure same shape
    if reference_plane.shape != test_plane.shape:
        raise ValueError(f"Shape mismatch: {reference_plane.shape} vs {test_plane.shape}")
    
    # Convert to float for MSE calculation
    ref = reference_plane.astype(np.float64)
    test = test_plane.astype(np.float64)
    
    # Calculate MSE
    mse = np.mean((ref - test) ** 2)
    
    # If MSE is 0 (identical), return 100.0 dB
    if mse == 0:
        return 100.0
    
    # PSNR = 20 * log10(MAX_PIXEL_VALUE / sqrt(MSE))
    # MAX_PIXEL_VALUE = 255 for 8-bit video
    psnr = 20.0 * np.log10(255.0 / np.sqrt(mse))
    return psnr


def compare_frames(ref_y, ref_u, ref_v, test_y, test_u, test_v):
    """
    Compare two NV12 frames and return PSNR for each component.
    
    Returns:
        dict: {'psnr_y': float, 'psnr_u': float, 'psnr_v': float}
    """
    return {
        'psnr_y': calculate_psnr(ref_y, test_y),
        'psnr_u': calculate_psnr(ref_u, test_u),
        'psnr_v': calculate_psnr(ref_v, test_v)
    }


def write_csv_output(output_path, frame_results, aggregate_stats):
    """
    Write per-frame and aggregate results to CSV.
    
    Args:
        output_path: Path to output CSV file
        frame_results: list of dicts with {'frame': int, 'psnr_y': float, 'psnr_u': float, 'psnr_v': float}
        aggregate_stats: dict with aggregate statistics
    """
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['Frame', 'PSNR_Y', 'PSNR_U', 'PSNR_V'])
        
        # Per-frame results
        for result in frame_results:
            writer.writerow([
                result['frame'],
                f"{result['psnr_y']:.4f}",
                f"{result['psnr_u']:.4f}",
                f"{result['psnr_v']:.4f}"
            ])
        
        # Blank line
        writer.writerow([])
        
        # Aggregate statistics
        writer.writerow(['Aggregate Statistics', '', '', ''])
        writer.writerow(['Metric', 'Y', 'U', 'V'])
        writer.writerow(['Mean PSNR', f"{aggregate_stats['mean_y']:.4f}", f"{aggregate_stats['mean_u']:.4f}", f"{aggregate_stats['mean_v']:.4f}"])
        writer.writerow(['Min PSNR', f"{aggregate_stats['min_y']:.4f}", f"{aggregate_stats['min_u']:.4f}", f"{aggregate_stats['min_v']:.4f}"])
        writer.writerow(['Max PSNR', f"{aggregate_stats['max_y']:.4f}", f"{aggregate_stats['max_u']:.4f}", f"{aggregate_stats['max_v']:.4f}"])


def write_json_output(output_path, frame_results, aggregate_stats):
    """
    Write per-frame and aggregate results to JSON.
    
    Args:
        output_path: Path to output JSON file
        frame_results: list of dicts with frame PSNR values
        aggregate_stats: dict with aggregate statistics
    """
    output = {
        'frames': frame_results,
        'aggregate_statistics': aggregate_stats
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


def calculate_aggregate_stats(frame_results):
    """
    Calculate aggregate statistics from per-frame results.
    
    Returns:
        dict: {'mean_y': float, 'min_y': float, 'max_y': float, 'mean_u': float, ...}
    """
    psnr_y_values = [r['psnr_y'] for r in frame_results]
    psnr_u_values = [r['psnr_u'] for r in frame_results]
    psnr_v_values = [r['psnr_v'] for r in frame_results]
    
    return {
        'mean_y': np.mean(psnr_y_values),
        'min_y': np.min(psnr_y_values),
        'max_y': np.max(psnr_y_values),
        'mean_u': np.mean(psnr_u_values),
        'min_u': np.min(psnr_u_values),
        'max_u': np.max(psnr_u_values),
        'mean_v': np.mean(psnr_v_values),
        'min_v': np.min(psnr_v_values),
        'max_v': np.max(psnr_v_values),
        'total_frames': len(frame_results)
    }


def process_videos(ref_path, test_path, width, height, start_frame=0, end_frame=None):
    """
    Process two NV12 videos and calculate PSNR for each frame.
    
    Returns:
        (frame_results, aggregate_stats)
    """
    ref_reader = NV12FrameReader(ref_path, width, height)
    test_reader = NV12FrameReader(test_path, width, height)
    
    ref_reader.open()
    test_reader.open()
    
    try:
        frame_results = []
        ref_frame_count = ref_reader.get_frame_count()
        test_frame_count = test_reader.get_frame_count()
        
        if ref_frame_count != test_frame_count:
            print(f"Warning: Frame count mismatch - reference: {ref_frame_count}, test: {test_frame_count}")
        
        # Determine end frame
        max_frame = min(ref_frame_count, test_frame_count)
        if end_frame is None:
            end_frame = max_frame
        else:
            end_frame = min(end_frame, max_frame)
        
        # Process frames
        for frame_idx in range(start_frame, end_frame):
            ref_planes = ref_reader.read_frame(frame_idx)
            test_planes = test_reader.read_frame(frame_idx)
            
            if ref_planes is None or test_planes is None:
                print(f"Warning: Could not read frame {frame_idx}")
                continue
            
            ref_y, ref_u, ref_v = ref_planes
            test_y, test_u, test_v = test_planes
            
            psnr_dict = compare_frames(ref_y, ref_u, ref_v, test_y, test_u, test_v)
            psnr_dict['frame'] = frame_idx
            frame_results.append(psnr_dict)
        
        aggregate_stats = calculate_aggregate_stats(frame_results)
        return frame_results, aggregate_stats
        
    finally:
        ref_reader.close()
        test_reader.close()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate PSNR values for NV12 format videos'
    )
    parser.add_argument('reference_video', help='Path to reference NV12 video')
    parser.add_argument('test_video', help='Path to test NV12 video')
    parser.add_argument('--width', type=int, default=1920, help='Video width (default: 1920)')
    parser.add_argument('--height', type=int, default=1080, help='Video height (default: 1080)')
    parser.add_argument('--start-frame', type=int, default=0, help='Start frame index')
    parser.add_argument('--end-frame', type=int, default=None, help='End frame index')
    parser.add_argument('--output-csv', help='Output CSV file path')
    parser.add_argument('--output-json', help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Validate files exist
    try:
        with open(args.reference_video, 'rb'):
            pass
    except FileNotFoundError:
        print(f"Error: Reference video not found: {args.reference_video}")
        sys.exit(1)
    
    try:
        with open(args.test_video, 'rb'):
            pass
    except FileNotFoundError:
        print(f"Error: Test video not found: {args.test_video}")
        sys.exit(1)
    
    print(f"Processing videos: {args.reference_video} vs {args.test_video}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frame range: {args.start_frame} to {args.end_frame or 'end'}")
    print()
    
    # Process videos
    frame_results, aggregate_stats = process_videos(
        args.reference_video,
        args.test_video,
        args.width,
        args.height,
        args.start_frame,
        args.end_frame
    )
    
    # Print console output
    print("Per-Frame PSNR Results:")
    print(f"{'Frame':<8} {'PSNR_Y':<12} {'PSNR_U':<12} {'PSNR_V':<12}")
    print("-" * 44)
    for result in frame_results:
        print(f"{result['frame']:<8} {result['psnr_y']:<12.4f} {result['psnr_u']:<12.4f} {result['psnr_v']:<12.4f}")
    
    print("\n" + "="*44)
    print("Aggregate Statistics:")
    print(f"{'Metric':<15} {'Y':<12} {'U':<12} {'V':<12}")
    print("-" * 51)
    print(f"{'Mean PSNR':<15} {aggregate_stats['mean_y']:<12.4f} {aggregate_stats['mean_u']:<12.4f} {aggregate_stats['mean_v']:<12.4f}")
    print(f"{'Min PSNR':<15} {aggregate_stats['min_y']:<12.4f} {aggregate_stats['min_u']:<12.4f} {aggregate_stats['min_v']:<12.4f}")
    print(f"{'Max PSNR':<15} {aggregate_stats['max_y']:<12.4f} {aggregate_stats['max_u']:<12.4f} {aggregate_stats['max_v']:<12.4f}")
    print(f"{'Total Frames':<15} {aggregate_stats['total_frames']:<12}")
    
    # Write CSV output if requested
    if args.output_csv:
        write_csv_output(args.output_csv, frame_results, aggregate_stats)
        print(f"\nCSV output written to: {args.output_csv}")
    
    # Write JSON output if requested
    if args.output_json:
        write_json_output(args.output_json, frame_results, aggregate_stats)
        print(f"JSON output written to: {args.output_json}")


if __name__ == '__main__':
    main()
