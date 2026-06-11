# Wide Activation for Efficient Image and Video Super-Resolution

## OpenVINO Enabling

Please following below steps to enable WDSR model and run with OpenVINO.

### Checkpoints

Please download check below table and download checkpoint and then put into .

* Small models

    | Networks | Checkpoint |
    | - | - |
    | WDSR x2 | [Download](https://github.com/ychfan/wdsr/files/4176974/wdsr_x2.zip) |
    | WDSR x3 | [Download](https://github.com/ychfan/wdsr/files/4176981/wdsr_x3.zip) |
    | WDSR x4 | [Download](https://github.com/ychfan/wdsr/files/4176985/wdsr_x4.zip) |

* Large models

    | Networks | Checkpoint |
    | - | - | 
    | WDSR x2 | [Download](https://drive.google.com/file/d/10OsQD--qWZIBinFignAWwppHw5z9LPMI/view?usp=sharing) |
    | WDSR x3 | [Download](https://drive.google.com/file/d/10Yh0mI2825k69vChRZRGMsAC7C-M5hbk/view?usp=sharing) |
    | WDSR x4 | [Download](https://drive.google.com/file/d/10sYc5F63-o3eovtGCG5SSawk4otEHIxe/view?usp=sharing) |

### Convert model to IR

Please use below command to convert model, and then you can find the models under directory, `ov_models\FP16`.

* Get static model:

    ```py
    python ov_convert.py --scale 3 --ckpt checkpoints/wdsr_x3/epoch_30.pth -ih 512 -iw 512
    ```

* Get dynamic model:

    ```py
    python ov_convert.py --scale 3 --ckpt checkpoints/wdsr_x3/epoch_30.pth --no-do_static
    ```

> Notes: 
>   * Here is using scale 3 as instance.
>   * `ov_convert.py` is based on `trainer.py`.


### Quantization

Please use below command to quantize model from __FP16__ to __INT8__ precision, and then you can find the models under directory, `ov_models\INT8`. 

* Prepare dataset

    Please follow directory structure below to collect your own dataset and put the image into directory, `quantization_datasets\pics4q\img`.

    ```sh
    quantization_datasets
    └── pics4q
        └── img
            ├── aaa.jpg
            ├── bbb.jpg
            ├── ccc.jpg
            ├──
            ├── ...
            ├──
            └── xxx.jpg
    ```

* Static model:

    ```py
    python ov_quantize.py -s 3 -ih 512 -iw 512
    ```

* Dynamic model:

    ```py
    python ov_quantize.py -s 3 --no-do_static
    ```

> Notes: 
>   * Here is using scale 3 as instance.


### Run inference

Please use below command to run inference for FP16 and INT8 model. 

* Run FP16 model

    * Static model:

        ```py
        python ov_infer.py -s 3 -i input_imgs/input.jpg -ih 512 -iw 512
        ```

    * Dynamic model:

        ```py
        python ov_infer.py -s 3 --no-do_static -i input_imgs/input.jpg
        ```

* Run INT8 model

    * Static model:

        ```py
        python ov_infer.py -s 3 -i input_imgs/input.jpg -ih 512 -iw 512 -mp INT8
        ```

    * Dynamic model:

        ```py
        python ov_infer.py -s 3 --no-do_static -i input_imgs/input.jpg -mp INT8
        ```

> Notes: 
>   * Here is using scale 3 as instance.

#### NV12 video inference

`ov_infer.py` also supports raw NV12 binary video files. Pass `--nv12` (or use a `.nv12` input file) to activate NV12 mode. The output is written as a raw NV12 file.

> **Note:** OpenCV cannot decode raw NV12 streams. This mode reads raw bytes directly and uses OpenVINO's `PrePostProcessor` (PPP) pipeline for all frame processing — no OpenCV involvement. The PPP pipeline handles: NV12 → BGR color conversion, resize from the video frame dimensions to the model input dimensions, and element type conversion.

Two sets of dimensions are used in NV12 mode:

| Option | Purpose |
| - | - |
| `-ih` / `-iw` | Model input dimensions (static model selection or dynamic reshape target) |
| `-ivh` / `-ivw` | Raw NV12 input frame dimensions. Defaults to `-ih` / `-iw` when not specified. |

When `-ivh`/`-ivw` differ from `-ih`/`-iw`, the PPP resize step scales the decoded BGR frame to the model's expected input size before inference.

* Dynamic model, process all frames (video and model share the same resolution):

    ```py
    python ov_infer.py -s 3 --no-do_static -i input_imgs/input.nv12 -ivh 320 -ivw 480 -d CPU --nv12
    ```

* Dynamic model, video 1920×1080 input scaled to 512×512 model:

    ```py
    python ov_infer.py -s 3 --no-do_static -i input_imgs/input.nv12 -ivh 1080 -ivw 1920 -ih 512 -iw 512 -d CPU --nv12
    ```

* Dynamic model, process first 10 frames:

    ```py
    python ov_infer.py -s 3 --no-do_static -i input_imgs/input.nv12 -ivh 320 -ivw 480 -d CPU --nv12 --num_frames 10
    ```

* Static model, INT8 precision:

    ```py
    python ov_infer.py -s 3 -i input_imgs/input.nv12 -ih 320 -iw 480 -ivh 320 -ivw 480 -d CPU -mp INT8 --nv12
    ```

> Notes:
>   * `-ivh`/`-ivw` (video frame dimensions) must be even numbers.
>   * `--num_frames` limits how many frames are processed; omit it to process the entire file.
>   * Here is using scale 3 as instance.

#### Fully use NPU computation capability

Pass NPU config options via `core.compile_model` in `ov_infer.py` to run with 6 tiles on LNL:

```py
compiled_model = core.compile_model(model, "NPU",
    {"NPU_DPU_GROUPS": 6, "NPU_MAX_TILES": 6, "PERFORMANCE_HINT": "LATENCY"}
)
```

## Tools

### PSNR Evaluation Script for NV12 Video

Evaluates Peak Signal-to-Noise Ratio (PSNR) values for NV12 format videos by comparing two video files.

#### Usage

```bash
python psnr_eval.py <reference_video> <test_video> [--width W] [--height H] [--start-frame N] [--end-frame N] [--output-csv file.csv] [--output-json file.json]
```

#### Parameters

- `reference_video`: Path to reference NV12 video file
- `test_video`: Path to test NV12 video file
- `--width`: Video width (default: 1920)
- `--height`: Video height (default: 1080)
- `--start-frame`: Optional start frame (default: 0)
- `--end-frame`: Optional end frame (default: all frames)
- `--output-csv`: Optional CSV output file path
- `--output-json`: Optional JSON output file path

#### Output

Console output shows per-frame PSNR values and overall statistics for Y, U, V components.

#### Examples

```bash
# Compare two videos, all frames
python psnr_eval.py ref.yuv test.yuv --width 1920 --height 1080

# Compare frames 10-100, save to CSV
python psnr_eval.py ref.yuv test.yuv --width 1920 --height 1080 --start-frame 10 --end-frame 100 --output-csv results.csv

# Save results as JSON
python psnr_eval.py ref.yuv test.yuv --width 1920 --height 1080 --output-json results.json
```

#### Notes

- NV12 is a 4:2:0 YUV format with full-resolution Y plane and half-resolution interleaved UV plane
- PSNR values are reported in dB; identical frames report 100.0 dB
- U and V planes are half the height/width of the Y plane due to 4:2:0 subsampling


### NV12 to JPG Frame Extraction Script

A Python utility to extract individual frames from raw NV12 video files, convert them to RGB, and save as JPG images.

#### Overview

This script is designed for developers and testers working with Intel® VPL (Video Processing Library) who need to quickly visualize individual frames from raw NV12 video data. NV12 is a planar YUV 4:2:0 format commonly used in video processing.

**Key Features:**
- Extract specific frames by frame number
- Convert NV12 (YUV 4:2:0) to RGB using ITU-R BT.601 standard
- Adjustable JPG quality output
- Comprehensive error handling with clear messages
- Works with any even-dimensioned frame size

#### Usage

##### Required Arguments

- `--input <file>` — Path to raw NV12 file
- `--width <w>` — Frame width in pixels (must be even)
- `--height <h>` — Frame height in pixels (must be even)
- `--frame <n>` — Frame number to extract (0-indexed, non-negative)

##### Optional Arguments

- `--output <jpg>` — Output JPG path (default: `frame_<n>.jpg` in current directory)
- `--quality <q>` — JPG quality level 1-100 (default: 95)

#### Examples

##### Extract frame 0 from a 1920×1080 NV12 file

```bash
python nv12_to_jpg.py --input video.nv12 --width 1920 --height 1080 --frame 0
# Output: frame_0.jpg
```

##### Extract frame 100 with custom output path and lower quality

```bash
python nv12_to_jpg.py \
  --input video.nv12 \
  --width 1920 \
  --height 1080 \
  --frame 100 \
  --output my_frame_100.jpg \
  --quality 85
```

##### Extract frame from 1280×720 video

```bash
python nv12_to_jpg.py --input hd_video.nv12 --width 1280 --height 720 --frame 10 --output frame_10.jpg
```

---

Reloaded PyTorch implementation of WDSR, *BMVC 2019* [[pdf]](https://bmvc2019.org/wp-content/uploads/papers/0288-paper.pdf).

[Previous Implementations](https://github.com/JiahuiYu/wdsr_ntire2018)

## Performance
Small models

| Networks | Parameters | DIV2K (val) | Set5 | B100 | Urban100 | Pre-trained | Eval cmd | Train cmd |
| - | - | - | - | - | - | - | - | - |
| WDSR x2 | 1,190,100 | 34.76 | 38.08 | 32.23 | 32.34 | [Download](https://github.com/ychfan/wdsr/files/4176974/wdsr_x2.zip) | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --scale 2 --job_dir X --ckpt ./wdsr_x2/epoch_30.pth --eval_only```</details> | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --scale 2 --job_dir ./wdsr_x2```</details> |
| WDSR x3 | 1,195,605 | 31.03 | 34.45 | 29.14 | 28.33 | [Download](https://github.com/ychfan/wdsr/files/4176981/wdsr_x3.zip) | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --scale 3 --job_dir X --ckpt ./wdsr_x3/epoch_30.pth --eval_only```</details> | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --scale 3 --job_dir ./wdsr_x3```</details> |
| WDSR x4 | 1,203,312 | 29.04 | 32.22 | 27.61 | 26.21 | [Download](https://github.com/ychfan/wdsr/files/4176985/wdsr_x4.zip) | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --scale 4 --job_dir X --ckpt ./wdsr_x4/epoch_30.pth --eval_only```</details> | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --scale 4 --job_dir ./wdsr_x4```</details> |

Large models

| Networks | Parameters | DIV2K (val) | Set5 | B100 | Urban100 | Pre-trained | Eval cmd | Train cmd |
| - | - | - | - | - | - | - | - | - |
| WDSR x2 | 37,808,180 | 35.06 | 38.28 | 32.38 | 33.07 | [Download](https://drive.google.com/file/d/10OsQD--qWZIBinFignAWwppHw5z9LPMI/view?usp=sharing) | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --num_blocks 32 --num_residual_units 128 --scale 2 --job_dir X --ckpt ./wdsr_x2/epoch_30.pth --eval_only```</details> | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --num_blocks 32 --num_residual_units 128 --scale 2 --job_dir ./wdsr_x2```</details> |
| WDSR x3 | 37,826,645 | 31.34 | 34.76 | 29.32 | 28.94 | [Download](https://drive.google.com/file/d/10Yh0mI2825k69vChRZRGMsAC7C-M5hbk/view?usp=sharing) | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --num_blocks 32 --num_residual_units 128 --scale 3 --job_dir X --ckpt ./wdsr_x3/epoch_30.pth --eval_only```</details> | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --num_blocks 32 --num_residual_units 128 --scale 3 --job_dir ./wdsr_x3```</details> |
| WDSR x4 | 37,852,496 | 29.33 | 32.58 | 27.78 | 26.79 | [Download](https://drive.google.com/file/d/10sYc5F63-o3eovtGCG5SSawk4otEHIxe/view?usp=sharing) | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --num_blocks 32 --num_residual_units 128 --scale 4 --job_dir X --ckpt ./wdsr_x4/epoch_30.pth --eval_only```</details> | <details><summary>details</summary>```python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --num_blocks 32 --num_residual_units 128 --scale 4 --job_dir ./wdsr_x4```</details> |

## Usage

### Dependencies
```bash
conda install pytorch torchvision -c pytorch
conda install tensorboard h5py scikit-image
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git
```

### Evaluation

```bash
python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --scale 2 --job_dir ./wdsr_x2 --eval_only
# or
python trainer.py --dataset div2k --eval_datasets div2k set5 bsds100 urban100 --model wdsr --scale 2 --job_dir ./wdsr_x2 --ckpt ./latest.pth --eval_only
```

## Datasets
[DIV2K dataset: DIVerse 2K resolution high quality images as used for the NTIRE challenge on super-resolution @ CVPR 2017](https://data.vision.ee.ethz.ch/cvl/DIV2K/)

[Benchmarks (Set5, BSDS100, Urban100)](http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_testing_datasets.zip)

Download and organize data like: 
```bash
wdsr/data/DIV2K/
├── DIV2K_train_HR
├── DIV2K_train_LR_bicubic
│   └── X2
│   └── X3
│   └── X4
├── DIV2K_valid_HR
└── DIV2K_valid_LR_bicubic
    └── X2
    └── X3
    └── X4
wdsr/data/Set5/*.png
wdsr/data/BSDS100/*.png
wdsr/data/Urban100/*.png
```
