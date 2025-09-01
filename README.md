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

#### Fully use NPU computation capability

Please following below to modify line `63` of `ov_infer.py` to make model runs with 6 tiles on LNL.

```py
compiled_model = core.compile_model(ov_model_path, "NPU"
    {"NPU_DPU_GROUPS" : 6, "NPU_MAX_TILES": 6, "PERFORMANCE_HINT": "LATENCY"}
)
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
