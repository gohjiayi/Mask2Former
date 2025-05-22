# Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation (CVPR 2022)

[Bowen Cheng](https://bowenc0221.github.io/), [Ishan Misra](https://imisra.github.io/), [Alexander G. Schwing](https://alexander-schwing.de/), [Alexander Kirillov](https://alexander-kirillov.github.io/), [Rohit Girdhar](https://rohitgirdhar.github.io/)

[[`arXiv`](https://arxiv.org/abs/2112.01527)] [[`Project`](https://bowenc0221.github.io/mask2former)] [[`BibTeX`](#CitingMask2Former)]

<div align="center">
  <img src="https://bowenc0221.github.io/images/maskformerv2_teaser.png" width="100%" height="100%"/>
</div><br/>

### Features
* A single architecture for panoptic, instance and semantic segmentation.
* Support major segmentation datasets: ADE20K, Cityscapes, COCO, Mapillary Vistas.

## Updates
* Add Google Colab demo.
* Video instance segmentation is now supported! Please check our [tech report](https://arxiv.org/abs/2112.10764) for more details.

## Installation

See [installation instructions](INSTALL.md).

## Getting Started

See [Preparing Datasets for Mask2Former](datasets/README.md).

See [Getting Started with Mask2Former](GETTING_STARTED.md).

Run our demo using Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uIWE5KbGFSjrxey2aRd5pWkKNY1_SaNq)

Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/Mask2Former)

Replicate web demo and docker image is available here: [![Replicate](https://replicate.com/facebookresearch/mask2former/badge)](https://replicate.com/facebookresearch/mask2former)

## Advanced usage

See [Advanced Usage of Mask2Former](ADVANCED_USAGE.md).

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Mask2Former Model Zoo](MODEL_ZOO.md).

## License

Shield: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The majority of Mask2Former is licensed under a [MIT License](LICENSE).


However portions of the project are available under separate license terms: Swin-Transformer-Semantic-Segmentation is licensed under the [MIT license](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE), Deformable-DETR is licensed under the [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE).

## <a name="CitingMask2Former"></a>Citing Mask2Former

If you use Mask2Former in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@inproceedings{cheng2021mask2former,
  title={Masked-attention Mask Transformer for Universal Image Segmentation},
  author={Bowen Cheng and Ishan Misra and Alexander G. Schwing and Alexander Kirillov and Rohit Girdhar},
  journal={CVPR},
  year={2022}
}
```

If you find the code useful, please also consider the following BibTeX entry.

```BibTeX
@inproceedings{cheng2021maskformer,
  title={Per-Pixel Classification is Not All You Need for Semantic Segmentation},
  author={Bowen Cheng and Alexander G. Schwing and Alexander Kirillov},
  journal={NeurIPS},
  year={2021}
}
```

## Acknowledgement

Code is largely based on MaskFormer (https://github.com/facebookresearch/MaskFormer).

# Customisation
This customisation portion is to finetune the model on a custom dataset to detect safety barrier components, consisting of 4 classes: `rail_top`, `rail_mid`, `toeboard`, `vertical_post`. Based on the pre-trained models, the *COCO instance segmentation model with a Swin-Large backbone* is preferred. Experimentations with the Swin-Tiny backbone is also conducted, and you will be able to see commands for both below.

## Custom Demo
To run the demo with the pre-trained models, download the relevant model weights from the model zoo, reference the correct weights and config file, indicate the input image and output image paths and run the following command.
If you'd prefer viewing the output in a new window showing using OpenCV, remove the `--output` flag, but take note if you are running this in a headless server environment, SSH path forwarding may be required.

```bash
cd demo/

# COCO Instance Segmentation Swin-Tiny backbone
python demo.py --config-file ../configs/coco/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml \
  --input ../assets/sample_image.jpg \
  --output ../assets/sample_output_tiny.jpg \
  --opts MODEL.WEIGHTS ../model_final_1e7f22.pkl

# COCO Instance Segmentation Swin-Large backbone
python demo.py --config-file ../configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml \
  --input ../assets/sample_image.jpg \
  --output ../assets/sample_output_large.jpg \
  --opts MODEL.WEIGHTS ../model_final_e5f453.pkl

cd ..
```

## Custom Finetuning

### Preparation
The dataset is expected to be in the following format based on dataset registration in `mask2former/data/datasets/register_safetybarrier_instance.py`.

```
datasets/
  safetybarrier/
    images/
      train/
        img_00001.jpg
        img_00002.jpg
        ...
      val/
        img_00001.jpg
        img_00002.jpg
        ...
    annotations/
      instances_train.json
      instances_val.json
```

`mask2former/data/dataset_mappers/safetybarrier_instance_dataset_mapper.py` is used to map the custom dataset to the Mask2Former format.

The configs for finetuning the models are in the paths below, with one for Swin-Tiny and one for Swin-Large as indicated in the file names.

```
configs/safetybarrier/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml
configs/safetybarrier/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml
```

### Training
Perform the commands in the root folder.

```bash
# Set your GPUs
export CUDA_VISIBLE_DEVICES=2,3

# Finetune Swin-Tiny
python train_net.py \
  --config-file configs/safetybarrier/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml \
  --num-gpus 2 \
  MODEL.WEIGHTS model_final_1e7f22.pkl \
  OUTPUT_DIR output_tiny

# Finetune Swin-Large
python train_net.py \
  --config-file configs/safetybarrier/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml \
  --num-gpus 2 \
  MODEL.WEIGHTS model_final_e5f453.pkl \
  OUTPUT_DIR output_large
```

### Demo

```bash
# Run the demo with the finetuned model
cd demo/

# Finetuned Swin-Tiny
python demo.py --config-file ../configs/safetybarrier/instance-segmentation/swin/maskformer2_swin_tiny_bs16_50ep.yaml \
  --input ../assets/sample_image.jpg \
  --output ../assets/sample_output_finetuned_tiny.jpg \
  --confidence-threshold 0.8 \
  --opts MODEL.WEIGHTS ../output_tiny/model_final.pth

# Finetuned Swin-Large
python demo.py --config-file ../configs/safetybarrier/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml \
  --input ../assets/sample_image.jpg \
  --output ../assets/sample_output_finetuned_large.jpg \
  --confidence-threshold 0.8 \
  --opts MODEL.WEIGHTS ../output_large/model_final.pth
```

## Model-Assisted Labelling Prediction
A custom model was used to generate initial predictions for semi-automated labeling of the safety barrier dataset, based on a small manually labeled subset. These preliminary annotations were uploaded to the labeling platform (CVAT) to kickstart the labeling process. The predictions were then refined by human annotators, accelerating the overall annotation workflow and producing high-quality labels for subsequent model fine-tuning.

Make sure input_dir contains:
```
input_dir/
├── images/                  # Images to predict
│   ├── image1.jpg
│   └── ...
└── annotations.xml          # Original annotations with metadata from CVAT
```

The output_dir will contain:
```
output_dir/
├── annotations.xml          # Final updated annotations
├── visuals/                 # Visualized predictions per image
│   ├── image1.jpg
│   └── ...
├── predictions/             # Checkpoint individual JSONs for each image (RLE, classes, etc.)
│   ├── image1.json
│   └── ...
```

This should only be executed on the Finetuned Swin-Large model, since it will have a better result in comparison to the Swin-Tiny backbone.

```bash
# Set your GPU, one is enough
export CUDA_VISIBLE_DEVICES=3

cd predict/

# Finetuned Swin-Large on Test Datasplit
python run_annotation_pipeline.py \
  --input_dir input/test \
  --output_dir output/test \
  --config-file ../configs/safetybarrier/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml \
  --confidence-threshold 0.5 \
  --batch-size 8 \
  --opts MODEL.WEIGHTS ../output_large/model_final.pth

# Finetuned Swin-Large on Train Datasplit
python run_annotation_pipeline.py \
  --input_dir input/train \
  --output_dir output/train \
  --config-file ../configs/safetybarrier/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml \
  --confidence-threshold 0.5 \
  --batch-size 8 \
  --opts MODEL.WEIGHTS ../output_large/model_final.pth

# Finetuned Swin-Large on Val Datasplit
python run_annotation_pipeline.py \
  --input_dir input/val \
  --output_dir output/val \
  --config-file ../configs/safetybarrier/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml \
  --confidence-threshold 0.5 \
  --batch-size 8 \
  --opts MODEL.WEIGHTS ../output_large/model_final.pth
```