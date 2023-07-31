### Ethnic group classifier

ResNet-18-based classifier of face ethnicity into 4 classes (African, East Asian, Indian, Caucasian) that was used in the paper:

-----
Artem Sevastopolsky, Yury Malkov, Nikita Durasov, Luisa Verdoliva, Matthias Nießner. <i>How to Boost Face Recognition with StyleGAN?</i> // International Conference on Computer Vision (ICCV), 2023

### <img align=center src=./docs/project.png width='32'/> [Project page](https://seva100.github.io/stylegan-for-facerec) &ensp; <img align=center src=./docs/paper.png width='24'/> [Paper](https://arxiv.org/abs/2210.10090) &ensp; <img align=center src=./docs/video.png width='24'> [Video](https://www.youtube.com/watch?v=Bsi0RMTdEaI) &ensp; <img align=center src=./docs/code.png width='24'> [Code](https://github.com/seva100/stylegan-for-facerec) 

-----
<img src=./docs/Demonstration.png width=1200>

The model was trained in 2 stages:

1. Training ResNet-18 (from ImageNet checkpoint) on images from [BUPT-BalancedFace](http://www.whdeng.cn/RFW/Trainingdataste.html) and corresponding ethnicity labels. We noticed that after this stage, the model performs reasonably well but still makes consistent mistakes.
2. Inferring the model on [WebFace-42M](https://www.face-benchmark.org/) dataset (containing many images of the same people) and applying consensus procedure: considering an identity of ethnicity X, if and only if for >= 80\% of this identity's images, the classifier predicted the race X (not more than 20 random images per identity considered). Training a new classifier on the subset of selected ethnicities WebFace-42M and corresponding labels. 

The identities list can be found in the data released for the paper and is accessible by filling in the [form]() (see [stylegan-for-facerec](https://github.com/seva100/stylegan-for-facerec) for more details).

This code only provides the inference interface.

### Running

1. Set up the environment: `pytorch`, `torchvision`, `pytorch-lightning`, `matplotlib`, `einops`, `tqdm`. For alignment (step 3), download the source of `mtcnn-pytorch` in a separate folder and install its dependencies.
2. Download the checkpoint from [here](https://drive.google.com/drive/folders/1-RaAavxHZ_ijFuf-QV3n8PhFk4SJj8Qx?usp=sharing) and put it in `pretrained_models` folder.
3. If required, align & crop the images in your folder `input_dir` via:

```bash
python facesets/mtcnn_crop_align.py \
    --in_dir <input_dir> \
    --out_dir <output folder with cropped and aligned images> \
    --mtcnn_pytorch_path <path, to which mtcnn-pytorch source has been downloaded> \
    --n_threads <number of parallel threads on the same GPU>
```

The resulting images should have 112x112 resolution and be in alignment with each other by facial landmarks.

4. Run the ethnicity classifier inference:

```bash
python infer.py \
    --input_dir <aligned & cropped input images> \
    --output_dir <output dir> \
    --ckpt pretrained_models/resnet18.ckpt \
    [--visualize] \
    [--n_vis_samples_per_pic <if --visualize, this defines how many samples should each plot contain>] \
    [--n_vis_rows_per_pic <if --visualize, this defines how many rows should each plot contain -- must divide --n_vis_samples_per_pic>] \ 
    [--figsize_h <fig height>] [--figsize_w <fig width>]
```

The `output dir` will contain a file `pred_ethnicities.txt` with space-delimited pairs `<filename> <ethnicity>`. If `--visualize` option is selected, grids of images with corresponding predictions will also be added there.

