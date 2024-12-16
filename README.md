# dlp-nppe-3


**Note**: The submission is also available as a github repository at:

[`https://github.com/jaidevd/dlp-nppe-3`](https://github.com/jaidevd/dlp-nppe-3)

This super resolution task is attempted in two stages:

**Stage 1**: Image denoising (Notebook: [`01_denoise.ipynb`](01_denoise.ipynb))

**Stage 2**: Image upsampling (Notebook: [`02_superres.ipynb`](02_superres.ipynb))

To reproduce the results, run the following steps exactly in the specified
order.

### Stage 1 - Image Denoising

The consolidated steps for this stage are present in the `01_denoise.ipynb`
notebook.

The training data for this stage is the original low-res, noisy images paired
with the downsampled version of the high-res images.

To train and save the denoising model, do the following steps:

1. Load the data in the current directory under a folder named `archive`, such
   that it has the following subfolders:

   ```
   archive/train/train/
   archive/train/gt/
   archive/val/val/
   archive/val/gt/
   archive/test/
   ```

2. Run the script `train_mprnet.py` as follows:

   ```
   python train_mprnet.py
   ```
   The number of epochs, batch sizes can be modified in the script.

   This will create a folder named `models`, and the denoising MPRNet model will
   be saved within it.

3. Generate the denoised images as follows:

    ```
    python denoise.py
    ```
    _Note_ that this script expects the `models/mprnet-denoiser.pt` model to be
    present. (Otherwise, change the paths as needed).

    This will create a folder named `denoised/` and put the denoised versions of
    the training and validation images in it.


### Stage 2 - Image Super-resolution

The consolidated steps for this stage are present in the `02_superres.ipynb`
notebook.

To run an SRGAN on the super-resolution task, do the following steps:

1. Train the model by running the script `train_srgan.py` as follows:

   ```
   python train_srgan.py
   ```
   The number of epochs, batch sizes can be modified in the script.
   The resulting generator and discriminator models will be saved in the
   `models/` folder.

   _Note_ that this script expects the `denoised/` folder to be present.

2. Generate the super-resolved images as follows:

    ```
    python test_srgan.py
    ```

    This will create a folder named `hr-output` containing the 4x scaled images.

3. Create a submission by running the following:
    
    ```
    python submission.py
    ```
    Note that this script contains paths assuming the `hr-output` folder exists - modify as needed.
    This should generate the final submission.csv file.
