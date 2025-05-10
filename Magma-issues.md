


After the `pyav` issue was temporarily resolved by `@jwyang`

Installation Process and Related Errors:

1. Repo Cloning

```
git clone https://github.com/microsoft/Magma
cd Magma
```

2. Create environment first

```
conda create -n magma python=3.10 pip -y
conda activate magma
pip install --upgrade pip
```

3. Installing the dependencies

```
pip install -e .
```

(a) ERROR: ModuleNotFoundError: No module named 'torch'

```
Collecting flash-attn (from magma==0.0.1)
  Downloading flash_attn-2.7.4.post1.tar.gz (6.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.0/6.0 MB 6.7 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... error
  error: subprocess-exited-with-error

  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [6 lines of output]
      Traceback (most recent call last):
        File "<string>", line 2, in <module>
        File "<pip-setuptools-caller>", line 35, in <module>
        File "/tmp/pip-install-yc_u2bcz/flash-attn_957eb523118c423193d37d6f61265f55/setup.py", line 22, in <module>
          import torch
      ModuleNotFoundError: No module named 'torch'
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.
```

To fix it, I first installed pytorch using:

```
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

as pytorch 2.3.2 with cuda 12.1 is mentioned in the magma.yml file.

Running `pip install -e .` again

(b) ERROR: OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.

```
Collecting flash-attn (from magma==0.0.1)
  Using cached flash_attn-2.7.4.post1.tar.gz (6.0 MB)
  Preparing metadata (setup.py) ... error
  error: subprocess-exited-with-error

  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [19 lines of output]
      /tmp/pip-install-bsfn6l8w/flash-attn_6afe81fcf1344407ae065cb8db7f254a/setup.py:106: UserWarning: flash_attn was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, only images whose names contain 'devel' will provide nvcc.
        warnings.warn(
      Traceback (most recent call last):
        File "<string>", line 2, in <module>
        File "<pip-setuptools-caller>", line 35, in <module>
        File "/tmp/pip-install-bsfn6l8w/flash-attn_6afe81fcf1344407ae065cb8db7f254a/setup.py", line 198, in <module>
          CUDAExtension(
        File "/home/io452/miniconda3/envs/magma/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 1077, in CUDAExtension
          library_dirs += library_paths(cuda=True)
        File "/home/io452/miniconda3/envs/magma/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 1204, in library_paths
          if (not os.path.exists(_join_cuda_home(lib_dir)) and
        File "/home/io452/miniconda3/envs/magma/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 2419, in _join_cuda_home
          raise OSError('CUDA_HOME environment variable is not set. '
      OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.


      torch.__version__  = 2.3.1


      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.
```

Solved using: `conda install -c conda-forge cudatoolkit-dev -y`. It installs cudatoolkit 11.7 as it is the latest available on conda-forge. I found this solution at [how to set CUDA_HOME environment](https://stackoverflow.com/questions/52731782/get-cuda-home-environment-path-pytorch)

Running `pip install -e .` again.

Note: Both errors come from installing `flash-attn`

This time it runs fine. Next steps:

```
pip install -e ".[train]"
pip install -e ".[agent]"
```

The last step gives quite a few errors:

```
Collecting paddle==1.0.2 (from magma==0.0.1)
  Downloading paddle-1.0.2.tar.gz (579 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 579.0/579.0 kB 4.2 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... error
  error: subprocess-exited-with-error

  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [8 lines of output]
      Traceback (most recent call last):
        File "<string>", line 2, in <module>
        File "<pip-setuptools-caller>", line 35, in <module>
        File "/tmp/pip-install-8umf4cuw/paddle_93ec0d71ef2945e3a25b26ba6d46c045/setup.py", line 3, in <module>
          import paddle
        File "/tmp/pip-install-8umf4cuw/paddle_93ec0d71ef2945e3a25b26ba6d46c045/paddle/__init__.py", line 5, in <module>
          import common, dual, tight, data, prox
      ModuleNotFoundError: No module named 'dual'
      [end of output]
   note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.
  ```

  The following libraries were missing which is why there is a series of errors coming from metadata generation with `paddle==1.0.2`:

  `common, dual, tight`

So, install them with the command: `pip install common dual tight`. I did them one by one, running `pip install -e ".[agent]"` everytime after each library and getting error for the next.

The next error from this step comes with the absence of the libraries: `cv2` and `ipython`. The errors are the same: `ModuleNotFoundError` and `error: metadata-generation-failed`. 

So, install them with the command: `pip install opencv-python ipython`. Again, I did them individually.

Next error comes from the absence of `cotracker`. So, I followed the next steps from the repository and installed `cotracker` and `kmeans_pytorch`:

```
# Install co-tracker
git clone https://github.com/facebookresearch/co-tracker
cd co-tracker
pip install -e .
pip install imageio[ffmpeg]
cd ../
```

```
# Install kmeans_pytorch, note: install with pip will leads to error
git clone https://github.com/subhadarship/kmeans_pytorch
cd kmeans_pytorch
pip install -e .
cd ../
```

Ran `pip install -e ".[agent]"` again. Got error with the absence of `faiss` and `prox`. I installed `faiss`, `decord`, and `prox` individually and then finally ran `pip install -e ".[agent]"`. This time it executed without error. 

You can install all the libraries with one single command: `pip install common dual tight prox opencv-python ipython faiss-cpu decord`. 

Then you can individually install `cotracker` and `kmeans_pytorch` following the instructions above.