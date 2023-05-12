# AESTransformers

## Requirements

The code is tested on Python 3.9 and PyTorch 1.10.1.
We recommend to create a new environment for experiments using `conda`:
```bash
conda create -y -n aes-transformers python=3.8
conda activate aes-transformers
```

For Jupyter Notebook, run:
```bash
conda install -n aes-transformers ipykernel --update-deps --force-reinstall
```

Then, from the project root, run:
```bash
pip install -r requirements.txt
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install git+https://github.com/huggingface/transformers
pip install protobuf==3.20.*
```

For further development or modification, we recommend installing `pre-commit`:
```bash
pre-commit install
```

To ensure that PyTorch is installed and CUDA works properly, run:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

We should see:
```bash
2.0.1
True
```
