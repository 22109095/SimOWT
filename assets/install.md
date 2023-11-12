# Install
## Requirements
We test the codes in the following environments, other versions may also be compatible but Pytorch vision should be >= 1.7

- CUDA 11.1
- Python 3.7
- Pytorch 1.8.1
- Torchvison 0.9.1

## Install 
First, clone the repository locally:

```bash
git clone https://github.com/22109095/SimOWT.git
cd SimOWT
```

Install dependencies and pycocotools:

```bash
pip install torch>=1.7.0(pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html)
pip install -r requirements.txt
pip install -e .
pip install shapely==1.7.1
pip install git+https://github.com/youtubevos/cocoapi.git#"egg=pycocotools&subdirectory=PythonAPI"
```

Install detectron2:
```bash
python setup.py develop
```

Compiling Deformable DETR CUDA operators:

```bash
cd projects/IDOL/idol/models/ops/
sh make.sh
cd ../../../../..
```