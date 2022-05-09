@echo off    
pip install torch==1.10.0+cpu torchvision==0.11.0+cpu torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
exit /b