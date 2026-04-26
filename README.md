first commit

python3 cnn.py --epochs 30 --batch_size 128 --lr 0.001



pip install torch torchvision matplotlib certifi snntorch ultralytics

python3 -m venv .venv
source .venv/bin/activate

--Windows
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser


py -m venv .venv
.\.venv\Scripts\Activate
.\.venv\Scripts\deactivate.bat

Certificate error when running it
open "/Applications/Python 3.14/Install Certificates.command"


GPU
pip uninstall torch torchvision torchaudio -y

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


for NVIDIA 5090
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130


nvidia-smi

Windows
for python 3.11 use python
for latest python use python3

python -m venv .venv
.\.venv\Scripts\Activate


python cnn.py --epochs 30 --batch_size 128 --lr 0.001

have to install CUDA
https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local

for SNN;
python ann_snn.py --weights cnn_cifar10.pth --num_steps 25 --batch_size 128
python ann_snn.py --weights cnn_cifar10_best.pth --num_steps 25 --batch_size 128

python surrogate_snn.py --epochs 10 --batch_size 128 --lr 0.001 --num_steps 25

python faster_R_CNN.py
python YOLO.py
python inference.py
