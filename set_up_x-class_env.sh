apt-get update
apt-get install wget
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
. ~/.bashrc
#conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
~/miniconda3/bin/conda install pytorch==1.4.0 cudatoolkit=10.1 -c pytorch