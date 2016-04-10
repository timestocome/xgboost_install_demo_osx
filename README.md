How I installed xgboost on OSX

Download
https://github.com/dmlc/xgboost


git clone --recursive https://github.com/dmlc/xgboost
cd xgboost/

cp make/minimum.mk ./config.mk

make
python setup.py install

cd python-package
sudo python setup.py install


Edit your .bashrc_profile and add the following


export PYTHONPATH=/xgboost/python-package


----------------------------------------------
I included a demo program to test your install and a more complex on that works through the SF Crime data
