#!/bin/bash
# First time setup

cd ~/IoTCamera 

sudo pip3 install opencv-python  
sudo apt-get install libcblas-dev 
sudo apt-get install libhdf5-dev 
sudo apt-get install libhdf5-serial-dev  
sudo apt-get install libatlas-base-dev  
sudo apt-get install libjasper-dev  
sudo apt-get install libqtgui4  
sudo apt-get install libqt4-test 

pip3 install -r requirements.txt 

sudo cp start_camera.sh /etc/network/if-up.d/