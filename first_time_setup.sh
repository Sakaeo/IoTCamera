#!/bin/bash
# First time setup

#cd /root/IoTCamera 

sudo pip3 install opencv-python  
sudo apt-get install libcblas-dev 
sudo apt-get install libhdf5-dev 
sudo apt-get install libhdf5-serial-dev  
sudo apt-get install libatlas-base-dev  
sudo apt-get install libjasper-dev  
sudo apt-get install libqtgui4  
sudo apt-get install libqt4-test
sudo apt-get install cmake

sudo pip3 install -r requirements.txt

sudo mkdir /root/IoTCamera
sudo cp -a src /root/IoTCamera

sudo cp iotcamera.defaults /etc/default/iotcamera
sudo cp iotcamera.service /etc/systemd/system/
sudo systemctl start iotcamera.service
