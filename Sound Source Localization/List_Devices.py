import pyaudio
import matplotlib.pyplot as plt 
import numpy as np
import wave

maxValue = 2**16
bars = 35
p=pyaudio.PyAudio()

numdevices = p.get_host_api_info_by_index(0).get('deviceCount')
for i in range(0, numdevices):
    if(p.get_device_info_by_host_api_device_index(0,i).get('maxInputChannels'))>0:
        print("input device id {} with channel {} -- {}".format(i,p.get_device_info_by_host_api_device_index(0,i).get('maxInputChannels'),p.get_device_info_by_host_api_device_index(0,i).get('name')));
