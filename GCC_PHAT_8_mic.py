import pyaudio
import queue
import threading
import numpy as np
import math
import wave
import matplotlib.pyplot as plt
from scipy.io import wavfile

############################# MICROPHONE CUBE CENTROID, Centroid at (0,0,0) ,Cube length 15cm , mic-distance 15cm ######################

########### mic1(-5.5,-5.5,-8.5) , mic2(5.5,5.5,-8.5) , mic3(-5.5,5.5,8.5) , mic4(5-5,-5.5,8.5)##########
########### mic5(8.5,5.5,5.5) , mic6(8.5,-5.5, -5.5) , mic7(-8.5,5.5,-5.5) , mic8(-8.5,-5.5,5.5)#########

distance = 0.15
max_time = distance/343.2


class MicArray:

    def __init__(self,device_index = None,rate = 16000,channels = 4,chunk_size = 1024):
        self.p = pyaudio.PyAudio()
        self.q = queue.Queue()
        self.thread_event = threading.Event()

        self.rate = rate
        self.channels = channels
        self.chunk_size = chunk_size
        
        if device_index == None:
            for i in range(self.p.get_device_count()):
                dev = self.p.get_device_info_by_index(i)
                name = dev['name'].encode('utf-8')
                print(i, name, dev['maxInputChannels'], dev['maxOutputChannels'])
                if dev['maxInputChannels'] == self.channels:
                    print('Use {}'.format(name))
                    device_index = i
                    break

                
        self.stream = self.p.open(
            input_device_index = device_index,
            start = False,
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._callback
        )
               
    def _callback(self,input_data,frame_count,time_info,status_flag):
        self.q.put(input_data)
        return (None,pyaudio.paContinue)
    
    def start(self):
        self.q.queue.clear()
        self.stream.start_stream()
    
    def stop(self):
        self.thread_event.set()
        self.stream.stop_stream()
        self.q.put('')

    def read_mic_data(self):
        self.thread_event.clear()
        while not self.thread_event.is_set():
            frames = self.q.get()
            if not frames:
                break
            
            frames = np.frombuffer(frames,dtype = 'int16')
            yield frames

    def __enter__(self):
        self.start()
        return self

    def __exit__(self,exception_type,exception_value,traceback):
        if exception_value:
            return False
        self.stop()
        

def gccphat(sig,refsig,fs,interp=16):
    
    global max_time
    n = sig.shape[0] + refsig.shape[0]

    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))
    
    max_shift = np.minimum(int(interp * fs * max_time), int(interp * n/2))
    
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    
    shift = np.argmax(np.abs(cc)) - max_shift
    
    tau = shift / float(interp * fs)
    
    return tau

def theta_intersection(tau1,tau2):
    
    global max_time
    theta1 = np.arcsin(tau1/max_time) * 180 / np.pi
    theta2 = np.arcsin(tau2/max_time) * 180 / np.pi
    
    if np.abs(theta1) < np.abs(theta2):
        if theta2 > 0:
            final_theta = (theta1 + 360) % 360
        else:
            final_theta = (180 - theta1)
    else:
        if theta1 < 0:
            final_theta = (theta2 + 360) % 360
        else:
            final_theta = (180 - theta2)

        final_theta = (final_theta + 90 + 180) % 360
        
    
    
    final_theta = int((-final_theta + 120) % 360)
    return final_theta


def find_direction_gccphat(data_frame1,data_frame2,fs):

    mic1 = data_frame1[0::4]
    mic2 = data_frame1[1::4]
    mic3 = data_frame1[2::4]
    mic4 = data_frame1[3::4]

    mic5 = data_frame2[0::4]
    mic6 = data_frame2[1::4]
    mic7 = data_frame2[2::4]
    mic8 = data_frame2[3::4]

    ############Elevation--------------1 --- YZ plane
    tau1 = gccphat(mic1,mic2,fs=fs,interp=16)
    tau2 = gccphat(mic3,mic4,fs=fs,interp=16)
    
    elevation1 = theta_intersection(tau1,tau2) 

    ################Elevation-----------2 --- XY plane 

    tau1 = gccphat(mic5,mic6,fs=fs,interp=16)
    tau2 = gccphat(mic7,mic8,fs=fs,interp=16)
    
    elevation2 = theta_intersection(tau1,tau2) 



    final_theta = (elevation1,elevation2)

    return final_theta

def find_direction_4_gccphat(data_frame1,fs):

    mic1 = data_frame1[0::4]
    mic2 = data_frame1[1::4]
    mic3 = data_frame1[2::4]
    mic4 = data_frame1[3::4]


    tau1 = gccphat(mic1,mic4,fs=fs,interp=16)
    tau2 = gccphat(mic2,mic3,fs=fs,interp=16)
    
    theta = theta_intersection(tau1,tau2) 

    return theta

def Cube_8_test():
    import signal
    import time
    
    is_quit = threading.Event()
    
    def signal_handler(sig, num):
        is_quit.set()
        print('Exited')
    
    signal.signal(signal.SIGINT, signal_handler)
    
    
    with MicArray(device_index=5 ,channels=4) as array1,MicArray(device_index=7 ,channels=4) as array2:
        for chunk1,chunk2 in zip(array1.read_mic_data(),array2.read_mic_data()):
            direction = find_direction_gccphat(chunk1,chunk2,fs=array1.rate)
            print(direction)

            if is_quit.is_set():
                print("quited")
                break


def Cube_8_test():
    import signal
    import time
    
    is_quit = threading.Event()
    
    def signal_handler(sig, num):
        is_quit.set()
        print('Exited')
    
    signal.signal(signal.SIGINT, signal_handler)

    
    with MicArray(device_index=5 ,channels=4) as array1,MicArray(device_index=7 ,channels=4) as array2:
        for chunk1,chunk2 in zip(array1.read_mic_data(),array2.read_mic_data()):
            direction = find_direction_gccphat(chunk1,chunk2,fs=array1.rate)
            print(direction)

            if is_quit.is_set():
                print("quited")
                break

def Cube_4_test():
    import signal
    import time
    
    is_quit = threading.Event()
    
    def signal_handler(sig, num):
        is_quit.set()
        print('Exited')
    
    signal.signal(signal.SIGINT, signal_handler)
    #f = open("data.dat","w+")
    
    with MicArray(device_index=5 ,channels=4) as array1:
        for chunk1 in array1.read_mic_data():
            direction = find_direction_4_gccphat(chunk1,fs=array1.rate)
            print(direction)
            #f.writelines(str(direction)[1:-1]+"\n")

            if is_quit.is_set():
                print("quited")
                #f.close()
                break


def main():
    #Cube_4_test()
    Cube_8_test()
    
    
if __name__ == "__main__":
    main()
    