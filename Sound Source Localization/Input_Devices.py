import pyaudio
import queue
import threading
import math
import numpy as np
import matplotlib.pyplot as plt
from Gcc_Phat import gccphat

#Mic-Array Class ----> Create mic objects according to channels, rate, chunk_size  

class MicArray:

    def __init__(self,device_index = None,rate = 16000,channels = 4,chunk_size = 1024):
        self.p = pyaudio.PyAudio()
        self.q = queue.Queue()
        self.thread_event = threading.Event()

        self.rate = rate
        self.channels = channels
        self.chunk_size = chunk_size
        
        #find the device with given number of channels and select it.
        if device_index == None:
            for i in range(self.p.get_device_count()):
                dev = self.p.get_device_info_by_index(i)
                name = dev['name'].encode('utf-8')
                print(i, name, dev['maxInputChannels'], dev['maxOutputChannels'])
                if dev['maxInputChannels'] == self.channels:
                    print('Use {}'.format(name))
                    device_index = i
                    break

        #Create a stream, start = false only starts when instance is created with and as        
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

    #calls _callback when input_data is present and is pushed in the queue          
    def _callback(self,input_data,frame_count,time_info,status_flag):
        self.q.put(input_data)
        return (None,pyaudio.paContinue)
    
    #Clear and start the stream
    def start(self):
        self.q.queue.clear()
        self.stream.start_stream()
    
    #Set the thread flag as set and all the threads waiting for it to become true are awakened
    def stop(self):
        self.thread_event.set()
        self.stream.stop_stream()
        self.q.put('')

    #Read the data that is queued in a queue and return the frames
    def read_mic_data(self):
        self.thread_event.clear()
        while not self.thread_event.is_set():
            frames = self.q.get()
            if not frames:
                break
            
            frames = np.frombuffer(frames,dtype = 'int16')
            yield frames

    #Enters into this when instantiated inside with statement
    def __enter__(self):
        self.start()
        return self

    #Enters into this when the object scope goes out of the with scope
    def __exit__(self,exception_type,exception_value,traceback):
        if exception_value:
            return False
        self.stop()

#find GCC_PHAT of single data_frame passed here    
def find_direction_gccphat(data_frame,fs,tau_mic_dist):
    tau,_ = gccphat(data_frame[0::4],data_frame[3::4],fs=fs,max_tau=tau_mic_dist,interp = 16)
    theta = math.asin(tau/tau_mic_dist) * 180 / math.pi 
    
    return theta+90
    