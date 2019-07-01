import pyaudio
import webrtcvad
import queue
import threading
import numpy as np
import math
import wave
import matplotlib.pyplot as plt
from scipy.io import wavfile


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
        

def gccphat(sig,refsig,fs,max_tau,interp=16):
    
    n = sig.shape[0] + refsig.shape[0]

    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))
    
    max_shift = np.minimum(int(interp * fs * max_tau), int(interp * n/2))
    
    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))
    
    shift = np.argmax(np.abs(cc)) - max_shift
    
    tau = shift / float(interp * fs)
    
    return tau



def find_direction_gccphat(data_frame,fs,tau_mic_dist):
    tau1 = gccphat(data_frame[0::4],data_frame[3::4],fs=fs,max_tau=tau_mic_dist,interp=16)
    tau2 = gccphat(data_frame[2::4],data_frame[1::4],fs=fs,max_tau=tau_mic_dist,interp=16)
    
    theta1 = math.asin(tau1/tau_mic_dist) * 180 / math.pi
    theta2 = math.asin(tau2/tau_mic_dist) * 180 / math.pi
    
    
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



def mic_array_test():
    import signal
    import time
    
    distance = 0.145
    max_time = distance/343.2
    
    is_quit = threading.Event()
    
    def signal_handler(sig, num):
        is_quit.set()
        print('Exited')
    
    signal.signal(signal.SIGINT, signal_handler)
    
    with MicArray(channels=4) as mic:
        for chunk in mic.read_mic_data():
            direction = find_direction_gccphat(chunk,fs=mic.rate,tau_mic_dist=max_time)
            print(direction)
            
            if is_quit.is_set():
                break


def vad_mic_test():
    Rate = 16000
    VAD_Frames = 10  #in ms
    DOA_Frames = 160 #in ms
    
    distance = 0.0568
    max_time = distance/343.2

    vad = webrtcvad.Vad(3) #Set vad to aggressive mode 0--3
    
    speech_count = 0
    chunks = []
    doa_chunks = int(DOA_Frames/VAD_Frames)  # 8 doa chunks
    
    
    try:
        with MicArray(channels=4 ,chunk_size = (Rate//1000) * VAD_Frames) as mic:  #chunk size 128
            for chunk in mic.read_mic_data():
                if vad.is_speech(chunk[0::4].tobytes(),Rate):
                    speech_count +=1
                    
                chunks.append(chunk)
                if len(chunks) == doa_chunks:
                    if speech_count > (doa_chunks / 2):
                        frames = np.concatenate(chunks)
                        direction = find_direction_gccphat(frames,fs=mic.rate,tau_mic_dist=max_time)
                        print(int(direction))
                        
                    speech_count = 0
                    chunks = []
                    
    except KeyboardInterrupt:
        pass


    
def main():
    #mic_array_test()
    vad_mic_test()

    
    
if __name__ == "__main__":
    main()