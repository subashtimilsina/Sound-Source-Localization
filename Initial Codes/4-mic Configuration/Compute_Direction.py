import numpy as np
import wave
from scipy.io import wavfile
from Input_Devices import *


def mic_array_test():
    import signal
    import time
    
    distance = 0.0568
    max_time = distance/343.2
    
    is_quit = threading.Event()
    
    def signal_handler(sig, num):
        is_quit.set()
        print('Exited')
    
    signal.signal(signal.SIGINT, signal_handler)
    
    with MicArray(channels=4) as mic:
        for chunk in mic.read_mic_data():
            direction = find_direction_gccphat(chunk,fs=mic.rate,tau_mic_dist=max_time)
            print(int(direction))
            
            if is_quit.is_set():
                break



def main():
    mic_array_test()
    
if __name__ == "__main__":
    main()