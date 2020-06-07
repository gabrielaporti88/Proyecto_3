""" PROYECTO FINAL """
import os
import sys
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import scipy.signal as signal
from linearFIR import filter_design, mfreqz
from wavelet_filtering_29_5_18 import wthresh, thselect, wnoisest
from IPython import get_ipython


#%% 
# Preprocesamiento y acondicionamiento de la señal

"""  FUNCIONES  """

# Funcion para filrar una señal de audio de auscultación
def linealFir (senial, sro):
    
      
    # Filtro pasa bajas
    order, lowpass = filter_design(sro, locutoff = 0, hicutoff = 1000, revfilt = 0);
    senal_lp = signal.filtfilt(lowpass, 1, senial);
        
    # Filtro pasa altas
    order, highpass = filter_design(sro, locutoff = 100, hicutoff = 0, revfilt = 1);
    senal_hp = signal.filtfilt(highpass, 1, senal_lp);
    
        
    return (senal_hp)

def fil_Wavelet(signal):
    LL = int(np.floor(np.log2(signal.shape[0])));   # Nivel de descomposición
    
    coeff = pywt.wavedec( signal, 'db6', level=LL );  # Transformada discreta de Wavelet

    thr = thselect(coeff);    # Umbral  (universal)
    coeff_t = wthresh(coeff,thr);

    x_rec = pywt.waverec( coeff_t, 'db6');  # transformada inversa de Wavelet

    x_rec = x_rec[0:signal.shape[0]];  

    x_filt = np.squeeze(signal - x_rec);    # Se le resta a la señal original, la señal obtenida con wavelet
    plt.plot(signal[0:1500],label='Original')
    plt.plot(x_rec[0:1500],label='Umbralizada por Wavelet')

    
    plt.plot(x_filt[0:1500],label='Original - Umbralizada')
    plt.legend()    
    
     
    return (x_filt)

def preproces (signal, sro):
    
    tiempo=np.arange(0, len(signal)/sro, 1/sro)     # Se crea vector de tiempo
    
    senal_1= linealFir(signal, sro)     # se aplica el filtro lineal
    senal_2= fil_Wavelet(senal_1)       # Se aplica el filtro con wavelet
    
    get_ipython().run_line_magic('matplotlib', 'qt')
    plt.plot(tiempo, signal, label='Señal original' )
    plt.plot(tiempo, senal_2, label='Señal filtrada')
    plt.title('Pre procesamiento de la señal')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud ')
    plt.legend()
    
    
    
    return (senal_2)
    
    
def carga (audio, texto):
    yo, sro = librosa.load(audio)    # recibe ruta de archivo de audio
    file= open (texto)  # recibe ruta de archivo de texto
    txt= np.loadtxt(file) # carga archivo de texto
    t=pd.DataFrame(columns=('No. ciclo','Inicio (muestra)', 'Fin (muestra)','Estertores', 'Sibilancias'))   # se crea un data frame, que contendrá la informacion del paciente
    
    for i in range (np.shape(txt)[0]):
        ti= txt[i][0]*22050     # Inicio del ciclo dado en numero de muestra
        tf= txt[i][1]*22050     # Fin del ciclo dado en numero de muestra
        e= txt [i][2]           # estertores del paciente
        s= txt [i][3]           # Sibilancias del paciente
        t.loc[len(t)]=[i+1,ti,tf,e,s]   # Añade la informacion del paciente a un dataframe
                
    segmentos=[]
  
    for i in range(np.shape(txt)[0]):     # Se corta la señal en ada ciclo segun su inicio y su fin
        segmento=yo[int(t.loc[i][1]):int(t.loc[i][2])]
        segmentos.append(segmento)
    
    """for i in range (10):   # permite guardar un audio
        librosa.output.write_wav('audio'+str(i)+'.wav', segmentos[i], 22050)
    """
    
    return (t, segmentos)
    
    

#%%
# Procesamiento y extracción de características de la señal
def indices (senal, dataframe, fs):
    varianza= np.var(senal)
    maximo= np.amax(senal)
    minimo= np.amin(senal)
    rango= abs(maximo-minimo)
    s=0
    for i in range(len(senal)-1):    
        s= abs(senal[i]-senal[i+1]) + s
        
    a= 0
    b= 799
    c= 100
    sma_f=[]
    
    while (b <= len(senal)):
        sma=0
        for i in range(800):
            sma= abs(senal[i+a]-senal[i+1+a]) + sma
        
        a= a+c
        b= b+c
        sma_f.append(sma)
    print (a,b)
    sma=0
    #for i in senal[a:len(senal)-2]:
    #    for j in senal[a+1:len(senal)-1]:
    #       sma= abs(i-j) + sma
            
    #sma_f.append(sma)
    for i in range (a, len(senal)-1):
        sma= abs(senal[i]- senal[i+1])+ sma
    
    sma_f.append(sma)
    
    sma_fine=np.amax(sma_f)
    
    f, Pxx = signal.welch(senal ,fs,'hamming', 1048, 419,1048, scaling='density', return_onesided= False);
    spectrum=np.mean(Pxx)
    
    return (varianza, rango, s, sma_fine, spectrum)
                
    
    
    
    
#%%

# pruebas de codigo

data,senal_1= carga("/Usuario/Desktop/Señales/Respiratory_Sound_Database/audio_and_txt_files/102_1b1_Ar_sc_Meditron.wav", "/Usuario/Desktop/Señales/Respiratory_Sound_Database/audio_and_txt_files/102_1b1_Ar_sc_Meditron.txt")
indexx= indices(senal_1[0], data, 22050)




"""
path = "/Usuario/Desktop/Señales/Respiratory_Sound_Database/audio_and_txt_files"    # Recibe la ruta donde se encuentran los archivos de audio de interés
archivos = os.listdir( path )    # lista los archivos de la ruta recibida en path
 
audio=[]
texto=[]

for i in range (len(archivos)):     # Seprara los archivos de audio y los de texto
    if i%2 == 0:
        texto.append(archivos[i])
    else:
        audio.append(archivos[i])






for i in range(10):    # se cargan solo 10 señales para facilidad de trabajo
    yo, sro = librosa.load('/Usuario/Desktop/Señales/Respiratory_Sound_Database/audio_and_txt_files/'+ audio[i]) #se cargn los audios contenidos en archivos con sus frecuencias de muestreo
    signals.append(yo)
    sr.append(sro)
    
txt=[]
for i in range (10): # Se cargan los archivos .txt
    file= open ('/Usuario/Desktop/Señales/Respiratory_Sound_Database/audio_and_txt_files/'+ texto[i])
    data= np.loadtxt(file)
    txt.append(data)
"""



"""

sr=22050


hop_length = 512
n_fft = 2048
D = np.abs(librosa.stft(c, n_fft=n_fft,hop_length=hop_length))

librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear');
plt.colorbar();

DB = librosa.amplitude_to_db(D, ref=np.max)
librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log');
plt.colorbar(format='%+2.0f dB');


plt.plot(signals[2])
plt.plot(b)
    
"""
    
   
    

"""  


# tiempo=np.arange(0, len(signals[0])/sro, 1/sro)     # Se crea vector de tiempo
    
    
# t=pd.DataFrame(columns=('Paciente No','Inicio', 'Fin'))     # Se crea un data Frame para organizar los tiempos de inicio y fin de cada ciclo de cada paciente



for i in range(10):
    for j in range (np.shape(txt[i])[0]):
        ti= txt[i][j][0]*22050
        tf= txt[i][j][1]*22050
        
        t.loc[len(t)]=[i+1,ti,tf]
        
segmentos=[]

for j in range (10):     # Se segmenta cada señal de audio en cada ciclo segun su punto de inicio y fin
    
    for i in range(np.shape(txt[j])[0]):     #recorrer ciclos
        segmento=signals[j][int(t.loc[i][1]):int(t.loc[i][2])]
        segmentos.append(segmento)
    
for i in range (10):   # permite guardar un audio
    librosa.output.write_wav('audio'+str(i)+'.wav', segmentos[i], 22050)
        
        
# Filtrado
signal_hp=[]
signal_lp=[] 


 

# order, lowpass = filter_design(fs, locutoff = 0, hicutoff = 4100, revfilt = 0)

    

for i in range (10):
    librosa.output.write_wav('audio_filtrado'+str(i)+'.wav', signal_hp[i], 22050)
    

"""
    
    
    
    
