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
from scipy.stats import ttest_ind, mannwhitneyu
import seaborn as sns 
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
    #plt.plot(signal[0:1500],label='Original')
    #plt.plot(x_rec[0:1500],label='Umbralizada por Wavelet')

    
    #plt.plot(x_filt[0:1500],label='Original - Umbralizada')
    #plt.legend()    
    
     
    return (x_filt)

def preproces (signal, sro):
    
    
    #tiempo=np.arange(0, len(signal)/sro, 1/sro)     # Se crea vector de tiempo
    
    senal_1= linealFir(signal, sro)     # se aplica el filtro lineal
    senal_2= fil_Wavelet(senal_1)       # Se aplica el filtro con wavelet
    
    #get_ipython().run_line_magic('matplotlib', 'qt')
    #plt.plot(tiempo, signal, label='Señal original' )
    #plt.plot(tiempo, senal_2, label='Señal filtrada')
    #plt.title('Pre procesamiento de la señal')
    #plt.xlabel('Tiempo (s)')
    #plt.ylabel('Amplitud ')
    #plt.legend()
    
    return (senal_2)
    
    
def carga (audio, texto):
    yo, sro = librosa.load(audio)    # recibe ruta de archivo de audio
    file= open (texto)  # recibe ruta de archivo de texto
    txt= np.loadtxt(file) # carga archivo de texto
    t=pd.DataFrame(columns=('No. ciclo','Inicio (muestra)', 'Fin (muestra)','Estertores', 'Sibilancias'))   # se crea un data frame, que contendrá la informacion del paciente
    
    for i in range (np.shape(txt)[0]):
        ti= txt[i][0]*sro     # Inicio del ciclo dado en numero de muestra
        tf= txt[i][1]*sro     # Fin del ciclo dado en numero de muestra
        e= txt [i][2]           # estertores del paciente
        s= txt [i][3]           # Sibilancias del paciente
        t.loc[len(t)]=[i+1,ti,tf,e,s]   # Añade la informacion del paciente a un dataframe
                
    
    return (t, yo, sro)
    
def segmento (senal, dataframe):   #recibe el t que retorna carga
    segmentos=[]
  
    for i in range(np.shape(dataframe)[0]):     # Se corta la señal en ada ciclo segun su inicio y su fin
        segmento=senal[int(dataframe.loc[i][1]):int(dataframe.loc[i][2])]
        segmentos.append(segmento)
    
    """for i in range (10):   # permite guardar un audio
        librosa.output.write_wav('audio'+str(i)+'.wav', segmentos[i], 22050)
    """
    return (segmentos)

#%%
# Procesamiento y extracción de características de la señal
def indices (senal, fs):
    
    varianza= np.var(senal)    # Calculo de la varianza
    maximo= np.amax(senal)     # calculo del valor máximo de la senal
    minimo= np.amin(senal)     # Calculo del valor mínimo de la senal
    rango= abs(maximo-minimo)  # Calculo del rango de la senal
    s=0
    for i in range(len(senal)-1):    # Calculo del promedio movil de la senal
        s= abs(senal[i]-senal[i+1]) + s
        
    
    
    f, Pxx = signal.welch(senal ,fs,'hamming', 1024, 419,1024, scaling='density', return_onesided= False);
    spectrum=np.mean(Pxx)      # Calculo del promedio del espectro de la senal
    
    return (varianza, rango, s, spectrum)
                
    
    
    
    
#%%
"""
# pruebas de codigo
t, senal_1, sro= carga("/Usuario/Desktop/Señales/Respiratory_Sound_Database/audio_and_txt_files/102_1b1_Ar_sc_Meditron.wav", "/Usuario/Desktop/Señales/Respiratory_Sound_Database/audio_and_txt_files/102_1b1_Ar_sc_Meditron.txt")
senal_1_filtrada= preproces(senal_1, sro)

segmentos_1= segmento(senal_1_filtrada, t)
indexx= indices(segmentos_1[0], sro)
"""
#%%

info= pd.DataFrame(columns=('No. Paciente','No. ciclo','Inicio (muestra)', 'Fin (muestra)','Estertores', 'Sibilancias','Varianza', 'Rango','Promedio movil','Prom. Espectro'))

path = "/Usuario/Desktop/Señales/Respiratory_Sound_Database/audio_and_txt_files"    # Recibe la ruta donde se encuentran los archivos de audio de interés
archivos = os.listdir( path )    # lista los archivos de la ruta recibida en path
 
audio=[]
texto=[]

for i in range (len(archivos)):     # Seprara los archivos de audio y los de texto
    if i%2 == 0:
        texto.append(archivos[i])
    else:
        audio.append(archivos[i])


#data,segmentos= carga(path+'/'+audio[1], path+'/'+texto[1])
#info.append(data, ignore_index = True)
#pd.concat([data, info])
#info= pd.concat([info,data])


for i in range (len(audio)):
    
    data,senal,sro= carga(path+'/'+audio[i], path+'/'+texto[i])    # se cargan las senales de audio de la base de datos
    #info= pd.concat([info,data])
    
    senal_fil= preproces(senal,sro)        # Se filtran las senales de la base de datos
    
    segmentos= segmento(senal_fil, data)   # Se segmenta cada señal de la base...
    
    data_index= pd.DataFrame(columns= ('Varianza', 'Rango', 'Promedio movil', 'Prom. Espectro', 'No. Paciente'))
    for j in range (len(segmentos)):   # Calculo de los indices de cada ciclo de cada senal
        index= indices (segmentos[j], sro)
        new=pd.DataFrame({'Varianza': [index[0]], 'Rango':[index[1]],'Promedio movil':[index[2]], 'Prom. Espectro':[index[3]], 'No. Paciente':[i+1]}, columns=['Varianza', 'Rango', 'Promedio movil', 'Prom. Espectro', 'No. Paciente'])
        data_index=pd.concat([data_index, new])
        data_index.index=range(data_index.shape[0])
    
    data=pd.concat([data,data_index], axis=1, )
    info= pd.concat ([info,data])
    info.index=range(info.shape[0])    # presenta el dataFrame con todos los datos de todos los segmentos de todas las señales de la base de datos
    
#%%
# Estadistica descriptiva

estertor= info.sort_values(['Estertores','Sibilancias'])
estertor.index=range(estertor.shape[0]) 
estertor= estertor.drop(['Fin (muestra)', 'Inicio (muestra)', 'No. ciclo'], axis=1)  

estertor_grupo_mean= estertor.groupby(['Estertores', 'Sibilancias'],as_index=False).mean()    
estertor_grupo_var= estertor.groupby(['Estertores', 'Sibilancias'],as_index=False).var()   

#Ciclos sin estertores y sin sibilancias
estertor_0=estertor[(estertor['Estertores'] == 0) & (estertor['Sibilancias'] == 0)  ]

count,bin_edges = np.histogram(estertor_0['Varianza'])
estertor_0['Varianza'].plot(kind='hist',xticks=bin_edges, color='r')
plt.xlabel('Varianza')
plt.ylabel('No. de ciclos')
plt.title('Varianza para ciclos que NO presentan estertores')
plt.grid()
plt.show()

plt.figure()
count,bin_edges = np.histogram(estertor_0['Rango'])
estertor_0['Rango'].plot(kind='hist',xticks=bin_edges, color='m')
plt.xlabel('Rango')
plt.ylabel('No. de ciclos')
plt.title('Rango para ciclos que NO presentan estertores')
plt.grid()
plt.show()

plt.figure()
count,bin_edges = np.histogram(estertor_0['Promedio movil'])
estertor_0['Promedio movil'].plot(kind='hist',xticks=bin_edges)
plt.xlabel('Promedio movil')
plt.ylabel('No. de ciclos')
plt.title('Promedio movil para ciclos que NO presentan estertores')
plt.grid()
plt.show()

plt.figure()
count,bin_edges = np.histogram(estertor_0['Prom. Espectro'])
estertor_0['Prom. Espectro'].plot(kind='hist',xticks=bin_edges, color='g')
plt.xlabel('Prom. Espectro')
plt.ylabel('No. de ciclos')
plt.title('Promedio movil para ciclos que NO presentan estertores')
plt.grid()
plt.show()

# Ciclos con estertores
estertor_1=estertor[(estertor['Estertores'] == 1)]
plt.figure()
count,bin_edges = np.histogram(estertor_1['Varianza'])
estertor_1['Varianza'].plot(kind='hist',xticks=bin_edges)
plt.xlabel('Varianza')
plt.ylabel('No. ciclos')
plt.title('Varianza para ciclos que presentan estertores')
plt.grid()
plt.show()

plt.figure()
count,bin_edges = np.histogram(estertor_1['Rango'])
estertor_1['Rango'].plot(kind='hist',xticks=bin_edges, color='m')
plt.xlabel('Rango')
plt.ylabel('No. de ciclos')
plt.title('Rango para ciclos que presentan estertores')
plt.grid()
plt.show()

plt.figure()
count,bin_edges = np.histogram(estertor_1['Promedio movil'])
estertor_1['Promedio movil'].plot(kind='hist',xticks=bin_edges)
plt.xlabel('Promedio movil')
plt.ylabel('No. de ciclos')
plt.title('Promedio movil para ciclos que presentan estertores')
plt.grid()
plt.show()

plt.figure()
count,bin_edges = np.histogram(estertor_1['Prom. Espectro'])
estertor_1['Prom. Espectro'].plot(kind='hist',xticks=bin_edges, color='g')
plt.xlabel('Prom. Espectro')
plt.ylabel('No. de ciclos')
plt.title('Promedio movil para ciclos que presentan estertores')
plt.grid()
plt.show()

# Ciclos son sibilancias
sibilancias_1=estertor[(estertor['Sibilancias'] == 1)]
plt.figure()
count,bin_edges = np.histogram(sibilancias_1['Varianza'])
sibilancias_1['Varianza'].plot(kind='hist',xticks=bin_edges)
plt.xlabel('Varianza')
plt.ylabel('No. ciclos')
plt.title('Varianza para ciclos que presentan estertores')
plt.grid()
plt.show()

plt.figure()
count,bin_edges = np.histogram(sibilancias_1['Rango'])
sibilancias_1['Rango'].plot(kind='hist',xticks=bin_edges, color='m')
plt.xlabel('Rango')
plt.ylabel('No. de ciclos')
plt.title('Rango para ciclos que presentan estertores')
plt.grid()
plt.show()

plt.figure()
count,bin_edges = np.histogram(sibilancias_1['Promedio movil'])
sibilancias_1['Promedio movil'].plot(kind='hist',xticks=bin_edges)
plt.xlabel('Promedio movil')
plt.ylabel('No. de ciclos')
plt.title('Promedio movil para ciclos que presentan estertores')
plt.grid()
plt.show()

plt.figure()
count,bin_edges = np.histogram(sibilancias_1['Prom. Espectro'])
sibilancias_1['Prom. Espectro'].plot(kind='hist',xticks=bin_edges, color='g')
plt.xlabel('Prom. Espectro')
plt.ylabel('No. de ciclos')
plt.title('Promedio movil para ciclos que presentan estertores')
plt.grid()
plt.show()

#%%
# Pruebas de hipótesis
# Histohgramas para la varianza en los tres posibles casos para los ciclos
plt.subplot(3,1,1)
plt.hist(estertor_0['Varianza'],color='r', label='Sin sibilancias ni estertores')
plt.legend()
plt.title('Varianza')

plt.subplot(3,1,2)
plt.hist(estertor_1['Varianza'], color='c',label='Con estertores')
plt.legend()

plt.subplot(3,1,3)
plt.hist(sibilancias_1['Varianza'], color='m',label='Con sibilancias')
plt.legend()

# Comparacion para la varianza
print('Varianza entre ciclos con estertores y sin estertores ni sibilancias:')
print(mannwhitneyu(estertor_0['Varianza'], estertor_1['Varianza'] ))

print('Varianza entre ciclos con sibilancias y sin estertores ni sibilancias:')
print(mannwhitneyu(estertor_0['Varianza'], sibilancias_1['Varianza'] ))

print('Varianza entre ciclos con estertores y ciclos con sibilancias:')
print(mannwhitneyu(sibilancias_1['Varianza'], estertor_1['Varianza'] ))

# comparacion para el rango
print('Rango entre ciclos con estertores y sin estertores ni sibilancias:')
print(mannwhitneyu(estertor_0['Rango'], estertor_1['Rango'] ))

print('Rango entre ciclos con sibilancias y sin estertores ni sibilancias:')
print(mannwhitneyu(estertor_0['Rango'], sibilancias_1['Rango'] ))

print('Rango entre ciclos con estertores y ciclos con sibilancias:')
print(mannwhitneyu(sibilancias_1['Rango'], estertor_1['Rango'] ))

#comparacion para el promedio movil
print('Promedio movil entre ciclos con estertores y sin estertores ni sibilancias:')
print(mannwhitneyu(estertor_0['Promedio movil'], estertor_1['Promedio movil'] ))

print('Promedio movil entre ciclos con sibilancias y sin estertores ni sibilancias:')
print(mannwhitneyu(estertor_0['Promedio movil'], sibilancias_1['Promedio movil'] ))

print('Promedio movil entre ciclos con estertores y ciclos con sibilancias:')
print(mannwhitneyu(sibilancias_1['Promedio movil'], estertor_1['Promedio movil'] ))

# comparacion para el promedio del espectro
print('Promedio de espectro entre ciclos con estertores y sin estertores ni sibilancias:')
print(mannwhitneyu(estertor_0['Prom. Espectro'], estertor_1['Promedio movil'] ))

print('Promedio de espectro entre ciclos con sibilancias y sin estertores ni sibilancias:')
print(mannwhitneyu(estertor_0['Prom. Espectro'], sibilancias_1['Promedio movil'] ))

print('Promedio de espectro entre ciclos con estertores y ciclos con sibilancias:')
print(mannwhitneyu(sibilancias_1['Prom. Espectro'], estertor_1['Promedio movil'] ))


