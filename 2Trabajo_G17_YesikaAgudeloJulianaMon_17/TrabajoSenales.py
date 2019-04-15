import numpy as np
import matplotlib.pyplot as plt
import LinearFIR
from scipy.stats import kurtosis
from scipy import signal 
from numpy import dtype
from scipy.stats import linregress

#Función para cargar las señales 
def CargarSenal(ruta,delimitador,eliminar,columnas):
    data=np.loadtxt(ruta,delimiter=delimitador,skiprows=eliminar,usecols=columnas)
    return data

my_data_1=CargarSenal('P1_RAWEEG_2018-11-15_Electrobisturí1_3min.txt',',',6,[1,2,3,4,5,6,7,8])
# my_data_2=CargarSenal('P1_RAWEEG_2018-11-15_Electrobisturí2_2min.txt',',',6,[1,2,3,4,5,6,7,8])
# my_data_3=CargarSenal('P1_RAWEEG_2018-11-15_FinProcedimiento_53min.txt',',',6, [1,2,3,4,5,6,7,8])
# my_data_4=CargarSenal('P1_RAWEEG_2018-11-15_OjosCerrados_2min.txt',',',6, [1,2,3,4,5,6,7,8])

#Filtrado de las señales 
filtro1=LinearFIR.eegfiltnew(my_data_1, 250, 1, 50, 0, 0)
# filtro2=LinearFIR.eegfiltnew(my_data_2, 250, 1, 50, 0, 0)
# filtro3=LinearFIR.eegfiltnew(my_data_3, 250, 1, 50, 0, 0)
# filtro4=LinearFIR.eegfiltnew(my_data_4, 250, 1, 50, 0, 0)

#Creacion del vector de tiempo 
tiempo_f1=np.arange(0,filtro1.shape[0]/250,1/250)
# tiempo_f2=np.arange(0,filtro2.shape[0]/250,1/250)
# tiempo_f3=np.arange(0,filtro3.shape[0]/250,1/250)
# tiempo_f4=np.arange(0,filtro4.shape[0]/250,1/250)

#Creación del nivel DC para la visualización 
DC=[0,150,250,350,450,550,650,750]
# plt.plot(filtro1+DC) 
# plt.plot(filtro2+DC)
# plt.plot(filtro3+DC) 
# plt.plot(filtro4+DC) 
# plt.xlabel('Time [s]')
# plt.show()

def segmentacion(filtro,tiempo,fs,epoca):
    
    Dim=filtro.shape
    Residuo=Dim[0]%(fs*epoca)
    Total_Time=Dim[0]/fs
    Segmentacion_Data=np.split(filtro[0:Dim[0]-Residuo,:], int(Total_Time//epoca))
    time=np.split(tiempo[0:Dim[0]-Residuo], int(Total_Time//epoca))
    
    return np.array(Segmentacion_Data),np.array(time)

S1_array,time1=segmentacion(filtro1,tiempo_f1,250,2)
# S2_array,time2=segmentacion(filtro2,tiempo_f2,250,2)
# S3_array,time3=segmentacion(filtro3,tiempo_f3,250,2)
# S4_array,time4=segmentacion(filtro4,tiempo_f4,250,2)

#PRIMER MÉTODO: RECHAZO POR VALORES EXTREMOS 

#Se hace una función para hallar los maximos de los segmentos y minimos de los segmentos que no cumplen las condiciones y se rechazan 
def Rechazo (umbralmaximo,umbralminimo,SegmentoArray):
    Vunos=np.ones((SegmentoArray.shape[0],1))
    valormaximo=SegmentoArray.max(axis=1)
    valorminimo=SegmentoArray.min(axis=1)
     
    for i in range(0,8):
        index=np.where(valormaximo[:,i]>umbralmaximo)
        Vunos[index]=0
        index=np.where(valorminimo[:,i]<umbralminimo)
        Vunos[index]=0
          
    Rechazo=Vunos.transpose()[0]   
    Data=SegmentoArray[Rechazo==1,:,:]
    size=Data.shape
    Signal=np.zeros((size[0]*size[1],size[2]),dtype=np.int)
    for i in range(0,size[2]):
        Signal[:,i]=Data[:,:,i].ravel()
         
    return Signal 

SenalSeg1_Valores=Rechazo(75,-75,S1_array)
#SenalSeg2_Valores=Rechazo(75,-75,S2_array)
#SenalSeg3_Valores=Rechazo(75,-75,S3_array)
#SenalSeg4_Valores=Rechazo(75,-75,S4_array)
  
# plt.plot(SenalSeg4_Valores+DC)
# plt.title('RECHAZO POR VALORES EXTREMOS')
# plt.xlabel('Time [s]')
# plt.show()

#SEGUNDO MÉTODO: RECHAZO POR TENDENCIA LINEAL 
 
def TENDENCIA_LINEAL(SegmentoArray,umbralmax,umbralmin,time):
    Dim=SegmentoArray.shape
    Unos=np.ones((Dim[0],1),dtype=np.int)
    Pendiente=np.zeros((Dim[0],Dim[2]),dtype=np.float32)
      
    for j in range (0,Dim[2]):
        for i in range(0,Dim[0]):
            Regresion=np.polyfit(time[j],SegmentoArray[i,:,j],1)
            Pendiente[i,j]=Regresion[0]
  
      
                 
    for k in range(0,Dim[2]):
        index=np.where(Pendiente[:,k]>umbralmax)
        Unos[index]=0
        index=np.where(Pendiente[:,k]<umbralmin)
        Unos[index]=0
   
    Tendencia=Unos.transpose()[0] 
    Data=SegmentoArray[Tendencia==1,:,:]
    size=Data.shape
    Signal=np.zeros((size[0]*size[1],size[2]),dtype=np.int)
    for i in range(0,size[2]):
        Signal[:,i]=Data[:,:,i].ravel()
   
    return Signal
   
# SenalSeg1_Regresion=TENDENCIA_LINEAL(S1_array,4,-4,time1)
# # SenalSeg2_Regresion=TENDENCIA_LINEAL(S2_array,4,-4,time2)
# # SenalSeg3_Regresion=TENDENCIA_LINEAL(S3_array,4,-4,time3)
# # SenalSeg4_Regresion=TENDENCIA_LINEAL(S4_array,4,-4,time4)
# plt.title('RECHAZO POR TENDENCIA LINEAL')
# plt.plot(SenalSeg1_Regresion+DC)
# plt.show()
 
#TERCER MÉTODO: RECHAZO POR IMPROBABILIDAD
 
#Se crea la función de Kurtosis para hallar los maximos y los minimos de los segmentos que no cumplen con las condiciones y se rechazan 
 
def Kurtosis (umbralmaximo,umbralminimo,SegmentoArray):
    Unos=np.ones((SegmentoArray.shape[0],1))
    Kurt1=kurtosis(SegmentoArray,axis=1)
          
    for i in range(0,8):
        index=np.where(Kurt1[:,i]>umbralmaximo)
        Unos[index]=0
        index=np.where(Kurt1[:,i]<umbralminimo)
        Unos[index]=0
            
    Rechazo=Unos.transpose()[0]   
    Data=SegmentoArray[Rechazo==1,:,:]
    size=Data.shape
    Signal=np.zeros((size[0]*size[1],size[2]),dtype=np.int)
    for i in range(0,size[2]):
        Signal[:,i]=Data[:,:,i].ravel()
           
    return Signal 
  
# SenalSeg1_Kurtosis=Kurtosis(3,-3,S1_array)
# SenalSeg2_Kurtosis=Kurtosis(3,-3,S2_array)
# SenalSeg3_Kurtosis=Kurtosis(3,-3,S3_array)
# SenalSeg4_Kurtosis=Kurtosis(3,-3,S4_array)
# plt.plot(SenalSeg4_Kurtosis+DC)
# plt.title("RECHAZO POR IMPROBABILIDAD")
# plt.xlabel('Time [s]')
# plt.show()
 
#CUARTO MÉTODO: RECHAZO POR PATRÓN ESPECTRAL 
 
def PATRON_ESPECTRAL(SegmentoArray,frec,umbralmax,umbralmin):
    Dim=SegmentoArray.shape 
    Unos=np.ones((Dim[0],1),dtype=np.int)
    Pmax=np.zeros((Dim[0],Dim[2]),dtype=np.float32)
    Pmin=np.zeros((Dim[0],Dim[2]),dtype=np.float32)
     
     
    for j in range (0,Dim[2]):
        for i in range(0,Dim[0]):
            F,Pxx= signal.welch(SegmentoArray[i,:,j],frec,"hanning",Dim[1])
            media=np.mean(Pxx)
            Pot=Pxx-media
            Pmax[i,j]=Pot.max()
            Pmin[i,j]=Pot.min()  
     
    for k in range(0,Dim[2]):
        index=np.where(Pmax[:,k]>umbralmax)
        Unos[index]=0
        index=np.where(Pmin[:,k]<umbralmin)
        Unos[index]=0
     
     
    Patron=Unos.transpose()[0] 
    Data=SegmentoArray[Patron==1,:,:]
    size=Data.shape
    Signal=np.zeros((size[0]*size[1],size[2]),dtype=np.int)
    for i in range(0,size[2]):
        Signal[:,i]=Data[:,:,i].ravel()
    return Signal
 
#SenalSeg1_Patron= PATRON_ESPECTRAL(S1_array, 250,200,-5)
#SenalSeg2_Patron= PATRON_ESPECTRAL(S2_array, 250,200,-5)
#SenalSeg3_Patron= PATRON_ESPECTRAL(S3_array, 250,200,-5)
#SenalSeg4_Patron= PATRON_ESPECTRAL(S4_array, 250,200,-5)
#plt.plot(SenalSeg1_Patron+DC)
#plt.xlabel('Time [s]')
#plt.title('RECHAZO POR PATRÓN ESPECTRAL')
#plt.show()



# #ESTRATEGIA 1
# S1_array,time1=segmentacion(filtro1,tiempo_f1,250,2)
# SenalSeg1_Valores=Rechazo(75,-75,S1_array)
# Valores_Extremos1,time_1=segmentacion(SenalSeg1_Valores, tiempo_f1, 250, 2)
# TendenciaLineal=TENDENCIA_LINEAL(Valores_Extremos1, 4, -4, time_1)
# TendenciaLineal1,time_2=segmentacion(TendenciaLineal, tiempo_f1, 250, 2)
# Improbabilidad=Kurtosis(3,-3,TendenciaLineal1)
# Improbabilidad1,time_3=segmentacion(Improbabilidad, time_2, 250, 2)
# PatronEspectral=PATRON_ESPECTRAL(Improbabilidad1, 250, 200,-5)
#  
#   
# Frec,Pxx= signal.welch(PatronEspectral[:,0],250, "hanning",len(PatronEspectral[:,0])/8)
# plt.figure(figsize=(10,6))
# plt.semilogy(Frec,Pxx)
# plt.title('Estrategia 1')
# plt.xlabel('Time [S]')
# plt.show()


# #ESTRATEGIA 2
# S1_array,time1=segmentacion(filtro1,tiempo_f1,250,2)
# SenalSeg1_Valores=Rechazo(75,-75,S1_array)
# Valores_Extremos1,time_1=segmentacion(SenalSeg1_Valores, tiempo_f1, 250, 2)
# TendenciaLineal=TENDENCIA_LINEAL(Valores_Extremos1, 4, -4, time_1)
# TendenciaLineal1,time_2=segmentacion(TendenciaLineal, tiempo_f1, 250, 2)
# Improbabilidad=Kurtosis(3,-3,TendenciaLineal1)
#  
#  
# Frec,Pxx= signal.welch(Improbabilidad[:,0],250, "hanning",len(Improbabilidad[:,0])/8)
# plt.figure(figsize=(10,6))
# plt.semilogy(Frec,Pxx)
# plt.title('Estrategia 2')
# plt.xlabel('Time [S]')
# plt.show()


# #ESTRATEGIA 3
# S1_array,time1=segmentacion(filtro1,tiempo_f1,250,2)
# SenalSeg1_Valores=Rechazo(75,-75,S1_array)
# Valores_Extremos1,time_1=segmentacion(SenalSeg1_Valores, tiempo_f1, 250, 2)
# TendenciaLineal=TENDENCIA_LINEAL(Valores_Extremos1, 4, -4, time_1)
# 
# 
# Frec,Pxx= signal.welch(TendenciaLineal[:,0],250, "hanning",len(TendenciaLineal[:,0])/8)
# plt.figure(figsize=(10,6))
# plt.semilogy(Frec,Pxx)
# plt.title('Estrategia 3')
# plt.xlabel('Time [S]')
# plt.show()


# #ESTRATEGIA 3.1
# S1_array,time1=segmentacion(filtro1,tiempo_f1,250,2)
# SenalSeg1_Valores=Rechazo(75,-75,S1_array)
# 
# 
# Frec,Pxx= signal.welch(SenalSeg1_Valores[:,0],250, "hanning",len(SenalSeg1_Valores[:,0])/8)
# plt.figure(figsize=(10,6))
# plt.semilogy(Frec,Pxx)
# plt.title('Estrategia 3.1')
# plt.xlabel('Time [S]')
# plt.show()


