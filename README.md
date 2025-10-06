# Laboratorio 3-Analisis espectral de la voz

## Resumen
Para este laboratorio se busca procesar señales de tres voces masculinas y tres voces femeninas, siendo analizadas por medio de la transformada de Fourier y extrayendo los datos más relevantes como las frecuenncias, brillo, jitter y shimmer. Permitiendo así identificar las diferencias y el comportamiento de la voz.

## Parte A

![Doc1_page-0001](https://github.com/user-attachments/assets/ba962810-98ca-45c6-a129-19ca0bb56756)

Para la primera sección del laboratorio se realizó la captura de voz de tres mujeres y tres hombres utilizando una grabadora de voz con una frecuancia de muestreo de 48 KHz. A los seis participantes se les pide que decir la misma frase: "Quiero irme a mi casa y comer una hamburguesa". Una vez adquiridas las señales se guarda cada archivo en formato **.wav** y se importan las señales a python usando el siguiente código:

```python 
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.io import wavfile

    #MUJERES
    fs1, signal1 = wavfile.read('/content/Punto A mujer 1.wav')
    fs2, signal2 = wavfile.read('/content/Punto-A-mujer-2.wav')
    fs3, signal3 = wavfile.read('/content/Punto-A-mujer-3.wav')

    #HOMBRES
    fs4, signal4 = wavfile.read('/content/Punto A hombre 1.wav')
    fs5, signal5 = wavfile.read('/content/Punto A hombre 2.wav')
    fs6, signal6 = wavfile.read('/content/Parte-A-hombre-3.wav')
```

Una vez los archivos están cargados en python se grafican las señales en función del tiempo: 

```python
 duration = len(signal1) / fs1
    time1 = np.arange(0, duration, 1/fs1)
    duration = len(signal2) / fs1
    time2 = np.arange(0, duration, 1/fs2)
    duration = len(signal3) / fs1
    time3 = np.arange(0, duration, 1/fs3)

    duration = len(signal4) / fs1
    time4 = np.arange(0, duration, 1/fs4)
    duration = len(signal5) / fs1
    time5 = np.arange(0, duration, 1/fs5)
    duration = len(signal6) / fs1
    time6 = np.arange(0, duration, 1/fs6)

fig, axs = plt.subplots(6, 1, figsize=(14, 10), sharex=False)

# --- Graficar cada señal en su propio eje ---
axs[0].plot(time1, signal1)
axs[0].set_title("Mujer 1")

axs[1].plot(time2, signal2)
axs[1].set_title("Mujer 2")

axs[2].plot(time3, signal3)
axs[2].set_title("Mujer 3")

axs[3].plot(time4, signal4)
axs[3].set_title("Hombre 1")

axs[4].plot(time5, signal5)
axs[4].set_title("Hombre 2")

axs[5].plot(time6, signal6)
axs[5].set_title("Hombre 3")


plt.tight_layout()
plt.show()
```

Obteniendo las señales de la siguente manera:

<img width="1388" height="990" alt="image" src="https://github.com/user-attachments/assets/4db7caf5-d2e1-4017-848b-3c222738f955" />

Después se calcula la transformada de Fourier de cada señal y se grafica el espectro de magnitudes frecuenciales de la siguiente manera:

```python
import numpy as np
import matplotlib.pyplot as plt

seniales = [
    ("Mujer 1", signal1, fs1),
    ("Mujer 2", signal2, fs2),
    ("Mujer 3", signal3, fs3),
    ("Hombre 1", signal4, fs4),
    ("Hombre 2", signal5, fs5),
    ("Hombre 3", signal6, fs6)
]


# --- Graficar FFT de cada señal ---
plt.figure(figsize=(12, 20))

for i, (titulo, señal, fs) in enumerate(seniales, 1):

    N = len(señal)
    freqs = np.fft.rfftfreq(N, 1/fs)
    espectro = np.abs(np.fft.rfft(señal))

    plt.subplot(6, 1, i)
    plt.semilogx(freqs, espectro)
    plt.title(titulo)
    plt.ylabel('Amplitud')
    plt.grid(True)

    # Aquí limitamos el eje X a que empiece desde 10^1 (10 Hz)
    plt.xlim(left=10)

plt.xlabel('Frecuencia (Hz)')
plt.tight_layout()
plt.show()
```

Obteniendo los siguientes gráficos:

<img width="1189" height="1989" alt="image" src="https://github.com/user-attachments/assets/a940dee0-2413-4872-998c-19612f649cb2" />

Finalmente se identifican algunas características de cada señal como: Frecuencia fundamental, frecuencia media, brillo e intensidad, esto se implemento en el siguiente código:

```python
import numpy as np
def caracteristicas(señal, fs):
    # Si es estéreo, convertir a mono
    if señal.ndim > 1:
        señal = señal.mean(axis=1)

    # FFT solo en frecuencias positivas
    N = len(señal)
    freqs = np.fft.rfftfreq(N, 1/fs)
    espectro = np.abs(np.fft.rfft(señal))

    # a) Frecuencia fundamental = primer pico distinto de DC (descartamos muy bajas)
    espectro[0] = 0  # quitar DC
    idx = np.argmax(espectro)  # máximo pico del espectro
    Frecuencia = freqs[idx]

    # b) Frecuencia media (centroide espectral)
    fmedia = np.sum(freqs * espectro) / np.sum(espectro)

    # c) Brillo (igual que centroide espectral aquí)
    brillo = fmedia

    # d) Intensidad (energía)
    energia = np.sum(señal.astype(float)**2)

    return Frecuencia, fmedia, brillo, energia


# --- Lista de señales ya cargadas ---
seniales = [
    ("Mujer 1", signal1, fs1),
    ("Mujer 2", signal2, fs2),
    ("Mujer 3", signal3, fs3),
    ("Hombre 1", signal4, fs4),
    ("Hombre 2", signal5, fs5),
    ("Hombre 3", signal6, fs6)
]

# --- Calcular y mostrar resultados ---
for nombre, s, fs in seniales:
    Frecuencia, fmedia, brillo, energia = caracteristicas(s, fs)
    print(f"{nombre}:")
    print(f"  Frecuencia fundamental: {Frecuencia:.2f} Hz")
    print(f"  Frecuencia media:       {fmedia:.2f} Hz")
    print(f"  Brillo:                 {brillo:.2f} Hz")
    print(f"  Intensidad (energía):   {energia:.2e}\n")
```

Teniendo como resultado las siguientes características:


* **Mujer 1:**

  Frecuencia fundamental: 224.20 Hz
  
  Frecuencia media:       4471.64 Hz
  
  Brillo:                 4471.64 Hz
  
  Intensidad (energía):   6.79e+11
  

* **Mujer 2:**
  
  Frecuencia fundamental: 218.46 Hz
  
  Frecuencia media:       4093.64 Hz
  
  Brillo:                 4093.64 Hz
  
  Intensidad (energía):   5.02e+11

* **Mujer 3:**
  
  Frecuencia fundamental: 199.22 Hz
  
  Frecuencia media:       2942.16 Hz
  
  Brillo:                 2942.16 Hz
  
  Intensidad (energía):   1.66e+12

* **Hombre 1:**
  
  Frecuencia fundamental: 225.80 Hz
  
  Frecuencia media:       3563.68 Hz
  
  Brillo:                 3563.68 Hz
  
  Intensidad (energía):   1.23e+12
  
* **Hombre 2:**
  
  Frecuencia fundamental: 122.40 Hz
  
  Frecuencia media:       3634.31 Hz
  
  Brillo:                 3634.31 Hz
  
  Intensidad (energía):   8.75e+11

* **Hombre 3:**
  
  Frecuencia fundamental: 242.96 Hz
  
  Frecuencia media:       3548.97 Hz
  
  Brillo:                 3548.97 Hz
  
  Intensidad (energía):   1.56e+12

# PARTE B 

![Doc1_page-0002](https://github.com/user-attachments/assets/4526d8c3-ac8a-4a9b-a789-9b63ea954c89)

Para este segunda parte se divide en dos apartados, para el primer apartado se selecciono la grabación de un hombre y de una mujer para aplicaar un filtro pasa-banda en un rango de 80-400 Hz para el hombre y 150-500 Hz paa la mujer para eliminar el ruido, evidenciandose así en el siguiente codigo:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Función filtro pasa-banda
def filtro_pasabanda(x, fs, lowcut, highcut):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, [low, high], btype='band')  # orden 4
    return filtfilt(b, a, x)

# Mujer 3 (150–500 Hz)
mujer1_filtrada = filtro_pasabanda(signal1, fs1, 150, 500)
mujer2_filtrada = filtro_pasabanda(signal2, fs2, 150, 500)
mujer3_filtrada = filtro_pasabanda(signal3, fs3, 150, 500)

# Hombre 3 (80–400 Hz)
hombre1_filtrada = filtro_pasabanda(signal4, fs6, 80, 400)
hombre2_filtrada = filtro_pasabanda(signal5, fs5, 80, 400)
hombre3_filtrada = filtro_pasabanda(signal6, fs6, 80, 400)

# Graficar resultados
plt.figure(figsize=(12,6))

plt.subplot(2,1,1)
plt.plot(signal3, label="Original")
plt.plot(mujer3_filtrada, label="Filtrada")
plt.title("Mujer 3 (150–500 Hz)")
plt.legend()

plt.subplot(2,1,2)
plt.plot(signal6, label="Original")
plt.plot(hombre3_filtrada, label="Filtrada")
plt.title("Hombre 3 (80–400 Hz)")
plt.legend()

plt.tight_layout()
plt.show()
```
Evidenciando así la siguiente grafica:

<img width="1190" height="590" alt="image" src="https://github.com/user-attachments/assets/c1d466e2-b2e0-4272-a8a4-71996f17b0eb" />

Para el segundo apartado se aplicaron dos medetos de medición siendo uno jitter donde se detectó los periodos de vibración de una señal, se calculo los periodos TI de la señal de las voces obteniendo el jitter absoluto y calculando el jitter relativo; y siendo el segundo metodo, shimmer donde se detectó los picos de amplitud Ai en cada ciclo obteniendo así shimmer absoluto y calculando el shimmer realtivo, haciendo este proceso en las seis grabaciones, obteniendo así el siguiete codigo.

Siendo el primer codigo el de jitter:

```python
import numpy as np

def medir_jitter(señal, fs):
    # Convertir a mono si es necesario
    if señal.ndim > 1:
        señal = señal.mean(axis=1)

    # Normalizar para estabilidad
    señal = señal / np.max(np.abs(señal))

    # 1. Detectar cruces por cero positivos (de negativo a positivo)
    cruces = np.where((señal[:-1] < 0) & (señal[1:] >= 0))[0]

    # 2. Calcular periodos Ti (en segundos)
    tiempos_cruce = cruces / fs
    periodos = np.diff(tiempos_cruce)  # T_i = t_(i+1) - t_i

    if len(periodos) < 2:
        return 0.0, 0.0, 0  # no hay suficientes ciclos

    # 3. Calcular jitter absoluto
    jitter_abs = np.mean(np.abs(np.diff(periodos)))  # |T_(i+1) - T_i|

    # 4. Calcular jitter relativo (%)
    T_prom = np.mean(periodos)
    jitter_rel = (jitter_abs / T_prom) * 100

    return jitter_abs, jitter_rel, len(periodos)


# --- Usar señales filtradas (ya pasaron por filtro pasa-banda) ---

# Mujer 3 filtrada
jitter_abs_m1, jitter_rel_m1, N_m1 = medir_jitter(mujer1_filtrada, fs1)
jitter_abs_m2, jitter_rel_m2, N_m2 = medir_jitter(mujer2_filtrada, fs2)
jitter_abs_m3, jitter_rel_m3, N_m3 = medir_jitter(mujer3_filtrada, fs3)

# Hombre 3 filtrada
jitter_abs_h1, jitter_rel_h1, N_h1 = medir_jitter(hombre1_filtrada, fs4)
jitter_abs_h2, jitter_rel_h2, N_h2 = medir_jitter(hombre2_filtrada, fs5)
jitter_abs_h3, jitter_rel_h3, N_h3 = medir_jitter(hombre3_filtrada, fs6)

# --- Imprimir resultados ---
print(f"Mujer 1 (filtrada) - Jitter absoluto: {jitter_abs_m1*1000:.3f} ms, Jitter relativo: {jitter_rel_m1:.2f} % con {N_m1} periodos")
print(f"Mujer 2 (filtrada) - Jitter absoluto: {jitter_abs_m2*1000:.3f} ms, Jitter relativo: {jitter_rel_m2:.2f} % con {N_m2} periodos")
print(f"Mujer 3 (filtrada) - Jitter absoluto: {jitter_abs_m3*1000:.3f} ms, Jitter relativo: {jitter_rel_m3:.2f} % con {N_m3} periodos")

print(f"Hombre 1 (filtrada) - Jitter absoluto: {jitter_abs_h1*1000:.3f} ms, Jitter relativo: {jitter_rel_h1:.2f} % con {N_h1} periodos")
print(f"Hombre 2 (filtrada) - Jitter absoluto: {jitter_abs_h2*1000:.3f} ms, Jitter relativo: {jitter_rel_h2:.2f} % con {N_h2} periodos")
print(f"Hombre 3 (filtrada) - Jitter absoluto: {jitter_abs_h3*1000:.3f} ms, Jitter relativo: {jitter_rel_h3:.2f} % con {N_h3} periodos")

```
Obteniendo los siguientes resultados:

Mujer 1 (filtrada) - Jitter absoluto: 0.693 ms, Jitter relativo: 19.82 % con 926 periodos

Mujer 2 (filtrada) - Jitter absoluto: 0.810 ms, Jitter relativo: 22.75 % con 969 periodos

Mujer 3 (filtrada) - Jitter absoluto: 0.812 ms, Jitter relativo: 22.52 % con 1159 periodos

Hombre 1 (filtrada) - Jitter absoluto: 1.633 ms, Jitter relativo: 38.14 % con 815 periodos

Hombre 2 (filtrada) - Jitter absoluto: 1.652 ms, Jitter relativo: 36.75 % con 853 periodos

Hombre 3 (filtrada) - Jitter absoluto: 1.548 ms, Jitter relativo: 35.11 % con 686 periodos


El siguiente codigo es el de shimmer:

```python
import numpy as np
from scipy.signal import find_peaks

def medir_shimmer(señal, fs):
    # Asegurar que la señal sea mono
    if señal.ndim > 1:
        señal = señal.mean(axis=1)

    # Normalizar señal para facilitar detección de picos
    señal = señal / np.max(np.abs(señal))

    # Detección de picos positivos (amplitudes A_i)
    indices_picos, _ = find_peaks(señal, distance=fs*0.002)  # mínimo 2ms entre picos
    amplitudes = señal[indices_picos]

    if len(amplitudes) < 2:
        return 0.0, 0.0, 0  # no hay suficientes ciclos

    # 1. Shimmer absoluto
    shimmer_abs = np.mean(np.abs(np.diff(amplitudes)))

    # 2. Shimmer relativo (%)
    A_prom = np.mean(amplitudes)
    shimmer_rel = (shimmer_abs / A_prom) * 100

    return shimmer_abs, shimmer_rel, len(amplitudes)


# --- Calcular shimmer para señales filtradas ---

# Mujer filtrada
shimmer_abs_m1, shimmer_rel_m1, N_amp_m1 = medir_shimmer(mujer1_filtrada, fs1)
shimmer_abs_m2, shimmer_rel_m2, N_amp_m2 = medir_shimmer(mujer2_filtrada, fs2)
shimmer_abs_m3, shimmer_rel_m3, N_amp_m3 = medir_shimmer(mujer3_filtrada, fs3)

# Hombre filtrada
shimmer_abs_h1, shimmer_rel_h1, N_amp_h1 = medir_shimmer(hombre1_filtrada, fs4)
shimmer_abs_h2, shimmer_rel_h2, N_amp_h2 = medir_shimmer(hombre2_filtrada, fs5)
shimmer_abs_h3, shimmer_rel_h3, N_amp_h3 = medir_shimmer(hombre3_filtrada, fs6)

# --- Imprimir resultados ---
print(f"Mujer 1 (filtrada) - Shimmer absoluto: {shimmer_abs_m1:.5f}, Shimmer relativo: {shimmer_rel_m1:.2f} % con {N_amp_m1} amplitudes")
print(f"Mujer 2 (filtrada) - Shimmer absoluto: {shimmer_abs_m2:.5f}, Shimmer relativo: {shimmer_rel_m2:.2f} % con {N_amp_m2} amplitudes")
print(f"Mujer 3 (filtrada) - Shimmer absoluto: {shimmer_abs_m3:.5f}, Shimmer relativo: {shimmer_rel_m3:.2f} % con {N_amp_m3} amplitudes")

print(f"Hombre 1 (filtrada) - Shimmer absoluto: {shimmer_abs_h1:.5f}, Shimmer relativo: {shimmer_rel_h1:.2f} % con {N_amp_h1} amplitudes")
print(f"Hombre 2 (filtrada) - Shimmer absoluto: {shimmer_abs_h2:.5f}, Shimmer relativo: {shimmer_rel_h2:.2f} % con {N_amp_h2} amplitudes")
print(f"Hombre 3 (filtrada) - Shimmer absoluto: {shimmer_abs_h3:.5f}, Shimmer relativo: {shimmer_rel_h3:.2f} % con {N_amp_h3} amplitudes")

```
Obteniendo los siguientes resultados:

Mujer 1 (filtrada) - Shimmer absoluto: 0.07070, Shimmer relativo: 28.87 % con 926 amplitudes

Mujer 2 (filtrada) - Shimmer absoluto: 0.06269, Shimmer relativo: 32.96 % con 968 amplitudes

Mujer 3 (filtrada) - Shimmer absoluto: 0.06084, Shimmer relativo: 26.27 % con 1172 amplitudes

Hombre 1 (filtrada) - Shimmer absoluto: 0.12133, Shimmer relativo: 66.74 % con 920 amplitudes

Hombre 2 (filtrada) - Shimmer absoluto: 0.10869, Shimmer relativo: 82.31 % con 1027 amplitudes

Hombre 3 (filtrada) - Shimmer absoluto: 0.09516, Shimmer relativo: 45.01 % con 782 amplitudes

# PARTE C
![Doc1_page-0003](https://github.com/user-attachments/assets/2fb42ece-e0c4-4f45-8b49-7fe7b152ed33)

Para esta ultima parte se realiza una comparacion de los resultados de las señales de voces masculinas y femeninas para responder las siguientes preguntas:

### 1.  ¿Qué diferencias se observan en la frecuencia fundamental?


### 2. ¿Qué otras diferencias notan en términos de brillo, media o intensidad?


### Conclusiones 


### Importancia clinica 

La importancia clinica de los metodos de medición jitter y shimmer estos son muy utiles para decterctar disfonias o patologias en la voz, también se puede identifficar problemas antes de la calidad vocal de la persona, esto sirve para detectar si hay un avance en las terapias que se le realizan al paciente ya que el jitter mide la variabilidad de la frecuencia y el shimmer la variabilidad de la amplitud; este se ve más frecuente en otorrinolaringologia y foniatria.



  
