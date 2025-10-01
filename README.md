# Laboratorio 3-Analisis-espectral-de-la-voz

## Resumen
Para este laboratorio se busca procesar señales de tres voces masculinas y tres voces femeninas, siendo analizadas por medio de la transformada de Fourier y extrayendo los datos más relevantes como las frecuenncias, brillo, jitter y shimmer. Permitiendo así identificar las diferencias y el comportamiento de la voz.

## Parte A
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

<img width="1388" height="990" alt="image" src="https://github.com/user-attachments/assets/a6daeb62-76c0-4857-8167-fb7af97a4b48" />

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

plt.xlabel('Frecuencia (Hz)')
plt.tight_layout()
plt.show()
```

Obteniendo los siguientes gráficos:

<img width="1189" height="1989" alt="image" src="https://github.com/user-attachments/assets/0e548c11-0c89-44fb-b091-c7034327752e" />

Finalmente se identifican algunas características de cada señal como: Frecuencia fundamental, frecuencia media, brillo e intensidad, esto se implemento en el siguiente código:

```python
import numpy as np

def caracteristicas(señal, fs):
    if señal.ndim > 1:   # convertir a mono si es estéreo
        señal = señal.mean(axis=1)

    # FFT
    N = len(señal)
    freqs = np.fft.rfftfreq(N, 1/fs)
    espectro = np.abs(np.fft.rfft(señal))

    # a. Frecuencia fundamental = pico más bajo distinto de DC
    idx = np.argmax(espectro[1:]) + 1
    Frecuencia = freqs[idx]

    # b. Frecuencia media (ponderada por espectro)
    fmedia = np.sum(freqs * espectro) / np.sum(espectro)

    # c. Brillo (centroide espectral)
    brillo = fmedia  # en muchos textos se define igual que el centroide

    # d. Intensidad (energía)
    energia = np.sum(señal.astype(float)**2)

    return Frecuencia, f_media, brillo, energia


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
    Frecuencia, f_media, brillo, energia = caracteristicas(s, fs)
    print(f"{nombre}:")
    print(f"  Frecuencia fundamental: {Frecuencia:.2f} Hz")
    print(f"  Frecuencia media:       {f_media:.2f} Hz")
    print(f"  Brillo:                 {brillo:.2f} Hz")
    print(f"  Intensidad (energía):   {energia:.2e}\n")

```

Teniendo como resultado las siguientes características:


* **Mujer 1:**

  Frecuencia fundamental: 224.33 Hz
  
  Frecuencia media:       4472.52 Hz
  
  Brillo:                 4472.52 Hz
  
  Intensidad (energía):   6.77e+11

* **Mujer 2:**
  
  Frecuencia fundamental: 218.33 Hz
  
  Frecuencia media:       4122.21 Hz
  
  Brillo:                 4122.21 Hz
  
  Intensidad (energía):   4.97e+11

* **Mujer 3:**
  
  Frecuencia fundamental: 199.00 Hz
  
  Frecuencia media:       2841.48 Hz
  
  Brillo:                 2841.48 Hz
  
  Intensidad (energía):   1.47e+12

* **Hombre 1:**
  
  Frecuencia fundamental: 424.33 Hz
  
  Frecuencia media:       3111.79 Hz
  
  Brillo:                 3111.79 Hz
  
  Intensidad (energía):   1.10e+12

* **Hombre 2:**
  
  Frecuencia fundamental: 122.33 Hz
  
  Frecuencia media:       3639.71 Hz
  
  Brillo:                 3639.71 Hz
  
  Intensidad (energía):   8.66e+11

* **Hombre 3:**
  
  Frecuencia fundamental: 239.67 Hz
  
  Frecuencia media:       3547.13 Hz
  
  Brillo:                 3547.13 Hz
  
  Intensidad (energía):   1.56e+12

# PARTE B 
Para este segunda parte se divide en dos apartados, para el primer apartado se selecciono la grabación de un hombre y de una mujer para aplicaar un filtro pasa-banda en un rango de 80-400 Hz para el hombre y 150-500 Hz paa la mujer para eliminar el ruido, evidenciandose así en el siguiente codigo:

```python
```

Para el segundo apartado se aplicaron dos medetos de medición siendo uno jitter donde se detectó los periodos de vibración de una señal, se calculo los periodos TI de la señal de las voces obteniendo el jitter absoluto y calculando el jitter relativo; y siendo el segundo metodo, shimmer donde se detectó los picos de amplitud Ai en cada ciclo obteniendo así shimmer absoluto y calculando el shimmer realtivo, haciendo este proceso en las seis grabaciones, obteniendo así el siguiete codigo:

```python
```

# PARTE C
Para esta ultima parte se realiza una comparacion de los resultados de las señales de voces masculinas y femeninas para responder las siguientes preguntas:

### 1.  ¿Qué diferencias se observan en la frecuencia fundamental?


### 2. ¿Qué otras diferencias notan en términos de brillo, media o intensidad?


### Conclusiones 


### Importaancia clinica 





  
