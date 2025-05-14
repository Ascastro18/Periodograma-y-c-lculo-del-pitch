import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch, spectrogram, windows


# Función para calcular el Periodograma estándar
def Periodograma_Estandar(x, fs):
    N = len(x)
    X = np.fft.fft(x)
    Pxx = (np.abs(X)**2) / (fs * N)
    f = np.fft.fftfreq(N, d=1/fs)
    idx = f >= 0
    return f[idx], Pxx[idx]

# Función para calcular el Periodograma Ventaneado
def Periodograma_Ventaneado(x, fs, window='hamming'):
    N = len(x)
    w = getattr(windows, window)(N)
    xw = x * w
    U = np.sum(w**2) / N
    Xw = np.fft.fft(xw)
    Pxxw = (np.abs(Xw)**2) / (fs * N * U)
    f = np.fft.fftfreq(N, d=1/fs)
    idx = f >= 0
    return f[idx], Pxxw[idx]

# Función para el método de Welch
def metodo_welch(x, fs, nperseg=1024, noverlap=None, window='hamming'):
    f, Pxx = welch(x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap)
    return f, Pxx

# Detectar el Pitch (frecuencia fundamental)
def detect_pitch(f, Pxx, harmonic_threshold=0.1):
    idx_peak = np.argmax(Pxx)
    f_peak = f[idx_peak]
    A_peak = Pxx[idx_peak]
    for h in [2, 3, 4]:
        f_sub = f_peak / h
        idx_sub = (np.abs(f - f_sub)).argmin()
        if Pxx[idx_sub] > harmonic_threshold * A_peak:
            return f_sub
    return f_peak

# Clasificar la voz en función de la cercanía a la media de cada rango
def Clasificacion_media(pitch):
    medios = {'Bajo': 208, 'Tenor': 326, 'Soprano': 570}
    clasificacion = min(medios, key=lambda k: abs(medios[k] - pitch))
    return clasificacion

# Función principal para analizar la voz
def analizar_voz(file_path, method_params=None):
    fs, x = wavfile.read(file_path)
    if x.ndim > 1:
        x = x.mean(axis=1)

    # Periodogramas y PSD
    f1, P1 = Periodograma_Estandar(x, fs)
    win = method_params.get('window', 'hamming')
    f2, P2 = Periodograma_Ventaneado(x, fs, window=win)
    nperseg = method_params.get('nperseg', 1024)
    noverlap = method_params.get('noverlap', nperseg // 2)
    wind = method_params.get('window', 'hamming')
    f3, P3 = metodo_welch(x, fs, nperseg, noverlap, window=wind)

    # Detección de pitch
    pitches = {
        'Standard': detect_pitch(f1, P1),
        'Windowed': detect_pitch(f2, P2),
        'Welch': detect_pitch(f3, P3)
    }

    # Calcular el promedio de los tres métodos
    average_pitch = np.mean(list(pitches.values()))

    # Clasificación usando el pitch promedio
    Tipo_Voz = Clasificacion_media(average_pitch)
    print(f"Pitch Promedio: {average_pitch:.2f} Hz, Clasificación: {Tipo_Voz}")

    # Gráficas
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.semilogy(f1, P1)
    plt.title(f'Standard Periodogram - {file_path}')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('PSD (W/Hz)')

    plt.subplot(3, 1, 2)
    plt.semilogy(f2, P2)
    plt.title('Windowed Periodogram')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('PSD (W/Hz)')

    plt.subplot(3, 1, 3)
    plt.semilogy(f3, P3)
    plt.title('Welch PSD')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('PSD (W/Hz)')
    plt.tight_layout()
    plt.show()

    # Espectrograma
    eps = 1e-12
    f_s, t_s, Sxx = spectrogram(x, fs, window=wind, nperseg=nperseg, noverlap=noverlap)
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t_s, f_s, 10 * np.log10(Sxx + eps), shading='gouraud')
    plt.title('Espectrograma')
    plt.ylabel('Frecuencia (Hz)')
    plt.xlabel('Tiempo (s)')
    plt.colorbar(label='PSD (dB/Hz)')
    plt.show()

    return pitches, Tipo_Voz

if __name__ == '__main__':
    files = ['voz1.wav', 'voz2.wav', 'voz3.wav']
    params = {'window': 'hamming', 'nperseg': 1024, 'noverlap': 512}
    for vf in files:
        print(f"\nAnálisis de {vf}")
        analizar_voz(vf, method_params=params)
