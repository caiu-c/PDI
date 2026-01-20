import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista02\Q21\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.tif'))]

for file in files:
    img = cv2.imread(os.path.join(path_input, file), cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    # Redimensionamento opcional para agilizar o teste (ex: 256x256)
    img_small = cv2.resize(img, (256, 256))
    
    # --- Medição FFT (Numpy) ---
    start_fft = time.time()
    f_fft = np.fft.fft2(img_small)
    fshift_fft = np.fft.fftshift(f_fft)
    magnitude_fft = 20 * np.log(np.abs(fshift_fft) + 1)
    end_fft = time.time()
    time_fft = end_fft - start_fft

    # --- Medição DFT (OpenCV) ---
    # OpenCV exige conversão para float32 e retorna 2 canais (real, imag)
    img_float32 = np.float32(img_small)
    start_dft = time.time()
    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_dft = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)
    end_dft = time.time()
    time_dft = end_dft - start_dft

    # --- Plotagem e Comparação ---
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_small, cmap='gray')
    plt.title('Imagem Original (256x256)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(magnitude_fft, cmap='gray')
    plt.title(f'Espectro FFT (Numpy)\nTempo: {time_fft:.5f}s')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(magnitude_dft, cmap='gray')
    plt.title(f'Espectro DFT (OpenCV)\nTempo: {time_dft:.5f}s')
    plt.axis('off')

    plt.tight_layout()
    nome_saida = f'q21_comparativo_dft_fft_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300)
    plt.close()

print("Questão 21 concluída!")