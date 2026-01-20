import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista02\Q19\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

def get_distance_matrix(shape):
    """Calcula a matriz de distâncias D(u,v) a partir do centro do espectro."""
    M, N = shape
    u = np.arange(M)
    v = np.arange(N)
    u, v = np.meshgrid(u, v, indexing='ij')
    
    center_u, center_v = M // 2, N // 2
    return np.sqrt((u - center_u)**2 + (v - center_v)**2)

def filter_high_pass(img, type_filter='butterworth', d0=30, n=2):
    """Aplica filtros passa-alta no domínio da frequência."""
    # FFT e Centralização
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    D = get_distance_matrix(img.shape)
    
    # Seleção do Filtro H(u,v)
    if type_filter == 'butterworth':
        # HP_Butter = 1 / (1 + (D0/D)^(2n))
        # Evitar divisão por zero no centro
        H = 1 / (1 + (d0 / (D + 1e-5))**(2 * n))
    elif type_filter == 'gaussian':
        # HP_Gauss = 1 - exp(-D^2 / 2D0^2)
        H = 1 - np.exp(-(D**2) / (2 * (d0**2)))
    elif type_filter == 'laplacian':
        # O Laplaciano em frequência é proporcional a -D^2(u,v)
        # Para visualização, normalizamos o espectro
        H = (D**2)
        H = H / np.max(H)
        
    # Filtragem: G(u,v) = H(u,v) * F(u,v)
    res_shift = fshift * H
    res = np.fft.ifftshift(res_shift)
    img_back = np.abs(np.fft.ifft2(res))
    
    # Normalização para exibição espacial (0-255)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return fshift, H, res_shift, img_back

# Processamento Principal
files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.tif'))]

for file in files:
    img = cv2.imread(os.path.join(path_input, file), cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    filters = ['butterworth', 'gaussian', 'laplacian']
    d0 = 40  # Frequência de corte para Butterworth e Gaussiano
    
    for f_type in filters:
        f_shift, H, res_shift, img_back = filter_high_pass(img, f_type, d0)
        
        # Escala logarítmica para visualização dos espectros
        spec_orig = np.log(1 + np.abs(f_shift))
        spec_res = np.log(1 + np.abs(res_shift))
        
        plt.figure(figsize=(20, 6))
        
        titles = [
            'Imagem Original', 
            'Espectro da Imagem (F)', 
            f'Filtro Passa-Alta {f_type.capitalize()} (H)', 
            'Espectro Filtrado (G)', 
            'Resultado Espacial'
        ]
        images = [img, spec_orig, H, spec_res, img_back]
        
        for i in range(5):
            plt.subplot(1, 5, i+1)
            plt.imshow(images[i], cmap='gray')
            plt.title(titles[i], fontsize=12, pad=10)
            plt.axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        nome_saida = f'q19_passa_alta_{f_type}_{file}.png'
        plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
        plt.close()

print("Questão 19 concluída! Filtros passa-alta aplicados.")