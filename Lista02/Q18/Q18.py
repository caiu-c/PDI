import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista02\Q18\results'

# Criação do diretório de saída caso não exista
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

def filter_freq(img, type_filter='ideal', d0=30, n=2):
    """Aplica filtros passa-baixa no domínio da frequência."""
    # FFT e Centralização
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    D = get_distance_matrix(img.shape)
    
    # Seleção do Filtro H(u,v)
    if type_filter == 'ideal':
        H = (D <= d0).astype(float)
    elif type_filter == 'butterworth':
        H = 1 / (1 + (D / d0)**(2 * n))
    elif type_filter == 'gaussian':
        H = np.exp(-(D**2) / (2 * (d0**2)))
        
    # Filtragem: G(u,v) = H(u,v) * F(u,v)
    res_shift = fshift * H
    res = np.fft.ifftshift(res_shift)
    img_back = np.abs(np.fft.ifft2(res))
    
    return fshift, H, res_shift, img_back

# Processamento Principal
files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.tif'))]

for file in files:
    img = cv2.imread(os.path.join(path_input, file), cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    filters = ['ideal', 'butterworth', 'gaussian']
    d0 = 40  # Frequência de corte
    
    for f_type in filters:
        f_shift, H, res_shift, img_back = filter_freq(img, f_type, d0)
        
        # Escala logarítmica para melhor visualização do espectro
        spec_orig = np.log(1 + np.abs(f_shift))
        spec_res = np.log(1 + np.abs(res_shift))
        
        # Ajuste da figura: Aumento da altura para acomodar os títulos sem cortes
        plt.figure(figsize=(20, 6))
        
        titles = [
            'Imagem Original', 
            'Espectro da Imagem (F)', 
            f'Filtro {f_type.capitalize()} (H)', 
            'Espectro Filtrado (G)', 
            'Resultado Espacial'
        ]
        images = [img, spec_orig, H, spec_res, img_back]
        
        for i in range(5):
            plt.subplot(1, 5, i+1)
            plt.imshow(images[i], cmap='gray')
            plt.title(titles[i], fontsize=12, pad=10) # Pad adiciona espaço ao título
            plt.axis('off')
            
        # O parâmetro rect garante que o layout não encoste no topo da imagem salva
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Salvamento da imagem
        nome_saida = f'q18_{f_type}_{file}.png'
        plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
        plt.close()

print("Questão 18 concluída com sucesso!")
