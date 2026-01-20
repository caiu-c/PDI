import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista02\Q20\results'

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

def filter_band(img, type_op='reject', d0=60, w=20, n=2):
    """
    Aplica filtros rejeita-banda ou passa-banda (Butterworth).
    d0: frequencia central da banda
    w: largura da banda
    """
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    D = get_distance_matrix(img.shape)
    
    # Fórmula do filtro Butterworth Band-Reject
    # Hbr = 1 / (1 + ( (D*W) / (D^2 - D0^2) )^(2n))
    # Evitar divisão por zero onde D == d0
    numerator = D * w
    denominator = D**2 - d0**2
    denominator[denominator == 0] = 1e-5 # Proteção contra divisão por zero
    
    Hbr = 1 / (1 + (numerator / denominator)**(2 * n))
    
    if type_op == 'reject':
        H = Hbr
    else:
        H = 1 - Hbr # Passa-banda é o inverso do rejeita-banda
        
    res_shift = fshift * H
    res = np.fft.ifftshift(res_shift)
    img_back = np.abs(np.fft.ifft2(res))
    
    # Normalização para exibição
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return fshift, H, res_shift, img_back

# Processamento
files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.tif'))]

for file in files:
    img = cv2.imread(os.path.join(path_input, file), cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    # Parâmetros da banda (ajustáveis conforme a imagem)
    d0_val = 50 
    w_val = 30

    for op in ['reject', 'pass']:
        f_shift, H, res_shift, img_back = filter_band(img, type_op=op, d0=d0_val, w=w_val)
        
        spec_orig = np.log(1 + np.abs(f_shift))
        spec_res = np.log(1 + np.abs(res_shift))
        
        plt.figure(figsize=(20, 6))
        
        label_op = "Rejeita-Banda" if op == 'reject' else "Passa-Banda"
        titles = [
            'Original', 
            'Espectro Original (F)', 
            f'Filtro {label_op} (H)', 
            'Espectro Filtrado (G)', 
            f'Resultado {label_op}'
        ]
        images = [img, spec_orig, H, spec_res, img_back]
        
        for i in range(5):
            plt.subplot(1, 5, i+1)
            plt.imshow(images[i], cmap='gray')
            plt.title(titles[i], fontsize=12, pad=10)
            plt.axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Salvamento da imagem
        nome_saida = f'q20_{op}_{file}.png'
        plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
        plt.close()

print("Questão 20 concluída! Filtros de banda aplicados.")
