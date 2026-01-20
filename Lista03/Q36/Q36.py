import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q36\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

def compute_lbp_numpy(img_gray):
    """
    Implementação vetorizada eficiente do LBP padrão 3x3.
    Retorna a imagem LBP com 2 pixels a menos na largura e altura (bordas).
    """
    h, w = img_gray.shape
    
    # Recorte do pixel central (ignora a borda de 1 pixel)
    center = img_gray[1:h-1, 1:w-1]
    
    # Inicializa a imagem LBP
    lbp_img = np.zeros((h-2, w-2), dtype=np.uint8)
    
    # Definição dos deslocamentos (dy, dx) e posição do bit para os 8 vizinhos
    # Ordem: Horário começando do canto superior esquerdo
    neighbors = [
        (-1, -1, 7), (-1, 0, 6), (-1, 1, 5), # Linha superior
        ( 0, 1, 4),                          # Direita
        ( 1, 1, 3), ( 1, 0, 2), ( 1, -1, 1), # Linha inferior
        ( 0, -1, 0)                          # Esquerda
    ]

    for dy, dx, bit_pos in neighbors:
        # Recorte do vizinho deslocado
        neighbor_slice = img_gray[1+dy:h-1+dy, 1+dx:w-1+dx]
        
        # Comparação e deslocamento de bit
        # Se vizinho >= centro, bit é 1. Desloca para a posição correta.
        lbp_img |= (neighbor_slice >= center).astype(np.uint8) << bit_pos
        
    return lbp_img

# 2. Seleção de 3 imagens aleatórias (buscando texturas distintas)
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.tif', '.bmp'))]
files = random.sample(all_files, 3)

for file in files:
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    # 3. Cálculo do LBP
    lbp_result = compute_lbp_numpy(img)

    # 4. Cálculo do Histograma LBP (o vetor de características da textura)
    # bins=256 (para os valores de 0 a 255)
    lbp_hist = cv2.calcHist([lbp_result], [0], None, [256], [0, 256])
    # Normalização do histograma para comparar imagens de tamanhos diferentes
    cv2.normalize(lbp_hist, lbp_hist, norm_type=cv2.NORM_L1)

    # --- PLOTAGEM PARA RELATÓRIO ---
    plt.figure(figsize=(18, 6))
    
    # Imagem Original
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'Textura Original: {file}', fontsize=12, pad=15)
    plt.axis('off')

    # Imagem LBP Resultante
    plt.subplot(1, 3, 2)
    plt.imshow(lbp_result, cmap='gray')
    plt.title('Imagem LBP (Micro-padrões)', fontsize=12, pad=15)
    plt.axis('off')

    # Histograma LBP
    plt.subplot(1, 3, 3)
    plt.plot(lbp_hist, color='black')
    plt.fill_between(range(256), lbp_hist.flatten(), color='gray', alpha=0.5)
    plt.title('Histograma LBP (Assinatura da Textura)', fontsize=12, pad=15)
    plt.xlabel('Valor LBP (0-255)')
    plt.ylabel('Frequência Normalizada')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim([0, 255])

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig(os.path.join(path_output, f'q36_lbp_texture_{file}.png'), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Questão 36 concluída! Análises de textura salvas em {path_output}")