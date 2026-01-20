import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# ---------------------------------------------------------
# 1. CONFIGURAÇÃO DE DIRETÓRIOS E AMBIENTE
# ---------------------------------------------------------
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q36\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# ---------------------------------------------------------
# 2. IMPLEMENTAÇÃO DO OPERADOR LBP (3x3)
# ---------------------------------------------------------
def compute_lbp_numpy(img_gray):
    """
    Implementação vetorizada do LBP padrão (P=8, R=1).
    Calcula a relação binária entre o pixel central e seus 8 vizinhos.
    """
    h, w = img_gray.shape
    
    # Recorte do pixel central (exclui bordas para evitar erros de índice)
    center = img_gray[1:h-1, 1:w-1]
    
    # Inicialização da matriz de saída
    lbp_img = np.zeros((h-2, w-2), dtype=np.uint8)
    
    # Definição dos deslocamentos (dy, dx) e peso (bit_pos) dos 8 vizinhos
    # Ordem horária partindo do canto superior esquerdo
    neighbors = [
        (-1, -1, 7), (-1, 0, 6), (-1, 1, 5),
        ( 0, 1, 4), 
        ( 1, 1, 3), ( 1, 0, 2), ( 1, -1, 1),
        ( 0, -1, 0)
    ]

    for dy, dx, bit_pos in neighbors:
        # Extração da vizinhança deslocada
        neighbor_slice = img_gray[1+dy:h-1+dy, 1+dx:w-1+dx]
        
        # Aplicação da função limiar s(x): 1 se vizinho >= centro, senão 0
        lbp_img |= (neighbor_slice >= center).astype(np.uint8) << bit_pos
        
    return lbp_img

# ---------------------------------------------------------
# 3. PROCESSAMENTO E ANÁLISE DE TEXTURA
# ---------------------------------------------------------
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.tif', '.bmp'))]
files = random.sample(all_files, min(3, len(all_files)))

for file in files:
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: 
        continue

    # Cálculo do mapa LBP (micro-padrões de textura)
    lbp_result = compute_lbp_numpy(img)

    # Geração do Histograma LBP (Vetor de características)
    # Representa a frequência de ocorrência de cada um dos 256 padrões possíveis
    lbp_hist = cv2.calcHist([lbp_result], [0], None, [256], [0, 256])
    
    # Normalização L1 para garantir invariância à escala/tamanho da imagem
    cv2.normalize(lbp_hist, lbp_hist, norm_type=cv2.NORM_L1)

    # ---------------------------------------------------------
    # 4. VISUALIZAÇÃO E DOCUMENTAÇÃO
    # ---------------------------------------------------------
    plt.figure(figsize=(18, 6))
    
    # Imagem Original (Input)
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'Textura Original: {file}', fontsize=12, pad=15)
    plt.axis('off')

    # Mapa LBP (Visualização das intensidades codificadas)
    plt.subplot(1, 3, 2)
    plt.imshow(lbp_result, cmap='gray')
    plt.title('Imagem LBP (Codificação Binária)', fontsize=12, pad=15)
    plt.axis('off')

    # Histograma de Assinatura
    plt.subplot(1, 3, 3)
    plt.plot(lbp_hist, color='black')
    plt.fill_between(range(256), lbp_hist.flatten(), color='gray', alpha=0.5)
    plt.title('Assinatura LBP (Histograma)', fontsize=12, pad=15)
    plt.xlabel('Padrão LBP (0-255)')
    plt.ylabel('Frequência Normalizada')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xlim([0, 255])

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    nome_saida = f'q36_lbp_texture_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Questão 36 concluída. Histogramas de textura salvos em: {path_output}")