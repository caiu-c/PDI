import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from skimage.feature import graycomatrix, graycoprops

# --- Configurações de Caminho ---
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q37\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# --- Seleção de 3 imagens aleatórias ---
extensoes = ('.png', '.jpg', '.tif', '.bmp')
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(extensoes)]
selected_files = random.sample(all_files, 3)

# Parâmetros: Distância 1 e Ângulos principais
distancias = [1]
angulos = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 0°, 45°, 90°, 135°
propriedades = ['contrast', 'correlation', 'energy', 'homogeneity']

for file in selected_files:
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    # Quantização para 32 níveis (estabiliza a GLCM e reduz custo computacional)
    img_quant = (img // 8).astype(np.uint8)

    # 1. Gerar a GLCM
    # levels=32 pois dividimos 256 por 8
    glcm = graycomatrix(img_quant, distances=distancias, angles=angulos, 
                        levels=32, symmetric=True, normed=True)

    # 2. Extrair Métricas de Haralick
    stats = {prop: graycoprops(glcm, prop)[0] for prop in propriedades}

    # --- PLOTAGEM COM LAYOUT SEGURO ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Imagem Original
    ax1.imshow(img, cmap='gray')
    ax1.set_title(f'Textura Original: {file}', fontsize=14, pad=25)
    ax1.axis('off')

    # Gráfico de Barras dos Descritores
    labels_ang = ['0°', '45°', '90°', '135°']
    x = np.arange(len(labels_ang))
    width = 0.2

    # Plotando os 4 descritores com escalas ajustadas para visualização
    ax2.bar(x - 1.5*width, stats['contrast'], width, label='Contraste')
    ax2.bar(x - 0.5*width, stats['homogeneity'] * 100, width, label='Homogen. (x100)')
    ax2.bar(x + 0.5*width, stats['energy'] * 100, width, label='Energia (x100)')
    ax2.bar(x + 1.5*width, stats['correlation'] * 10, width, label='Correl. (x10)')

    ax2.set_title('Análise Direcional de Haralick', fontsize=14, pad=25)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_ang)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    # Margem de segurança para o título (0.90)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    
    nome_saida = f'q37_analise_haralick_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Questão 37 finalizada com sucesso para as imagens: {selected_files}")