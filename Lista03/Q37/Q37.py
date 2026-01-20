import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from skimage.feature import graycomatrix, graycoprops

# ---------------------------------------------------------
# 1. CONFIGURAÇÃO DE DIRETÓRIOS E AMBIENTE
# ---------------------------------------------------------
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q37\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# ---------------------------------------------------------
# 2. DEFINIÇÃO DE PARÂMETROS E AMOSTRAGEM
# ---------------------------------------------------------
extensoes = ('.png', '.jpg', '.tif', '.bmp')
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(extensoes)]
selected_files = random.sample(all_files, min(3, len(all_files)))

# Configurações da GLCM: Distância unitária e orientações principais
distancias = [1]
angulos = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0°, 45°, 90°, 135°
propriedades = ['contrast', 'correlation', 'energy', 'homogeneity']

# ---------------------------------------------------------
# 3. EXTRAÇÃO DE CARACTERÍSTICAS DE HARALICK
# ---------------------------------------------------------
for file in selected_files:
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: 
        continue

    # Quantização para 32 níveis de cinza
    # Reduz a esparsidade da matriz GLCM e otimiza a estabilidade estatística
    img_quant = (img // 8).astype(np.uint8)

    # Geração da GLCM (Gray-Level Co-occurrence Matrix)
    # Symmetric=True considera a relação i-j e j-i; Normed=True normaliza as probabilidades
    glcm = graycomatrix(img_quant, distances=distancias, angles=angulos, 
                        levels=32, symmetric=True, normed=True)

    # Extração das métricas de Haralick para cada ângulo definido
    stats = {prop: graycoprops(glcm, prop)[0] for prop in propriedades}

    # ---------------------------------------------------------
    # 4. VISUALIZAÇÃO E DOCUMENTAÇÃO TÉCNICA
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Renderização da imagem de textura original
    ax1.imshow(img, cmap='gray')
    ax1.set_title(f'Textura Original: {file}', fontsize=14, pad=25)
    ax1.axis('off')

    # Representação gráfica dos descritores direcionais
    labels_ang = ['0°', '45°', '90°', '135°']
    x = np.arange(len(labels_ang))
    width = 0.2

    # Plotagem dos descritores com fatores de escala para visualização comparativa
    # Contrast: Intensidade das variações locais
    # Homogeneity: Proximidade da distribuição dos elementos à diagonal da GLCM
    # Energy: Uniformidade da distribuição de níveis de cinza
    # Correlation: Dependência linear dos níveis de cinza entre vizinhos
    ax2.bar(x - 1.5*width, stats['contrast'], width, label='Contraste')
    ax2.bar(x - 0.5*width, stats['homogeneity'] * 100, width, label='Homogen. (x100)')
    ax2.bar(x + 0.5*width, stats['energy'] * 100, width, label='Energia (x100)')
    ax2.bar(x + 1.5*width, stats['correlation'] * 10, width, label='Correl. (x10)')

    ax2.set_title('Análise Direcional de Haralick', fontsize=14, pad=25)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_ang)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    # Ajuste de layout para preservação dos metadados visuais
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    
    nome_saida = f'q37_analise_haralick_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Processamento concluído para as amostras: {selected_files}")