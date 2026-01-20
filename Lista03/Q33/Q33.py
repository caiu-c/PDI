import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# ---------------------------------------------------------
# 1. CONFIGURAÇÃO DE DIRETÓRIOS E AMBIENTE
# ---------------------------------------------------------
path_input = r'C:\cod_mestrado\pdi\BancoImagens_Binarias'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q33\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# ---------------------------------------------------------
# 2. SELEÇÃO ALEATÓRIA DE AMOSTRAS
# ---------------------------------------------------------
extensoes = ('.png', '.jpg', '.tif', '.bmp', '.gif')
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(extensoes)]
files = random.sample(all_files, min(3, len(all_files)))

# ---------------------------------------------------------
# 3. SEGMENTAÇÃO POR WATERSHED
# ---------------------------------------------------------
for file in files:
    img_path = os.path.join(path_input, file)
    # O Watershed requer entrada colorida para renderização das fronteiras
    img = cv2.imread(img_path)
    if img is None: 
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binarização Inversa via Otsu para isolamento do objeto
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Refinamento Morfológico
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Identificação do Background Determinístico (Sure Background)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Transformada de Distância e Foreground Determinístico (Sure Foreground)
    # A Transformada de Distância calcula o mapa de distâncias ao pixel zero mais próximo
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Localização da Região Desconhecida (Fronteiras candidatas)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # ---------------------------------------------------------
    # 4. ROTULAGEM E APLICAÇÃO DO ALGORITMO
    # ---------------------------------------------------------
    # Criação de marcadores para os componentes conectados
    _, markers = cv2.connectedComponents(sure_fg)
    
    # Ajuste de rótulos: Background = 1, Objetos > 1, Desconhecido = 0
    markers = markers + 1
    markers[unknown == 255] = 0

    # Aplicação do Watershed
    img_result = img.copy()
    markers = cv2.watershed(img_result, markers)
    
    # Marcação das linhas de crista (bordas) em Vermelho [BGR: 0, 0, 255]
    img_result[markers == -1] = [0, 0, 255]

    # ---------------------------------------------------------
    # 5. VISUALIZAÇÃO DOS ESTÁGIOS DE PROCESSAMENTO
    # ---------------------------------------------------------
    plt.figure(figsize=(20, 6))
    
    # Normalização do mapa de distância para fins de visualização
    dist_vis = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    titles = [
        f'Original ({file})', 
        'Transformada de Distância', 
        'Marcadores (Heatmap)', 
        'Segmentação Watershed'
    ]
    
    images = [
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        dist_vis,
        markers,
        cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
    ]

    for i in range(4):
        plt.subplot(1, 4, i+1)
        if i == 2:
            plt.imshow(images[i], cmap='jet')
        elif i == 1:
            plt.imshow(images[i], cmap='gray')
        else:
            plt.imshow(images[i])
            
        plt.title(titles[i], fontsize=13, pad=25)
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    
    nome_saida = f'q33_watershed_final_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Processamento concluído. Resultados salvos em: {path_output}")