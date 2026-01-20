import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_Binarias'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q26\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# 2. Seleção de arquivos (Suporte a .gif adicionado)
extensoes = ('.png', '.jpg', '.tif', '.bmp', '.gif')
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(extensoes)]

# Seleciona 5 imagens aleatórias (ou o máximo disponível se for menos de 5)
num_amostras = min(5, len(all_files))
files = random.sample(all_files, num_amostras)

print(f"Imagens selecionadas para análise: {files}")

for file in files:
    img_path = os.path.join(path_input, file)
    
    # Leitura da imagem (OpenCV lê o primeiro frame de GIFs estáticos)
    img_bin = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_bin is None: continue

    # Garantir binarização estrita
    _, img_bin = cv2.threshold(img_bin, 127, 255, cv2.THRESH_BINARY)

    # 3. Detecção de Componentes Conectados (Blobs)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin, connectivity=8)

    # Geração do Mapa de Rótulos Coloridos
    # Usamos o espaço HSV para garantir cores distintas para cada ID
    label_hue = np.uint8(179 * labels / np.max(labels) if np.max(labels) > 0 else 0)
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0 # Fundo permanece preto

    # Desenho de Bounding Boxes e Centroides
    img_out = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        
        if area > 10: # Filtro de ruído
            cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(img_out, (int(cx), int(cy)), 3, (0, 0, 255), -1)

    # --- PLOTAGEM COM LAYOUT SEGURO ---
    plt.figure(figsize=(18, 7))
    
    titles = ['Entrada Binária', 'Mapa de Rótulos (IDs)', 'Objetos e Centroides']
    images = [img_bin, labeled_img, img_out]

    for i in range(3):
        plt.subplot(1, 3, i+1)
        if i > 0:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(images[i], cmap='gray')
            
        plt.title(titles[i], fontsize=14, pad=30) # Pad aumentado para evitar cortes
        plt.axis('off')

    # Ajuste do rect para garantir espaço para o título no topo
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    
    nome_saida = f'q26_random_analise_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Questão 26 concluída! 5 amostras processadas em {path_output}")