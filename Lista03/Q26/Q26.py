import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# ---------------------------------------------------------
# 1. DEFINIÇÃO DE CAMINHOS E INFRAESTRUTURA
# ---------------------------------------------------------
path_input = r'C:\cod_mestrado\pdi\BancoImagens_Binarias'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q26\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# ---------------------------------------------------------
# 2. SELEÇÃO ALEATÓRIA E FILTRAGEM DE ARQUIVOS
# ---------------------------------------------------------
extensoes = ('.png', '.jpg', '.tif', '.bmp', '.gif')
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(extensoes)]

# Seleção de amostragem para análise (limite de 5 unidades)
num_amostras = min(5, len(all_files))
files = random.sample(all_files, num_amostras)

# ---------------------------------------------------------
# 3. DETECÇÃO E ANÁLISE DE COMPONENTES CONECTADOS
# ---------------------------------------------------------
for file in files:
    img_path = os.path.join(path_input, file)
    
    # Leitura em escala de cinza (frame único para formatos estáticos)
    img_bin = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_bin is None: 
        continue

    # Normalização de binarização (Threshold Fixo)
    _, img_bin = cv2.threshold(img_bin, 127, 255, cv2.THRESH_BINARY)

    # Extração de estatísticas e rótulos (Conectividade 8-vizinhos)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_bin, connectivity=8)

    # Geração do Mapa de Rótulos em Cores (Espaço HSV)
    label_hue = np.uint8(179 * labels / np.max(labels) if np.max(labels) > 0 else 0)
    blank_ch = 255 * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue == 0] = 0 # Preservação do background

    # Renderização de Bounding Boxes e Centroides
    img_out = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        
        # Filtro morfológico de área para redução de ruído
        if area > 10: 
            cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(img_out, (int(cx), int(cy)), 3, (0, 0, 255), -1)

    # ---------------------------------------------------------
    # 4. EXPORTAÇÃO E VISUALIZAÇÃO DOS RESULTADOS
    # ---------------------------------------------------------
    plt.figure(figsize=(18, 7))
    
    titles = ['Entrada Binária', 'Mapa de Rótulos (IDs)', 'Objetos e Centroides']
    images = [img_bin, labeled_img, img_out]

    for i in range(3):
        plt.subplot(1, 3, i+1)
        if i > 0:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(images[i], cmap='gray')
            
        plt.title(titles[i], fontsize=14, pad=30)
        plt.axis('off')

    # Ajuste de margens para preservação dos títulos superiores
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    
    nome_saida = f'q26_random_analise_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Processamento concluído para {num_amostras} amostras.")