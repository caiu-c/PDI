import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_Binarias'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q34\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# 2. Seleção de 3 imagens aleatórias
extensoes = ('.png', '.jpg', '.tif', '.bmp', '.gif')
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(extensoes)]
files = random.sample(all_files, 3)

for file in files:
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    # Garantir binarização (objetos brancos em fundo preto)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Encontrar o contorno principal (maior objeto)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: continue
    c = max(cnts, key=cv2.contourArea)

    # 3. Cálculo dos Momentos usando OpenCV
    M = cv2.moments(c)

    # Centroide (usando momentos brutos)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Extração de Momentos Centrais (mu)
    # mu20 e mu02 estão relacionados à dispersão (variância) da forma
    m_centrais = {
        "mu20": M['mu20'],
        "mu02": M['mu02'],
        "mu11": M['mu11'],
        "mu30": M['mu30']
    }

    # --- VISUALIZAÇÃO ---
    img_out = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_out, [c], -1, (0, 255, 0), 2)
    cv2.circle(img_out, (cx, cy), 5, (0, 0, 255), -1)

    plt.figure(figsize=(12, 7))
    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    
    # Texto descritivo com os momentos calculados
    info_text = (f"Centroide: ({cx}, {cy})\n"
                 f"Area (m00): {M['m00']:.1f}\n"
                 f"mu20: {M['mu20']:.2e}\n"
                 f"mu02: {M['mu02']:.2e}\n"
                 f"mu11: {M['mu11']:.2e}")
    
    plt.text(10, 50, info_text, fontsize=10, color='white', 
             bbox=dict(facecolor='black', alpha=0.7))
    
    plt.title(f"Momentos Centrais - {file}", fontsize=14, pad=25)
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    
    nome_saida = f'q34_momentos_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Questão 34 concluída! Resultados em {path_output}")