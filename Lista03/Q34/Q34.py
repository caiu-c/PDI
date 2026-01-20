import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# ---------------------------------------------------------
# 1. CONFIGURAÇÃO DE DIRETÓRIOS E AMBIENTE
# ---------------------------------------------------------
path_input = r'C:\cod_mestrado\pdi\BancoImagens_Binarias'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q34\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# ---------------------------------------------------------
# 2. SELEÇÃO ALEATÓRIA DE AMOSTRAS
# ---------------------------------------------------------
extensoes = ('.png', '.jpg', '.tif', '.bmp', '.gif')
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(extensoes)]
files = random.sample(all_files, min(3, len(all_files)))

# ---------------------------------------------------------
# 3. EXTRAÇÃO DE MOMENTOS E DESCRITORES DE FORMA
# ---------------------------------------------------------
for file in files:
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: 
        continue

    # Normalização para binarização estrita (Objetos = 255, Fundo = 0)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Identificação de contornos para isolamento do objeto principal
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: 
        continue
    c = max(cnts, key=cv2.contourArea)

    # Cálculo dos Momentos de imagem
    # m_ij representa o momento bruto de ordem (i+j)
    M = cv2.moments(c)

    # Cálculo do Centroide (Coordenadas do centro de massa)
    # cx = m10 / m00 | cy = m01 / m00
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0

    # ---------------------------------------------------------
    # 4. RENDERIZAÇÃO E DOCUMENTAÇÃO DOS RESULTADOS
    # ---------------------------------------------------------
    img_out = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_out, [c], -1, (0, 255, 0), 2)  # Contorno em Verde
    cv2.circle(img_out, (cx, cy), 5, (0, 0, 255), -1)   # Centroide em Vermelho

    plt.figure(figsize=(12, 7))
    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    
    # Bloco informativo com Momentos Brutos e Centrais (mu)
    # mu_pq são momentos invariantes à translação
    info_text = (f"Centroide: ({cx}, {cy})\n"
                 f"Área (m00): {M['m00']:.1f}\n"
                 f"mu20: {M['mu20']:.2e}\n"
                 f"mu02: {M['mu02']:.2e}\n"
                 f"mu11: {M['mu11']:.2e}")
    
    plt.text(10, 50, info_text, fontsize=10, color='white', 
             bbox=dict(facecolor='black', alpha=0.7))
    
    plt.title(f"Análise de Momentos - {file}", fontsize=14, pad=25)
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    
    nome_saida = f'q34_momentos_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Processamento concluído. Resultados salvos em: {path_output}")