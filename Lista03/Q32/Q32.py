import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q32\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# 2. Seleção de 3 imagens aleatórias
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.tif', '.bmp'))]
files = random.sample(all_files, 3)

for file in files:
    img = cv2.imread(os.path.join(path_input, file))
    if img is None: continue
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- DETECÇÃO DE LINHAS (Probabilistic Hough Transform) ---
    # Primeiro: Detecção de bordas Canny
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    # HoughLinesP: retorna segmentos (x1, y1, x2, y2)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    
    img_lines = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # --- DETECÇÃO DE CÍRCULOS (Hough Gradient Method) ---
    # Necessário aplicar um blur para reduzir ruídos/falsos círculos
    img_blur = cv2.medianBlur(img_gray, 5)
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=100, param2=30, minRadius=10, maxRadius=100)
    
    img_circles = img.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Desenhar o círculo externo (vermelho)
            cv2.circle(img_circles, (i[0], i[1]), i[2], (0, 0, 255), 3)
            # Desenhar o centro do círculo (azul)
            cv2.circle(img_circles, (i[0], i[1]), 2, (255, 0, 0), 3)

    # --- PLOTAGEM COM LAYOUT SEGURO ---
    plt.figure(figsize=(18, 6))
    
    # Converter para RGB para exibição correta no Matplotlib
    res_lines = cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB)
    res_circles = cv2.cvtColor(img_circles, cv2.COLOR_BGR2RGB)
    
    titles = [f'Original ({file})', 'Hough: Linhas (Verde)', 'Hough: Circulos (Vermelho)']
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), res_lines, res_circles]

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i])
        plt.title(titles[i], fontsize=14, pad=25)
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    nome_saida = f'q32_hough_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Questão 32 concluída! Detecções salvas em {path_output}")