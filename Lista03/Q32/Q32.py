import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# ---------------------------------------------------------
# 1. CONFIGURAÇÃO DE DIRETÓRIOS E AMBIENTE
# ---------------------------------------------------------
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q32\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# ---------------------------------------------------------
# 2. SELEÇÃO DE AMOSTRAGEM ALEATÓRIA
# ---------------------------------------------------------
extensoes = ('.png', '.jpg', '.tif', '.bmp')
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(extensoes)]
files = random.sample(all_files, min(3, len(all_files)))

# ---------------------------------------------------------
# 3. DETECÇÃO GEOMÉTRICA (TRANSFORMADA DE HOUGH)
# ---------------------------------------------------------
for file in files:
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path)
    if img is None: 
        continue
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- MÉTODO A: Transformada de Hough Probabilística (Linhas) ---
    # Aplicação de Canny para extração de bordas (pré-requisito)
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    
    # HoughLinesP: Identificação de segmentos de reta (x1, y1, x2, y2)
    # Parametrização: rho=1, theta=pi/180, threshold=100
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    
    img_lines = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # --- MÉTODO B: Transformada de Hough de Círculos (Hough Gradient) ---
    # Aplicação de Filtro Mediano para atenuação de ruído impulsivo
    img_blur = cv2.medianBlur(img_gray, 5)
    
    # Detecção baseada no gradiente: dp=1.2, minDist=30px
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=100, param2=30, minRadius=10, maxRadius=100)
    
    img_circles = img.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Desenho da circunferência (Vermelho) e Centroide (Azul)
            cv2.circle(img_circles, (i[0], i[1]), i[2], (0, 0, 255), 3)
            cv2.circle(img_circles, (i[0], i[1]), 2, (255, 0, 0), 3)

    # ---------------------------------------------------------
    # 4. VISUALIZAÇÃO E EXPORTAÇÃO
    # ---------------------------------------------------------
    plt.figure(figsize=(18, 6))
    
    # Normalização de espaço de cor BGR para RGB (Matplotlib)
    res_lines = cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB)
    res_circles = cv2.cvtColor(img_circles, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    titles = [f'Original ({file})', 'Hough: Linhas (Verde)', 'Hough: Círculos (Vermelho)']
    images = [img_rgb, res_lines, res_circles]

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i])
        plt.title(titles[i], fontsize=14, pad=25)
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    nome_saida = f'q32_hough_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Processamento concluído. Resultados salvos em: {path_output}")