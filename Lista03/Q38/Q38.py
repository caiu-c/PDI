import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# ---------------------------------------------------------
# 1. CONFIGURAÇÃO DE DIRETÓRIOS E AMBIENTE
# ---------------------------------------------------------
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q38\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# ---------------------------------------------------------
# 2. SELEÇÃO DE AMOSTRAGEM ALEATÓRIA
# ---------------------------------------------------------
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.tif', '.bmp'))]
files = random.sample(all_files, min(3, len(all_files)))

# ---------------------------------------------------------
# 3. EXTRAÇÃO DE CARACTERÍSTICAS LOCAIS (SIFT)
# ---------------------------------------------------------
for file in files:
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path)
    if img is None: 
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Inicialização do detector SIFT (Scale-Invariant Feature Transform)
    sift = cv2.SIFT_create()

    # Detecção de pontos-chave (keypoints) e extração de descritores
    # Keypoints: Coordenadas espaciais, escala e orientação
    # Descriptors: Vetor de características (normalmente 128 dimensões)
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # Renderização dos pontos-chave sobre a imagem original
    # O flag DRAW_RICH_KEYPOINTS exibe o raio e a orientação dominante de cada ponto
    img_sift = cv2.drawKeypoints(img, keypoints, None, 
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                 color=(0, 255, 0))

    # ---------------------------------------------------------
    # 4. VISUALIZAÇÃO E EXPORTAÇÃO DOS RESULTADOS
    # ---------------------------------------------------------
    plt.figure(figsize=(16, 8))
    
    # Normalização de cores BGR para RGB para compatibilidade com Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res_sift = cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB)
    
    titles = [f'Original ({file})', f'SIFT: {len(keypoints)} Pontos Detectados']
    images = [img_rgb, res_sift]

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.imshow(images[i])
        plt.title(titles[i], fontsize=14, pad=30)
        plt.axis('off')

    # Ajuste de margens para preservação da integridade visual dos títulos
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    nome_saida = f'q38_sift_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Processamento concluído. Características SIFT salvas em: {path_output}")