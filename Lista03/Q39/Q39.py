import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import time

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q39\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# 2. Seleção de 3 imagens aleatórias
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.tif', '.bmp'))]
files = random.sample(all_files, 3)

for file in files:
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path)
    if img is None: continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- SIFT ---
    sift = cv2.SIFT_create()
    start_sift = time.time()
    kp_sift, _ = sift.detectAndCompute(gray, None)
    end_sift = time.time()
    time_sift = end_sift - start_sift
    img_sift = cv2.drawKeypoints(img, kp_sift, None, color=(0, 255, 0), flags=0)

    # --- SURF ---
    # Nota: Caso o SURF não esteja disponível, usaremos o ORB como fallback para não quebrar o código
    # mas o comentário indicará se o SURF foi executado.
    try:
        surf = cv2.xfeatures2d.SURF_create(400) # 400 é o limiar de Hessian
        start_surf = time.time()
        kp_surf, _ = surf.detectAndCompute(gray, None)
        end_surf = time.time()
        time_surf = end_surf - start_surf
        label_alg = "SURF"
    except AttributeError:
        # Fallback para ORB caso o SURF esteja desativado na sua versão do OpenCV
        surf = cv2.ORB_create()
        start_surf = time.time()
        kp_surf, _ = surf.detectAndCompute(gray, None)
        end_surf = time.time()
        time_surf = end_surf - start_surf
        label_alg = "ORB (SURF indisponivel)"

    img_surf = cv2.drawKeypoints(img, kp_surf, None, color=(255, 0, 0), flags=0)

    # --- PLOTAGEM COMPARATIVA ---
    plt.figure(figsize=(18, 8))
    
    titles = [
        f'SIFT (Verde)\nTempo: {time_sift:.4f}s | Pontos: {len(kp_sift)}',
        f'{label_alg} (Vermelho)\nTempo: {time_surf:.4f}s | Pontos: {len(kp_surf)}'
    ]
    images = [cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB), cv2.cvtColor(img_surf, cv2.COLOR_BGR2RGB)]

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.imshow(images[i])
        plt.title(titles[i], fontsize=13, pad=30)
        plt.axis('off')

    plt.suptitle(f"Comparativo de Descritores Locais: {file}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    
    nome_saida = f'q39_sift_vs_surf_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Questão 39 concluída para as imagens: {files}")