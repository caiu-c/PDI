# Questão 7 - Técnicas de Aguçamento (Laplaciano e Sobel)
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista01\Q07\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

files = os.listdir(path_input)

for file in files:
    if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
        continue

    # 2. Carregar imagem em tons de cinza
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    # --- TÉCNICA 1: AGUÇAMENTO VIA LAPLACIANO (PASSA-ALTA) ---
    # O Laplaciano destaca as bordas. Ao subtraí-lo da imagem original 
    # (dependendo do sinal do kernel), realçamos os detalhes.
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    # Convertendo para uint8 para exibição do filtro isolado
    lap_abs = cv2.convertScaleAbs(laplacian)
    # Aguçamento: Imagem - Laplaciano (para kernels com centro negativo)
    # Usamos cv2.addWeighted ou operação direta com clip para evitar estouro de 255
    sharp_lap = np.clip(img.astype(np.float64) - laplacian, 0, 255).astype(np.uint8)

    # --- TÉCNICA 2: AGUÇAMENTO VIA GRADIENTE (SOBEL) ---
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag_sobel = np.sqrt(sobelx**2 + sobely**2)
    # Aguçamento: Somar a magnitude do gradiente à imagem original
    sharp_grad = np.clip(img.astype(np.float64) + mag_sobel, 0, 255).astype(np.uint8)

    # 3. Exibição e Comparação
    titles = ['Original', 'Filtro Laplaciano', 'Aguçamento (Laplaciano)', 'Aguçamento (Gradiente)']
    images = [img, lap_abs, sharp_lap, sharp_grad]

    plt.figure(figsize=(20, 10))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()

    # 4. Salvar resultado
    nome_saida = f"q7_agucamento_{file}"
    plt.savefig(os.path.join(path_output, nome_saida))
    plt.close()

print("Questão 7 concluída! Aguçamento por gradiente e Laplaciano realizados.")
