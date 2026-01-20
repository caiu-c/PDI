# Questão 9 - Limiarização Simples com Diferentes Valores de Limiar (T)
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_Binarias'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista01\Q09\results'

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

    # 3. Aplicar Limiarização Simples com diferentes valores de limiar (T)
    _, t_50  = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    _, t_127 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    _, t_200 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    # 4. Exibição e Comparação
    titles = ['Original (Grayscale)', 'Limiar T=50', 'Limiar T=127', 'Limiar T=200']
    images = [img, t_50, t_127, t_200]

    plt.figure(figsize=(20, 5))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()

    # 5. Salvar resultado
    nome_saida = f"q9_limiarizacao_{file}"
    plt.savefig(os.path.join(path_output, nome_saida))
    plt.close()

print("Questão 9 concluída! Verifique a pasta Q09/results.")
