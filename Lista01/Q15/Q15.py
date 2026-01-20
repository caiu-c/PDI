import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista01\Q15\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]

for file in files:
    # 2. Carregar imagem
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    # 3. Aplicar Transformação Negativa
    # Para imagens uint8 (0-255), o negativo é simplesmente 255 - pixel
    img_negative = 255 - img

    # 4. Exibição e Comparação
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f'Original: {file}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_negative, cmap='gray')
    plt.title('Transformação Negativa')
    plt.axis('off')

    plt.tight_layout()

    # 5. Salvar resultado
    nome_saida = f"q15_negativo_{file}.png"
    plt.savefig(os.path.join(path_output, nome_saida))
    plt.close()

print("Questão 15 concluída! Transformação negativa aplicada.")
