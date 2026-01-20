# Questão 4 - Filtro Laplaciano
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista01\Q04\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

files = os.listdir(path_input)

for file in files:
    if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
        continue

    # 2. Carregar imagem em tons de cinza (filtros derivativos são aplicados em 1 canal)
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    # 3. Definir máscaras do Laplaciano
    kernel_std = np.array([[0,  1, 0],
                           [1, -4, 1],
                           [0,  1, 0]], dtype=np.float32)

    kernel_diag = np.array([[1,  1, 1],
                            [1, -8, 1],
                            [1,  1, 1]], dtype=np.float32)

    # 4. Aplicar os filtros usando filter2D
    lap_std = cv2.filter2D(img, cv2.CV_64F, kernel_std)
    lap_diag = cv2.filter2D(img, cv2.CV_64F, kernel_diag)

    # Converter de volta para uint8 (Pegando o valor absoluto)
    lap_std_abs = cv2.convertScaleAbs(lap_std)
    lap_diag_abs = cv2.convertScaleAbs(lap_diag)

    # 5. Exibição e Comparação
    titles = ['Original', 'Laplaciano (Simples)', 'Laplaciano (Diagonais)']
    images = [img, lap_std_abs, lap_diag_abs]

    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()

    # 6. Salvar resultado
    nome_saida = f"q4_laplaciano_{file}"
    plt.savefig(os.path.join(path_output, nome_saida))
    plt.close()

print("Questão 4 concluída! Verifique a pasta Q04/results.")