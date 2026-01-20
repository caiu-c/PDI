# Questão 5 - Aplicação do Filtro de Prewitt
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista01\Q05\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

files = os.listdir(path_input)

for file in files:
    if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
        continue

    # 2. Carregar imagem
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    # 3. Definir Kernels de Prewitt
    # Gradiente Horizontal (Detecta bordas verticais)
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]], dtype=np.float32)
    
    # Gradiente Vertical (Detecta bordas horizontais)
    kernel_y = np.array([[-1, -1, -1],
                         [ 0, 0, 0],
                         [ 1, 1, 1]], dtype=np.float32)

    # 4. Aplicar Convolução (cv2.filter2D)
    prewitt_x = cv2.filter2D(img, cv2.CV_64F, kernel_x)
    prewitt_y = cv2.filter2D(img, cv2.CV_64F, kernel_y)

    # Calcular Magnitude do Gradiente: sqrt(Gx^2 + Gy^2)
    magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)
    magnitude = cv2.convertScaleAbs(magnitude)

    # 5. Exibição/Resultados
    titles = ['Original', 'Prewitt Horizontal', 'Prewitt Vertical', 'Magnitude Final']
    images = [img, cv2.convertScaleAbs(prewitt_x), cv2.convertScaleAbs(prewitt_y), magnitude]

    plt.figure(figsize=(20, 5))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()

    # 6. Salvar resultado
    nome_saida = f"q5_prewitt_{file}"
    plt.savefig(os.path.join(path_output, nome_saida))
    plt.close()

print("Questão 5 concluída! Verifique a pasta Q05/results.")
