import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. CONFIGURAÇÃO DE DIRETÓRIOS
# ---------------------------------------------------------
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q25\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# ---------------------------------------------------------
# 2. DEFINIÇÃO DOS ELEMENTOS ESTRUTURANTES (Kernel 5x5)
# ---------------------------------------------------------
kernels = {
    "Retangular": cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    "Cruz": cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)),
    "Eliptico": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
}

# ---------------------------------------------------------
# 3. PROCESSAMENTO DE IMAGENS
# ---------------------------------------------------------
# Filtragem de arquivos suportados
files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.tif'))]

for file in files:
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        continue

    # Binarização automática via Método de Otsu
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    for name, k in kernels.items():
        # Operações Morfológicas: Erosão e Dilatação
        img_erosion = cv2.erode(img_bin, k, iterations=1)
        img_dilation = cv2.dilate(img_bin, k, iterations=1)

        # Configuração da visualização
        plt.figure(figsize=(16, 6))
        
        titles = ['Original Binária', f'Erosão ({name})', f'Dilatação ({name})']
        images = [img_bin, img_erosion, img_dilation]

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.imshow(images[i], cmap='gray')
            plt.title(titles[i], fontsize=13, pad=15)
            plt.axis('off')

        # Ajuste de layout para evitar sobreposição de títulos (reserva de espaço superior)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Exportação dos resultados
        nome_saida = f'q25_{name}_{file}.png'
        plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
        plt.close()

print(f"Processamento concluído. Resultados salvos em: {path_output}")