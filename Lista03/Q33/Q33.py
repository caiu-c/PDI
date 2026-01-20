import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_Binarias'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q33\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# 2. Seleção de 3 imagens aleatórias
extensoes = ('.png', '.jpg', '.tif', '.bmp', '.gif')
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(extensoes)]
files = random.sample(all_files, 3)

for file in files:
    img_path = os.path.join(path_input, file)
    # Watershed exige imagem colorida (3 canais) para pintar as bordas
    img = cv2.imread(img_path)
    if img is None: continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Binarização Inversa (Objetos em branco, fundo em preto)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Processamento Morfológico para Marcadores
    kernel = np.ones((3,3), np.uint8)
    
    # Remoção de pequenos ruídos
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure Background: Área que temos certeza que é fundo (dilatação)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure Foreground: Área que temos certeza que é objeto (Transformada de Distância)
    # Quanto mais longe da borda, maior o valor. Pegamos os 'picos' (centros).
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Região Desconhecida: Onde as 'águas' irão se encontrar para definir a borda
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 4. Rotulagem dos Marcadores
    _, markers = cv2.connectedComponents(sure_fg)
    # O fundo deve ser 1, os objetos > 1 e o desconhecido 0
    markers = markers + 1
    markers[unknown == 255] = 0

    # 5. Aplicação do Watershed
    img_result = img.copy()
    markers = cv2.watershed(img_result, markers)
    # Pintar as linhas divisórias de Vermelho
    img_result[markers == -1] = [255, 0, 0]

    # --- PLOTAGEM COM LAYOUT SEGURO ---
    plt.figure(figsize=(20, 6))
    
    # Preparação das imagens para exibição
    # Normalizamos a transf. de distância para visualização
    dist_vis = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    titles = [f'Original ({file})', 'Transf. de Distancia', 'Marcadores (Heatmap)', 'Segmentacao Watershed']
    
    images = [
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        dist_vis,
        markers,
        cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
    ]

    for i in range(4):
        plt.subplot(1, 4, i+1)
        if i == 2:
            plt.imshow(images[i], cmap='jet') # Heatmap para os IDs dos objetos
        elif i == 1:
            plt.imshow(images[i], cmap='gray')
        else:
            plt.imshow(images[i])
            
        plt.title(titles[i], fontsize=13, pad=25)
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    
    nome_saida = f'q33_watershed_final_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Questão 33 concluída! Resultados salvos em: {path_output}")