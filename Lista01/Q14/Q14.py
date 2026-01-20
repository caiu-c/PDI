import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_Binarias'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista01\Q14\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]

for file in files:
    # 2. Carregar imagem binária
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    
    rows, cols = img.shape

    # --- TRANSFORMAÇÕES INDIVIDUAIS ---

    # A) Translação: Deslocamento de 50px em X e 30px em Y
    M_trans = np.float32([[1, 0, 50], [0, 1, 30]])
    img_trans = cv2.warpAffine(img, M_trans, (cols, rows))

    # B) Rotação: 45 graus em relação ao centro
    center = (cols // 2, rows // 2)
    M_rot = cv2.getRotationMatrix2D(center, 45, 1.0)
    img_rot = cv2.warpAffine(img, M_rot, (cols, rows))

    # C) Escala: Redução de 50%
    img_scale = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    # Adicionando bordas para manter o mesmo tamanho de exibição no grid
    img_scale_v = cv2.copyMakeBorder(img_scale, rows//4, rows//4, cols//4, cols//4, cv2.BORDER_CONSTANT, value=0)
    img_scale_v = cv2.resize(img_scale_v, (cols, rows))

    # D) Cisalhamento (Shear) Horizontal
    M_shear_h = np.float32([[1, 0.2, 0], [0, 1, 0]])
    img_shear_h = cv2.warpAffine(img, M_shear_h, (cols, rows))

    # E) Cisalhamento (Shear) Vertical
    M_shear_v = np.float32([[1, 0, 0], [0.2, 1, 0]])
    img_shear_v = cv2.warpAffine(img, M_shear_v, (cols, rows))

    # --- TRANSFORMAÇÃO COMBINADA ---
    # Rotação seguida de Translação
    M_combined = np.dot(M_trans, np.vstack([M_rot, [0, 0, 1]]))
    img_combined = cv2.warpAffine(img, M_combined, (cols, rows))

    # 3. Exibição e Comparação
    titles = ['Original', 'Translação', 'Rotação', 'Escala', 'Cis. Horiz.', 'Cis. Vert.', 'Combinada']
    images = [img, img_trans, img_rot, img_scale_v, img_shear_h, img_shear_v, img_combined]

    plt.figure(figsize=(20, 10))
    for i in range(len(images)):
        plt.subplot(2, 4, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()

    # 4. Salvar resultado
    nome_saida = f"q14_geometricas_{file}.png"
    plt.savefig(os.path.join(path_output, nome_saida))
    plt.close()

print("Questão 14 concluída! Transformações geométricas aplicadas.")
