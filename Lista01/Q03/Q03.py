# Questão 3 - Filtro Gaussiano com diferentes Sigmas
import os
import cv2
import matplotlib.pyplot as plt

path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista01\Q03\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

files = os.listdir(path_input)

for file in files:
    if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
        continue

    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path)
    if img is None: continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Aplicar Filtro Gaussiano com diferentes Sigmas
    gauss_s1 = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=1)
    gauss_s3 = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=3)
    gauss_s5 = cv2.GaussianBlur(img_rgb, (0, 0), sigmaX=5)

    # Exibição
    titles = ['Original', 'Gauss σ=1', 'Gauss σ=3', 'Gauss σ=5']
    images = [img_rgb, gauss_s1, gauss_s3, gauss_s5]

    plt.figure(figsize=(15, 5))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()

    # Salvar
    nome_saida = f"q3_gaussiana_{file}"
    plt.savefig(os.path.join(path_output, nome_saida))
    plt.close()

print("Questão 3 concluída!")