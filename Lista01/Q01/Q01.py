# Questão 1 - Aplicação do Filtro da Média em Imagens
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Definindo os caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista01\Q01\results'

# Cria a pasta de resultados se ela não existir
if not os.path.exists(path_output):
    os.makedirs(path_output)

files = os.listdir(path_input)

print(f"Arquivos encontrados: {files}")

for file in files:
    # Ignorar arquivos que não sejam imagens (opcional)
    if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
        continue

    # Carregar a imagem
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path)
    
    if img is None:
        continue

    # Converter para RGB para exibição correta no Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Aplicar o filtro da média
    blur_3x3 = cv2.blur(img_rgb, (3, 3))
    blur_5x5 = cv2.blur(img_rgb, (5, 5))
    blur_7x7 = cv2.blur(img_rgb, (7, 7))

    # Preparar a exibição/figura
    titles = ['Original', 'Média 3x3', 'Média 5x5', 'Média 7x7']
    images = [img_rgb, blur_3x3, blur_5x5, blur_7x7]

    plt.figure(figsize=(15, 10))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()

    # Salvar o resultado
    nome_saida = f"resultado_{file}"
    caminho_final = os.path.join(path_output, nome_saida)

    plt.savefig(caminho_final)
    print(f"Salvo: {nome_saida}")

    # Fechar a figura para liberar memória (importante para muitas imagens)
    plt.close()

print("Processamento concluído!")