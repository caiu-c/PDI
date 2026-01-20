# Questão 2 - Comparação entre Filtro de Média e Mediana
import os
import cv2
import matplotlib.pyplot as plt

# Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista01\Q02\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

files = os.listdir(path_input)

for file in files:
    if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
        continue

    # Carregar imagem
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path)
    if img is None: continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Aplicar Filtro da Média (Kernel 5x5)
    mean_blur = cv2.blur(img_rgb, (5, 5))

    # Aplicar Filtro da Mediana (Kernel 5x5)
    median_blur = cv2.medianBlur(img_rgb, 5)

    # Exibição e Comparação
    titles = ['Original', 'Média (5x5)', 'Mediana (5x5)']
    images = [img_rgb, mean_blur, median_blur]

    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()

    # Salvar resultado da Questão 2
    nome_saida = f"q2_comparativo_{file}"
    plt.savefig(os.path.join(path_output, nome_saida))
    plt.show()

    plt.close()

print("Questão 2 concluída! Verifique a pasta de resultados.")