import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista01\Q17\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

def apply_gamma(image, gamma):
    """
    Aplica a correção gama usando uma Look-Up Table (LUT).
    É mais eficiente do que calcular a potência para cada pixel individualmente.
    """
    # Construir a tabela de consulta: mapeando cada valor [0, 255] para seu novo valor gama
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    
    # Aplicar a transformação usando a tabela
    return cv2.LUT(image, table)

files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]

for file in files:
    # 2. Carregar imagem
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    # 3. Aplicar Transformação Gama com diferentes valores
    # Testando Gama < 1 (Clarear) e Gama > 1 (Escurecer)
    gamma_05 = apply_gamma(img, 0.5) # Clareia a imagem
    gamma_15 = apply_gamma(img, 1.5) # Escurece levemente
    gamma_25 = apply_gamma(img, 2.5) # Escurece significativamente

    # 4. Exibição e Comparação
    titles = ['Original', 'Gama = 0.5 (Claro)', 'Gama = 1.5', 'Gama = 2.5 (Escuro)']
    images = [img, gamma_05, gamma_15, gamma_25]

    plt.figure(figsize=(20, 5))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()

    # 5. Salvar resultado
    nome_saida = f"q17_gama_{file}.png"
    plt.savefig(os.path.join(path_output, nome_saida))
    plt.close()

print("Questão 17 concluída! Correção Gama aplicada em todas as imagens.")
