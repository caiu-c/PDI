import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista01\Q16\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]

for file in files:
    # 2. Carregar imagem
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    # Converter para float para cálculos de precisão
    img_float = img.astype(float)

    # --- TRANSFORMAÇÕES DE INTENSIDADE ---

    # A) Linear (Identidade/Normalizada)
    # Serve como base de comparação para o brilho e contraste original
    img_linear = cv2.normalize(img_float, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # B) Logarítmica: s = c * log(1 + r)
    # Útil para realçar detalhes em imagens muito escuras
    c_log = 255 / np.log(1 + np.max(img_float))
    img_log = c_log * (np.log(1 + img_float))
    img_log = np.array(img_log, dtype=np.uint8)

    # C) Exponencial: s = c * (exp(r) - 1)
    # Nota: Como exp(255) é muito alto, normalizamos a entrada para [0, 1]
    img_norm = img_float / 255.0
    img_exp = np.exp(img_norm) - 1
    # Re-normalizar o resultado para a escala [0, 255]
    img_exp = cv2.normalize(img_exp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 4. Exibição e Comparação
    titles = ['Original (Linear)', 'Logarítmica (Expande Escuros)', 'Exponencial (Expande Claros)']
    images = [img_linear, img_log, img_exp]

    plt.figure(figsize=(18, 6))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()

    # 5. Salvar resultado
    nome_saida = f"q16_intensidade_{file}.png"
    plt.savefig(os.path.join(path_output, nome_saida))
    plt.close()

print("Questão 16 concluída! Comparações de intensidade realizadas.")
