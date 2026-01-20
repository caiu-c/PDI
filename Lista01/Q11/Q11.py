# Questão 11 - Comparação entre Limiarização Manual e de Otsu
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista01\Q11\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

files = os.listdir(path_input)

for file in files:
    if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
        continue

    # 2. Carregar imagem em tons de cinza
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    # 3. Aplicar Limiarização Global Fixa (referência da Questão 9)
    # Usando T=127 como exemplo de limiar manual
    _, thresh_manual = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 4. Aplicar Limiarização Automática de Otsu
    # O valor 0 é passado como limiar inicial, pois o algoritmo irá calculá-lo
    otsu_threshold, thresh_otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Exibição e Comparação
    plt.figure(figsize=(18, 6))
    
    # Imagem Original
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original (Grayscale)')
    plt.axis('off')

    # Histograma com a indicação do Limiar de Otsu
    plt.subplot(1, 3, 2)
    plt.hist(img.ravel(), 256, [0, 256], color='black')
    plt.axvline(otsu_threshold, color='r', linestyle='--', label=f'Otsu T={int(otsu_threshold)}')
    plt.title('Histograma e Limiar Calculado')
    plt.legend()

    # Comparativo: Manual vs Otsu
    # Criamos uma visualização lado a lado para o resultado binário
    plt.subplot(1, 3, 3)
    # Concatenando horizontalmente para comparação direta
    comparison = np.hstack((thresh_manual, thresh_otsu))
    plt.imshow(comparison, cmap='gray')
    plt.title(f'Manual (T=127) vs Otsu (T={int(otsu_threshold)})')
    plt.axis('off')

    plt.tight_layout()

    # 6. Salvar resultado
    nome_saida = f"q11_otsu_comparativo_{file}"
    plt.savefig(os.path.join(path_output, nome_saida))
    plt.close()

print(f"Questão 11 concluída! Resultados salvos em {path_output}")
