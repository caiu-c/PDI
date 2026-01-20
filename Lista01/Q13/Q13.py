# Questão 13 - Operações Lógicas entre Imagens Binárias
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_Binarias'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista01\Q13\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# Listar arquivos binários
files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]

if len(files) >= 2:
    # 2. Carregar duas imagens binárias
    img1 = cv2.imread(os.path.join(path_input, files[0]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(path_input, files[1]), cv2.IMREAD_GRAYSCALE)

    if img1 is not None and img2 is not None:
        # Garantir que tenham o mesmo tamanho
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Binarização explícita para garantir que sejam 0 ou 255
        _, bin1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
        _, bin2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)

        # 3. Aplicar Operações Lógicas
        op_and = cv2.bitwise_and(bin1, bin2)
        op_or  = cv2.bitwise_or(bin1, bin2)
        op_xor = cv2.bitwise_xor(bin1, bin2)
        op_not1 = cv2.bitwise_not(bin1)

        # 4. Exibição e Comparação
        titles = ['Imagem 1', 'Imagem 2', 'AND (Interseção)', 'OR (União)', 'XOR', 'NOT (Img 1)']
        images = [bin1, bin2, op_and, op_or, op_xor, op_not1]

        plt.figure(figsize=(20, 10))
        for i in range(6):
            plt.subplot(2, 3, i+1)
            plt.imshow(images[i], cmap='gray')
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()

        # 5. Salvar resultado
        nome_saida = f"q13_operacoes_logicas_{files[0]}_{files[1]}.png"
        plt.savefig(os.path.join(path_output, nome_saida))
        plt.close()

print("Questão 13 concluída! Operações lógicas aplicadas com sucesso.")
