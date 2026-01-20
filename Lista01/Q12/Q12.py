# Questão 12 - Operações Aritméticas entre Duas Imagens
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista01\Q12\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# Listar arquivos e selecionar dois para a operação
files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]

if len(files) >= 2:
    # 2. Carregar as duas primeiras imagens do diretório
    img1 = cv2.imread(os.path.join(path_input, files[0]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(path_input, files[1]), cv2.IMREAD_GRAYSCALE)

    if img1 is not None and img2 is not None:
        # Garantir que as imagens tenham o mesmo tamanho para operações aritméticas
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # 3. Realizar Operações Aritméticas
        # cv2.add e cv2.subtract realizam a operação com saturação (clip entre 0-255)
        soma = cv2.add(img1, img2)
        subtracao = cv2.subtract(img1, img2)
        
        # Multiplicação e Divisão costumam exigir conversão para float para evitar perda de dados
        # Multiplicação (exemplo: aplicação de ganho ou máscara)
        multiplicacao = cv2.multiply(img1.astype(np.float32), img2.astype(np.float32))
        multiplicacao = cv2.normalize(multiplicacao, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Divisão (útil para correção de iluminação ou detecção de proporção)
        # Adiciona-se 1 para evitar divisão por zero
        divisao = cv2.divide(img1.astype(np.float32), img2.astype(np.float32) + 1.0)
        divisao = cv2.normalize(divisao, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 4. Exibição e Comparação
        titles = ['Imagem 1', 'Imagem 2', 'Soma', 'Subtração', 'Multiplicação', 'Divisão']
        images = [img1, img2, soma, subtracao, multiplicacao, divisao]

        plt.figure(figsize=(20, 10))
        for i in range(6):
            plt.subplot(2, 3, i+1)
            plt.imshow(images[i], cmap='gray')
            plt.title(titles[i])
            plt.axis('off')

        plt.tight_layout()

        # 5. Salvar resultado
        nome_saida = f"q12_operacoes_aritméticas_{files[0]}_e_{files[1]}.png"
        plt.savefig(os.path.join(path_output, nome_saida))
        plt.close()

print("Questão 12 concluída! Operações aritméticas entre imagens realizadas.")
