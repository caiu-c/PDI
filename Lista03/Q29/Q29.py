import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q29\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# Variáveis globais
img_gray_atual = None
result_img = None
seed_point = None
# Critério de similaridade: tolerância de intensidade (0-255)
tolerancia = 20 

def callback_region_growing_gray(event, x, y, flags, param):
    global result_img, seed_point, img_gray_atual, tolerancia
    
    if event == cv2.EVENT_LBUTTONDOWN:
        seed_point = (x, y)
        h, w = img_gray_atual.shape
        
        # Criar cópia colorida para visualização do resultado
        temp_result = cv2.cvtColor(img_gray_atual, cv2.COLOR_GRAY2BGR)
        
        # Máscara para o floodFill (tamanho original + 2 pixels)
        mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # Parâmetros: (imagem, máscara, semente, nova_cor, diff_baixa, diff_alta, flags)
        # O flag cv2.FLOODFILL_FIXED_RANGE compara cada pixel com a semente inicial
        cv2.floodFill(temp_result, mask, seed_point, (0, 255, 0), 
                      (tolerancia, tolerancia, tolerancia), 
                      (tolerancia, tolerancia, tolerancia), 
                      flags=4 | cv2.FLOODFILL_FIXED_RANGE)
        
        result_img = temp_result
        print(f"Semente: {seed_point} | Intensidade: {img_gray_atual[y, x]} | Tolerância: {tolerancia}")

# 2. Seleção de 3 imagens aleatórias
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.tif', '.bmp'))]
files = random.sample(all_files, 3)

for file in files:
    img_path = os.path.join(path_input, file)
    img_gray_atual = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray_atual is None: continue

    result_img = cv2.cvtColor(img_gray_atual, cv2.COLOR_GRAY2BGR)
    seed_point = None

    window_name = f"Q29 - {file} (Tolerancia: {tolerancia})"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, callback_region_growing_gray)

    print(f"\nProcessando: {file}")
    print("- Clique para definir a semente.")
    print("- Pressione 'S' ou ESPAÇO para salvar.")
    print("- Pressione '+' para aumentar tolerância ou '-' para diminuir.")

    while True:
        cv2.imshow(window_name, result_img)
        key = cv2.waitKey(1) & 0xFF
        
        if (key == ord('s') or key == 32) and seed_point is not None:
            break
        elif key == ord('+'):
            tolerancia = min(255, tolerancia + 5)
            print(f"Nova Tolerância: {tolerancia}")
        elif key == ord('-'):
            tolerancia = max(0, tolerancia - 5)
            print(f"Nova Tolerância: {tolerancia}")
        elif key == ord('q'):
            cv2.destroyAllWindows()
            exit()

    cv2.destroyAllWindows()

    # --- PLOTAGEM PARA RELATÓRIO ---
    plt.figure(figsize=(16, 7))
    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    titles = [f'Original ({file})', f'Segmentação (Tolerância: {tolerancia})']
    images = [img_gray_atual, result_rgb]

    for i in range(2):
        plt.subplot(1, 2, i+1)
        if i == 0:
            plt.imshow(images[i], cmap='gray')
            plt.scatter(seed_point[0], seed_point[1], c='red', s=50, label='Semente')
            plt.legend()
        else:
            plt.imshow(images[i])
            
        plt.title(titles[i], fontsize=14, pad=25)
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig(os.path.join(path_output, f'q29_gray_growing_{file}.png'), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Questão 29 finalizada.")