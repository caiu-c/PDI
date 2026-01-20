import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração de caminhos
input_files = [
    r'C:\cod_mestrado\pdi\Agora_vai\Lista02\webcam-toy-photo2.jpg',
    r'C:\cod_mestrado\pdi\Agora_vai\Lista02\webcam-toy-photo3.jpg',
    r'C:\cod_mestrado\pdi\Agora_vai\Lista02\webcam-toy-photo4.jpg',

]
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista02\Q23\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

def detect_skin(image):
    # --- Processamento no Modelo HSV ---
    # O modelo HSV separa matiz (H), saturação (S) e valor (V)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Limiares típicos para pele no espaço HSV
    lower_hsv = np.array([0, 48, 80], dtype="uint8")
    upper_hsv = np.array([20, 255, 255], dtype="uint8")
    mask_hsv = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
    res_hsv = cv2.bitwise_and(image, image, mask=mask_hsv)

    # --- Processamento no Modelo YCrCb ---
    # Y (Luminância), Cr (Crominância Vermelha), Cb (Crominância Azul)
    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # Limiares típicos (Cr entre 133-173 e Cb entre 77-127)
    lower_ycrcb = np.array([0, 133, 77], dtype="uint8")
    upper_ycrcb = np.array([255, 173, 127], dtype="uint8")
    mask_ycrcb = cv2.inRange(img_ycrcb, lower_ycrcb, upper_ycrcb)
    res_ycrcb = cv2.bitwise_and(image, image, mask=mask_ycrcb)

    return res_hsv, res_ycrcb

for file_path in input_files:
    if not os.path.exists(file_path):
        print(f"Arquivo não encontrado: {file_path}")
        continue

    img = cv2.imread(file_path)
    if img is None: continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    skin_hsv, skin_ycrcb = detect_skin(img)
    skin_hsv_rgb = cv2.cvtColor(skin_hsv, cv2.COLOR_BGR2RGB)
    skin_ycrcb_rgb = cv2.cvtColor(skin_ycrcb, cv2.COLOR_BGR2RGB)

    # Plotagem Vertical
    plt.figure(figsize=(10, 15))
    
    plt.subplot(3, 1, 1)
    plt.imshow(img_rgb)
    plt.title('Original (Webcam)')
    plt.axis('off')

    plt.subplot(3, 1, 2)
    plt.imshow(skin_hsv_rgb)
    plt.title('Detecção de Pele - Modelo HSV')
    plt.axis('off')

    plt.subplot(3, 1, 3)
    plt.imshow(skin_ycrcb_rgb)
    plt.title('Detecção de Pele - Modelo YCrCb')
    plt.axis('off')

    plt.tight_layout()
    file_name = os.path.basename(file_path)
    nome_saida = f'q23_comparativo_pele_{file_name}'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print("Questão 23 concluída! Detecção de pele realizada em ambos os modelos.")