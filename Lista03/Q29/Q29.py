import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# ---------------------------------------------------------
# 1. CONFIGURAÇÃO DE AMBIENTE E PARÂMETROS INICIAIS
# ---------------------------------------------------------
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q29\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# Variáveis globais de controle
img_gray_atual = None
result_img = None
seed_point = None
# Critério de similaridade: tolerância de intensidade (intervalo 0-255)
tolerancia = 20 

# ---------------------------------------------------------
# 2. LÓGICA DE SEGMENTAÇÃO (REGION GROWING)
# ---------------------------------------------------------
def callback_region_growing_gray(event, x, y, flags, param):
    """
    Executa o algoritmo de preenchimento por inundação (Flood Fill) 
    baseado em um critério de intervalo fixo em relação à semente.
    """
    global result_img, seed_point, img_gray_atual, tolerancia
    
    if event == cv2.EVENT_LBUTTONDOWN:
        seed_point = (x, y)
        h, w = img_gray_atual.shape
        
        # Buffer colorido para visualização da máscara segmentada
        temp_result = cv2.cvtColor(img_gray_atual, cv2.COLOR_GRAY2BGR)
        
        # Máscara auxiliar (tamanho da imagem + borda de 2 pixels conforme OpenCV)
        mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # Aplicação do cv2.floodFill:
        # Utiliza-se FLOODFILL_FIXED_RANGE para comparar pixels vizinhos com a semente inicial
        # loDiff e upDiff definem os limites inferiores e superiores da tolerância
        cv2.floodFill(temp_result, mask, seed_point, (0, 255, 0), 
                      (tolerancia, tolerancia, tolerancia), 
                      (tolerancia, tolerancia, tolerancia), 
                      flags=4 | cv2.FLOODFILL_FIXED_RANGE)
        
        result_img = temp_result
        print(f"Log: Semente em {seed_point} | Intensidade: {img_gray_atual[y, x]} | Tolerância Atual: {tolerancia}")

# ---------------------------------------------------------
# 3. SELEÇÃO E PROCESSAMENTO INTERATIVO
# ---------------------------------------------------------
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.tif', '.bmp'))]
files = random.sample(all_files, min(3, len(all_files)))

for file in files:
    img_path = os.path.join(path_input, file)
    img_gray_atual = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray_atual is None: continue

    result_img = cv2.cvtColor(img_gray_atual, cv2.COLOR_GRAY2BGR)
    seed_point = None

    window_name = f"Segmentacao Interativa - Q29 (Tol: {tolerancia})"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, callback_region_growing_gray)

    print(f"\nArquivo em análise: {file}")
    print("Comandos: [S/Espaço] Confirmar | [+] Aumentar Tol | [-] Diminuir Tol | [Q] Sair")

    while True:
        cv2.imshow(window_name, result_img)
        key = cv2.waitKey(1) & 0xFF
        
        if (key == ord('s') or key == 32) and seed_point is not None:
            break
        elif key == ord('+'):
            tolerancia = min(255, tolerancia + 5)
            cv2.setWindowTitle(window_name, f"Segmentacao Interativa - Q29 (Tol: {tolerancia})")
        elif key == ord('-'):
            tolerancia = max(0, tolerancia - 5)
            cv2.setWindowTitle(window_name, f"Segmentacao Interativa - Q29 (Tol: {tolerancia})")
        elif key == ord('q'):
            cv2.destroyAllWindows()
            exit()

    cv2.destroyAllWindows()

    # ---------------------------------------------------------
    # 4. EXPORTAÇÃO DOS RESULTADOS E PLOTAGEM
    # ---------------------------------------------------------
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
    output_filename = f'q29_gray_growing_{file}.png'
    plt.savefig(os.path.join(path_output, output_filename), dpi=300, bbox_inches='tight')
    plt.close()

print(f"\nProcessamento finalizado. Resultados salvos em: {path_output}")