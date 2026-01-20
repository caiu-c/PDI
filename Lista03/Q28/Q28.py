import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# ---------------------------------------------------------
# 1. CONFIGURAÇÃO DE DIRETÓRIOS E VARIÁVEIS GLOBAIS
# ---------------------------------------------------------
path_input = r'C:\cod_mestrado\pdi\BancoImagens_Binarias'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q28\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# Estado global para persistência entre eventos de mouse
img_bin_atual = None
result_img = None
seed_point = None

# ---------------------------------------------------------
# 2. LÓGICA DE INTERAÇÃO (CALLBACK)
# ---------------------------------------------------------
def callback_region_growing(event, x, y, flags, param):
    """
    Gerencia a seleção da semente e execução do floodFill.
    """
    global result_img, seed_point, img_bin_atual
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Validação do ponto de semente sobre a região de interesse (pixels brancos)
        if img_bin_atual[y, x] == 255:
            # Geração de buffer colorido para visualização do crescimento
            temp_result = cv2.cvtColor(img_bin_atual, cv2.COLOR_GRAY2BGR)
            
            # Máscara auxiliar para o algoritmo floodFill (requisito OpenCV: dimensão + 2)
            h, w = img_bin_atual.shape
            mask = np.zeros((h + 2, w + 2), np.uint8)
            
            # Execução do Region Growing (Cor: Verde [0, 255, 0])
            cv2.floodFill(temp_result, mask, (x, y), (0, 255, 0))
            
            result_img = temp_result
            seed_point = (x, y)
        else:
            print("[Aviso] Selecione um ponto pertencente ao objeto (Pixel 255).")

# ---------------------------------------------------------
# 3. SELEÇÃO DE AMOSTRAS E PROCESSAMENTO
# ---------------------------------------------------------
extensoes = ('.png', '.jpg', '.tif', '.bmp', '.gif')
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(extensoes)]
files = random.sample(all_files, min(3, len(all_files)))

for file in files:
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    # Normalização para binarização estrita
    _, img_bin_atual = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    result_img = cv2.cvtColor(img_bin_atual, cv2.COLOR_GRAY2BGR)
    seed_point = None

    # Inicialização da interface gráfica
    window_name = f"Segmentacao: {file}"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, callback_region_growing)

    # Loop de interação em tempo real
    while True:
        cv2.imshow(window_name, result_img)
        key = cv2.waitKey(1) & 0xFF
        
        # 'S' ou 'Espaço' para confirmar segmentação
        if (key == ord('s') or key == 32) and seed_point is not None:
            break
        # 'R' para resetar a seleção
        elif key == ord('r'):
            result_img = cv2.cvtColor(img_bin_atual, cv2.COLOR_GRAY2BGR)
            seed_point = None
        # 'Q' para encerrar a execução
        elif key == ord('q'):
            cv2.destroyAllWindows()
            exit()

    cv2.destroyAllWindows()

    # ---------------------------------------------------------
    # 4. EXPORTAÇÃO E GERAÇÃO DE RELATÓRIO VISUAL
    # ---------------------------------------------------------
    plt.figure(figsize=(16, 7))
    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    titles = [f'Entrada Binária ({file})', 'Resultado: Crescimento de Região']
    images = [img_bin_atual, result_rgb]

    for i in range(2):
        plt.subplot(1, 2, i+1)
        if i == 0:
            plt.imshow(images[i], cmap='gray')
            # Marcação visual da semente selecionada
            plt.scatter(seed_point[0], seed_point[1], c='red', s=50, label='Semente')
            plt.legend(loc='upper right')
        else:
            plt.imshow(images[i])
            
        plt.title(titles[i], fontsize=14, pad=25)
        plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    nome_saida = f'q28_region_growing_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Processamento concluído. Resultados exportados para: {path_output}")