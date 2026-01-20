import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_Binarias'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q28\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# Variáveis globais para controle da interação
img_bin_atual = None
result_img = None
seed_point = None

def callback_region_growing(event, x, y, flags, param):
    global result_img, seed_point, img_bin_atual
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Verificar se o clique foi em um pixel branco (objeto)
        if img_bin_atual[y, x] == 255:
            # Criar cópia colorida para destacar o crescimento em verde
            temp_result = cv2.cvtColor(img_bin_atual, cv2.COLOR_GRAY2BGR)
            
            # Máscara necessária para o floodFill (tamanho original + 2 pixels)
            h, w = img_bin_atual.shape
            mask = np.zeros((h + 2, w + 2), np.uint8)
            
            # Aplicar o preenchimento (Crescimento de Região)
            # Preenchemos com a cor verde (0, 255, 0)
            cv2.floodFill(temp_result, mask, (x, y), (0, 255, 0))
            
            result_img = temp_result
            seed_point = (x, y)
            print(f"Semente selecionada em: ({x}, {y})")
        else:
            print("Clique em um pixel BRANCO (255) para iniciar o crescimento.")

# 2. Seleção de 3 imagens aleatórias
extensoes = ('.png', '.jpg', '.tif', '.bmp', '.gif')
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(extensoes)]
files = random.sample(all_files, 3)

for file in files:
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    # Binarização estrita
    _, img_bin_atual = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    result_img = cv2.cvtColor(img_bin_atual, cv2.COLOR_GRAY2BGR)
    seed_point = None

    # Janela Interativa
    window_name = f"Questao 28 - {file}"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, callback_region_growing)

    print(f"\nProcessando: {file}")
    print("- Clique com o botaão ESQUERDO no objeto.")
    print("- Pressione 'S' ou ESPAÇO para confirmar e salvar.")
    print("- Pressione 'R' para resetar.")

    while True:
        cv2.imshow(window_name, result_img)
        key = cv2.waitKey(1) & 0xFF
        
        # Confirmar e prosseguir
        if (key == ord('s') or key == 32) and seed_point is not None:
            break
        # Resetar a imagem atual
        elif key == ord('r'):
            result_img = cv2.cvtColor(img_bin_atual, cv2.COLOR_GRAY2BGR)
            seed_point = None
        # Sair do programa
        elif key == ord('q'):
            cv2.destroyAllWindows()
            exit()

    cv2.destroyAllWindows()

    # --- SALVAMENTO E PLOTAGEM PARA O RELATÓRIO ---
    plt.figure(figsize=(16, 7))
    
    # Converter para RGB para o Matplotlib exibir o verde corretamente
    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    titles = [f'Entrada Binaria ({file})', 'Crescimento de Regiao (Verde)']
    images = [img_bin_atual, result_rgb]

    for i in range(2):
        plt.subplot(1, 2, i+1)
        if i == 0:
            plt.imshow(images[i], cmap='gray')
            # Marcar o ponto da semente na original
            plt.scatter(seed_point[0], seed_point[1], c='red', s=50, label='Semente')
            plt.legend(loc='upper right')
        else:
            plt.imshow(images[i])
            
        plt.title(titles[i], fontsize=14, pad=25)
        plt.axis('off')

    # Ajuste de layout para evitar corte de títulos
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    nome_saida = f'q28_region_growing_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print(f"\nQuestão 28 concluída! Resultados salvos em {path_output}")