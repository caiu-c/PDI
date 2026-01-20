import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_Binarias'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q35\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

# 2. Seleção de 3 imagens aleatórias
extensoes = ('.png', '.jpg', '.tif', '.bmp', '.gif')
all_files = [f for f in os.listdir(path_input) if f.lower().endswith(extensoes)]
files = random.sample(all_files, 3)

for file in files:
    img_path = os.path.join(path_input, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    # Binarização
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Encontrar contorno do objeto principal
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: continue
    c = max(cnts, key=cv2.contourArea)

    # 3. Cálculo dos Momentos de Hu
    moments = cv2.moments(c)
    huMoments = cv2.HuMoments(moments)

    # Transformação Logarítmica (para melhor visualização/comparação)
    # h = -sign(h) * log10(abs(h))
    huLog = []
    for i in range(0, 7):
        val = huMoments[i][0]
        if val != 0:
            log_val = -1 * np.sign(val) * np.log10(np.abs(val))
        else:
            log_val = 0
        huLog.append(log_val)

    # --- VISUALIZAÇÃO ---
    img_out = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_out, [c], -1, (0, 255, 0), 2)

    plt.figure(figsize=(14, 8))
    
    # Subplot 1: Imagem com Contorno
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    plt.title(f"Objeto: {file}", fontsize=14)
    plt.axis('off')

    # Subplot 2: Tabela de Momentos de Hu
    plt.subplot(1, 2, 2)
    plt.axis('off')
    col_labels = ['Momento', 'Valor Log-Transf']
    table_data = [[f"h{i+1}", f"{huLog[i]:.4f}"] for i in range(7)]
    
    table = plt.table(cellText=table_data, colLabels=col_labels, 
                      loc='center', cellLoc='center')
    table.scale(1, 2.5)
    plt.title("Momentos Invariantes de Hu", fontsize=14, pad=20)

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    
    nome_saida = f'q35_hu_moments_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print(f"Questão 35 concluída! Análise de Hu salva em {path_output}")