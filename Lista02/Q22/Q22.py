import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt # Necessário: pip install PyWavelets

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista02\Q22\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

def organizar_wavelet(coeffs):
    """
    Organiza os coeficientes da DWT no layout visual clássico.
    """
    LL = coeffs[0]
    for i in range(1, len(coeffs)):
        LH, HL, HH = coeffs[i]
        # Empilha horizontalmente (LL e LH) e (HL e HH)
        top = np.hstack((LL, LH))
        bottom = np.hstack((HL, HH))
        # Empilha verticalmente para formar o quadrante completo
        LL = np.vstack((top, bottom))
    return LL

files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.tif'))]

for file in files:
    img = cv2.imread(os.path.join(path_input, file), cv2.IMREAD_GRAYSCALE)
    if img is None: continue

    # 2. Aplicar Transformada de Haar em 3 níveis
    # pywt.wavedec2 retorna [LLn, (LHn, HLn, HHn), ..., (LH1, HL1, HH1)]
    coeffs = pywt.wavedec2(img, 'haar', level=3)
    
    # 3. Visualização dos níveis individuais e composição
    # Nível 1
    c1 = pywt.wavedec2(img, 'haar', level=1)
    vis_l1 = organizar_wavelet(c1)
    
    # Nível 2
    c2 = pywt.wavedec2(img, 'haar', level=2)
    vis_l2 = organizar_wavelet(c2)
    
    # Nível 3
    vis_l3 = organizar_wavelet(coeffs)

    # 4. Plotagem
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(vis_l1, cmap='gray', vmin=0, vmax=255)
    plt.title('Decomposição Nível 1')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(vis_l2, cmap='gray', vmin=0, vmax=255)
    plt.title('Decomposição Nível 2')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(vis_l3, cmap='gray', vmin=0, vmax=255)
    plt.title('Decomposição Nível 3')
    plt.axis('off')

    plt.tight_layout()
    nome_saida = f'q22_haar_decomposicao_{file}.png'
    plt.savefig(os.path.join(path_output, nome_saida), dpi=300, bbox_inches='tight')
    plt.close()

print("Questão 22 concluída! Decomposições de Haar geradas.")