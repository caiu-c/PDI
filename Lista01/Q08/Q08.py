# Questão 8 - Histogram Matching
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Configuração de caminhos
path_input = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista01\Q08\results'

if not os.path.exists(path_output):
    os.makedirs(path_output)

def find_lut(src_hist, ref_hist):
    """Função auxiliar para encontrar a Tabela de Consulta (LUT) para Histogram Matching."""
    src_cdf = src_hist.cumsum()
    src_cdf_normalized = src_cdf / src_cdf.max()
    
    ref_cdf = ref_hist.cumsum()
    ref_cdf_normalized = ref_cdf / ref_cdf.max()
    
    lut = np.zeros(256)
    ref_idx = 0
    for src_idx in range(256):
        while ref_idx < 255 and ref_cdf_normalized[ref_idx] < src_cdf_normalized[src_idx]:
            ref_idx += 1
        lut[src_idx] = ref_idx
    return lut.astype('uint8')

files = [f for f in os.listdir(path_input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]

# Para o Histogram Matching, precisamos de pelo menos duas imagens no diretório
if len(files) >= 2:
    # Usaremos a primeira imagem como origem e a segunda como referência (target)
    for i in range(len(files)):
        file_src = files[i]
        # Escolhe uma imagem diferente para ser a referência
        file_ref = files[(i + 1) % len(files)]
        
        # 2. Carregar Imagens
        img_src = cv2.imread(os.path.join(path_input, file_src), cv2.IMREAD_GRAYSCALE)
        img_ref = cv2.imread(os.path.join(path_input, file_ref), cv2.IMREAD_GRAYSCALE)
        
        if img_src is None or img_ref is None: continue

        # --- ETAPA 1: Equalização de Histograma ---
        img_equalized = cv2.equalizeHist(img_src)

        # --- ETAPA 2: Especificação de Histograma (Matching) ---
        # Calcular histogramas para obter as CDFs
        hist_src = cv2.calcHist([img_src], [0], None, [256], [0, 256])
        hist_ref = cv2.calcHist([img_ref], [0], None, [256], [0, 256])
        
        lut = find_lut(hist_src, hist_ref)
        img_matched = cv2.LUT(img_src, lut)

        # --- ETAPA 3: Preparar Exibição ---
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        
        # Imagem Original e seu Histograma
        axs[0, 0].imshow(img_src, cmap='gray')
        axs[0, 0].set_title(f'Original: {file_src}')
        axs[0, 0].axis('off')
        axs[1, 0].hist(img_src.ravel(), 256, [0, 256], color='black')
        axs[1, 0].set_title('Histograma Original')

        # Imagem Equalizada e seu Histograma
        axs[0, 1].imshow(img_equalized, cmap='gray')
        axs[0, 1].set_title('Equalizada')
        axs[0, 1].axis('off')
        axs[1, 1].hist(img_equalized.ravel(), 256, [0, 256], color='black')
        axs[1, 1].set_title('Histograma Equalizado')

        # Imagem de Referência e seu Histograma
        axs[0, 2].imshow(img_ref, cmap='gray')
        axs[0, 2].set_title(f'Referência: {file_ref}')
        axs[0, 2].axis('off')
        axs[1, 2].hist(img_ref.ravel(), 256, [0, 256], color='black')
        axs[1, 2].set_title('Histograma Referência')

        # Imagem com Matching e seu Histograma
        axs[0, 3].imshow(img_matched, cmap='gray')
        axs[0, 3].set_title('Matched (Src -> Ref)')
        axs[0, 3].axis('off')
        axs[1, 3].hist(img_matched.ravel(), 256, [0, 256], color='black')
        axs[1, 3].set_title('Histograma Matched')

        plt.tight_layout()
        
        # 4. Salvar Resultado
        nome_saida = f"q8_analise_histograma_{file_src}.png"
        plt.savefig(os.path.join(path_output, nome_saida))
        plt.close()

print("Questão 8 concluída! Verifique a pasta Q08/results.")
