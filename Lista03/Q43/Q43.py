import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Questao 43 - Processamento Digital de Imagens
# Aluno: Caio Cavalcanti

path_input = r"C:\cod_mestrado\pdi\BancoImagens_TomCinza"
path_output = r"C:\cod_mestrado\pdi\Agora_vai\Lista03\Q43\results"

if not os.path.exists(path_output):
    os.makedirs(path_output)

print("Processando Questao 43...")
