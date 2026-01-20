import os
import shutil
import random

# Configurações de Caminhos
BASE_PATH = r'C:\cod_mestrado\pdi\Agora_vai\Lista01'
FIG_PATH = os.path.join(BASE_PATH, 'fig')

def coletar_balanceado(n_total=10):
    # Divisão solicitada
    topicos = {
        "T1_PassaBaixa": [f"Q{i:02d}" for i in range(1, 4)],
        "T2_PassaAlta": [f"Q{i:02d}" for i in range(4, 8)],
        "T3_Intensidade_Geometria": [f"Q{i:02d}" for i in range(8, 18)]
    }
    
    if not os.path.exists(FIG_PATH):
        os.makedirs(FIG_PATH)

    selecionadas_caminho = []
    
    # 1. Mapear arquivos disponíveis por tópico
    arquivos_por_topico = {t: [] for t in topicos}
    for t, questoes in topicos.items():
        for q in questoes:
            res_path = os.path.join(BASE_PATH, q, "results")
            if os.path.exists(res_path):
                f_list = [os.path.join(res_path, f) for f in os.listdir(res_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
                arquivos_por_topico[t].extend(f_list)

    # 2. Seleção Obrigatória: 2 de cada um dos dois primeiros tópicos
    for t in ["T1_PassaBaixa", "T2_PassaAlta"]:
        amostra = random.sample(arquivos_por_topico[t], min(2, len(arquivos_por_topico[t])))
        selecionadas_caminho.extend(amostra)
        # Remove as selecionadas do pool para não repetir
        for f in amostra:
            arquivos_por_topico[t].remove(f)

    # 3. Completar o restante (6 imagens) com o Tópico 3
    t3_pool = arquivos_por_topico["T3_Intensidade_Geometria"]
    if t3_pool:
        qtd_restante = n_total - len(selecionadas_caminho)
        amostra_t3 = random.sample(t3_pool, min(qtd_restante, len(t3_pool)))
        selecionadas_caminho.extend(amostra_t3)

    # 4. Copiar para \fig e listar
    print("\n" + "="*60)
    print(f" ARQUIVOS COPIADOS PARA: {FIG_PATH}")
    print("="*60)
    
    for origin in selecionadas_caminho:
        nome_f = os.path.basename(origin)
        # Para evitar conflito de nomes iguais em pastas diferentes, 
        # adicionamos o prefixo da questão no nome final
        pasta_q = os.path.basename(os.path.dirname(os.path.dirname(origin)))
        nome_final = f"{pasta_q}_{nome_f}"
        
        shutil.copy2(origin, os.path.join(FIG_PATH, nome_final))
        print(nome_final)
        
    print("="*60)
    print("COLE A LISTA DE NOMES ACIMA NO CHAT.")

if __name__ == "__main__":
    coletar_balanceado(10)