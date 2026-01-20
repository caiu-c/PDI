import os

# Definição do novo caminho base para a Lista 02
BASE_PATH = r'C:\cod_mestrado\pdi\Agora_vai\Lista02'

def preparar_ambiente_lista_02():
    # Questões solicitadas: 18 a 24
    questoes = [f"Q{i}" for i in range(18, 25)]
    
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
        print(f"Diretório base criado: {BASE_PATH}")

    print("\nIniciando criação da estrutura...")
    print("-" * 30)

    for q_id in questoes:
        # Caminho da pasta da questão (ex: Lista02/Q18)
        q_path = os.path.join(BASE_PATH, q_id)
        # Caminho da pasta de resultados (ex: Lista02/Q18/results)
        res_path = os.path.join(q_path, "results")
        
        # Criar pastas
        os.makedirs(res_path, exist_ok=True)
        
        # Criar arquivo Python placeholder (ex: Q18.py)
        script_name = f"{q_id}.py"
        script_path = os.path.join(q_path, script_name)
        
        if not os.path.exists(script_path):
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(f'# Código para a {q_id}\n')
                f.write('import cv2\nimport numpy as np\nimport os\n\n')
                f.write('print("Script da ' + q_id + ' inicializado.")\n')
        
        print(f"Estrutura configurada: {q_id} [Pasta + results + {script_name}]")

    print("-" * 30)
    print("Ambiente da Lista 02 pronto para o desenvolvimento.")

if __name__ == "__main__":
    preparar_ambiente_lista_02()