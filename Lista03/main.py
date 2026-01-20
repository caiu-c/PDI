import os

# Configuração do caminho base
base_path = r'C:\cod_mestrado\pdi\Agora_vai\Lista03'

def criar_estrutura_lista_03():
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f"Diretório base criado: {base_path}")

    # Loop para criar as pastas das questões 25 a 45
    for i in range(25, 46):
        # Caminhos
        q_folder = os.path.join(base_path, f'Q{i}')
        results_folder = os.path.join(q_folder, 'results')
        py_file = os.path.join(q_folder, f'Q{i}.py')

        # Criar pastas
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        # Criar arquivo .py com um template básico de mestrado
        if not os.path.exists(py_file):
            with open(py_file, 'w', encoding='utf-8') as f:
                f.write(f'import cv2\nimport numpy as np\nimport os\nimport matplotlib.pyplot as plt\n\n'
                        f'# Questao {i} - Processamento Digital de Imagens\n'
                        f'# Aluno: Caio Cavalcanti\n\n'
                        f'path_input = r"C:\\cod_mestrado\\pdi\\BancoImagens_TomCinza"\n'
                        f'path_output = r"{results_folder}"\n\n'
                        f'if not os.path.exists(path_output):\n    os.makedirs(path_output)\n\n'
                        f'print("Processando Questao {i}...")\n')
    
    print("-" * 30)
    print("Estrutura da Lista 03 criada com sucesso!")
    print(f"Local: {base_path}")
    print("-" * 30)

if __name__ == "__main__":
    criar_estrutura_lista_03()