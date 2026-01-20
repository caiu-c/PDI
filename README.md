# Projeto PDI - Processamento Digital de Imagens

Este é um projeto acadêmico do Mestrado em Processamento Digital de Imagens (PDI) contendo soluções para listas de exercícios.

## 📋 Estrutura do Projeto

O projeto está organizado em três listas de exercícios:

- **Lista01/**: Exercícios 01 a 17 - Fundamentos de processamento de imagens
- **Lista02/**: Exercícios 18 a 24 - Técnicas intermediárias
- **Lista03/**: Exercícios 25 a 45 - Operações avançadas

Cada exercício possui:
- Um arquivo Python principal (`QXX.py`)
- Uma pasta `results/` para armazenar os resultados gerados

## 🔧 Configuração Necessária

### ⚠️ IMPORTANTE: Alterar Caminhos de Input e Output

Antes de executar qualquer arquivo Python, é necessário alterar os caminhos de entrada e saída no código para corresponder à sua estrutura de diretórios local.

**Passos:**

1. Abra o arquivo Python desejado (ex: `Lista01/Q01/Q01.py`)
2. Localize as variáveis que definem os caminhos:
   - `input_path`: Caminho para as imagens de entrada
   - `output_path`: Caminho para salvar os resultados
   - Caminhos de leitura/escrita dentro do código

3. Atualize os caminhos para sua máquina local:

**Exemplo - Antes:**
```python
input_path = "C:/Users/seu_usuario/imagens/"
output_path = "./results/"
```

**Exemplo - Depois (com seus caminhos):**
```python
input_path = "c:/seu_caminho/imagens/"
output_path = "./results/"
```

## 🚀 Como Executar

1. Navegue até a pasta do exercício desejado:
```bash
cd Lista01/Q01
```

2. Execute o arquivo Python:
```bash
python Q01.py
```

3. Os resultados serão salvos na pasta `results/`

## 📁 Tipos de Arquivos Esperados

- Imagens de entrada: `.jpg`, `.png`, `.bmp`, `.tiff`, etc.
- Resultados gerados: Imagens processadas em diversos formatos

## 🔍 Notas

- Certifique-se de que todas as bibliotecas necessárias estão instaladas
- Ajuste os caminhos em CADA arquivo Python antes de executar
- Os caminhos relativos funcionam melhor dentro de cada pasta de exercício

---

**Autor**: Caio Cavalcanti  
**Data de última alteração**: 20/01/2026
