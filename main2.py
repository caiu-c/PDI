from pathlib import Path

caminho = Path(r"C:\Users\caio_\Downloads\_DESESPERO_PDI3\fig")

for arquivo in caminho.iterdir():
    if arquivo.is_file():
        print(arquivo.name)
