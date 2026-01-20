import os
import cv2
import numpy as np
import subprocess

# Configurações de Caminhos
BASE_PATH = r'C:\cod_mestrado\pdi\Agora_vai\Lista01'
PATH_CINZA = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
PATH_BIN = r'C:\cod_mestrado\pdi\BancoImagens_Binarias'

def create_structure():
    # Definição dos códigos para cada questão
    scripts = {
        "Q01": f"import cv2, os; p_in, p_out = r'{PATH_CINZA}', 'results'; img = cv2.imread(os.path.join(p_in, os.listdir(p_in)[0])); [cv2.imwrite(f'results/media_{{k}}x{{k}}.png', cv2.blur(img, (k, k))) for k in [3, 5, 7]]",
        "Q02": f"import cv2, os; p_in = r'{PATH_CINZA}'; img = cv2.imread(os.path.join(p_in, os.listdir(p_in)[0])); cv2.imwrite('results/mediana.png', cv2.medianBlur(img, 5)); cv2.imwrite('results/media_comp.png', cv2.blur(img, (5, 5)))",
        "Q03": f"import cv2, os; p_in = r'{PATH_CINZA}'; img = cv2.imread(os.path.join(p_in, os.listdir(p_in)[0])); [cv2.imwrite(f'results/gauss_s{{s}}.png', cv2.GaussianBlur(img, (0, 0), s)) for s in [1, 3, 5]]",
        "Q04": f"import cv2, os, numpy as np; p_in = r'{PATH_CINZA}'; img = cv2.imread(os.path.join(p_in, os.listdir(p_in)[0]), 0); lap = cv2.Laplacian(img, cv2.CV_64F); cv2.imwrite('results/laplaciano.png', np.uint8(np.absolute(lap)))",
        "Q05": f"import cv2, os, numpy as np; p_in = r'{PATH_CINZA}'; img = cv2.imread(os.path.join(p_in, os.listdir(p_in)[0]), 0); kernel = np.array([[-1,-1,-1],[0,0,0],[1,1,1]]); res = cv2.filter2D(img, -1, kernel); cv2.imwrite('results/prewitt_h.png', res)",
        "Q06": f"import cv2, os; p_in = r'{PATH_CINZA}'; img = cv2.imread(os.path.join(p_in, os.listdir(p_in)[0]), 0); sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3); cv2.imwrite('results/sobel_x.png', cv2.convertScaleAbs(sobelx))",
        "Q07": f"import cv2, os, numpy as np; p_in = r'{PATH_CINZA}'; img = cv2.imread(os.path.join(p_in, os.listdir(p_in)[0]), 0); lap = cv2.Laplacian(img, cv2.CV_64F); sharp = img - np.uint8(np.absolute(lap)); cv2.imwrite('results/aguçamento.png', sharp)",
        "Q08": f"import cv2, os; p_in = r'{PATH_CINZA}'; img = cv2.imread(os.path.join(p_in, os.listdir(p_in)[0]), 0); eq = cv2.equalizeHist(img); cv2.imwrite('results/equalizada.png', eq)",
        "Q09": f"import cv2, os; p_in = r'{PATH_BIN}'; img = cv2.imread(os.path.join(p_in, os.listdir(p_in)[0]), 0); _, t = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY); cv2.imwrite('results/limiar_127.png', t)",
        "Q10": f"import cv2, os, numpy as np; p_in = r'{PATH_CINZA}'; img = cv2.imread(os.path.join(p_in, os.listdir(p_in)[0]), 0); mult = np.digitize(img, bins=[85, 170]) * 127; cv2.imwrite('results/multilimiar.png', mult.astype(np.uint8))",
        "Q11": f"import cv2, os; p_in = r'{PATH_CINZA}'; img = cv2.imread(os.path.join(p_in, os.listdir(p_in)[0]), 0); _, t = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU); cv2.imwrite('results/otsu.png', t)",
        "Q12": f"import cv2, os; p_in = r'{PATH_CINZA}'; files = os.listdir(p_in); img1 = cv2.imread(os.path.join(p_in, files[0])); img2 = cv2.imread(os.path.join(p_in, files[1])); cv2.imwrite('results/soma.png', cv2.add(img1, img2))",
        "Q13": f"import cv2, os; p_in = r'{PATH_BIN}'; files = os.listdir(p_in); img1 = cv2.imread(os.path.join(p_in, files[0]), 0); img2 = cv2.imread(os.path.join(p_in, files[1]), 0); cv2.imwrite('results/and.png', cv2.bitwise_and(img1, img2))",
        "Q14": f"import cv2, os, numpy as np; p_in = r'{PATH_BIN}'; img = cv2.imread(os.path.join(p_in, os.listdir(p_in)[0]), 0); h, w = img.shape; M = cv2.getRotationMatrix2D((w/2, h/2), 45, 1); cv2.imwrite('results/rotacao.png', cv2.warpAffine(img, M, (w, h)))",
        "Q15": f"import cv2, os; p_in = r'{PATH_CINZA}'; img = cv2.imread(os.path.join(p_in, os.listdir(p_in)[0])); cv2.imwrite('results/negativo.png', 255 - img)",
        "Q16": f"import cv2, os, numpy as np; p_in = r'{PATH_CINZA}'; img = cv2.imread(os.path.join(p_in, os.listdir(p_in)[0]), 0); log_img = (np.log(1 + img) / np.log(1 + np.max(img)) * 255).astype(np.uint8); cv2.imwrite('results/log.png', log_img)",
        "Q17": f"import cv2, os, numpy as np; p_in = r'{PATH_CINZA}'; img = cv2.imread(os.path.join(p_in, os.listdir(p_in)[0]), 0); gamma = np.power(img/255.0, 0.5)*255; cv2.imwrite('results/gamma_05.png', gamma.astype(np.uint8))"
    }

    for q_id, code in scripts.items():
        q_dir = os.path.join(BASE_PATH, q_id)
        res_dir = os.path.join(q_dir, "results")
        os.makedirs(res_dir, exist_ok=True)
        
        py_file = os.path.join(q_dir, f"{q_id}.py")
        with open(py_file, "w") as f:
            f.write(code)
            
        print(f"Rodando {q_id}...")
        subprocess.run(["python", py_file], cwd=q_dir)

if __name__ == "__main__":
    create_structure()