import cv2
import numpy as np
import os
import time

def rastreamento_caio_com_captura():
    # 1. Configuração de caminhos para salvar as fotos
    path_output = r'C:\cod_mestrado\pdi\Agora_vai\Lista02\Q24\results'
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    # 2. Intervalo para o Azul Escuro (Pacco Navy)
    azul_baixo = np.array([100, 150, 20])
    azul_alto = np.array([140, 255, 255])

    # 3. Inicialização da captura
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Erro: Não foi possível acessar a câmera.")
        return

    print("Rastreamento e Captura iniciados (Fotos a cada 3s). Pressione 'q' para sair.")

    # Variável para controlar o tempo das capturas
    ultima_captura = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, azul_baixo, azul_alto)
        
        # Limpeza morfológica reforçada
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Encontrar contornos
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(cnts) > 0:
            for c in cnts:
                area = cv2.contourArea(c)
                if area > 2000:
                    # Solução para Diagonais: Retângulo Rotacionado
                    rect = cv2.minAreaRect(c)
                    (centro, dimensoes, angulo) = rect
                    w_rot, h_rot = dimensoes
                    
                    if w_rot == 0 or h_rot == 0: continue
                    
                    # Razão de Aspecto e Solidez
                    maior_lado = max(w_rot, h_rot)
                    menor_lado = min(w_rot, h_rot)
                    razao_invariante = maior_lado / menor_lado
                    solidez_rot = area / (w_rot * h_rot)
                    
                    # Filtros Geométricos (Garrafa Pacco)
                    if 1.4 < razao_invariante < 6.0 and solidez_rot > 0.6:
                        M = cv2.moments(c)
                        if M["m00"] > 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])

                            # Desenho do Retângulo Rotacionado
                            box = cv2.boxPoints(rect)
                            box = np.array(box, dtype=np.int32) 
                            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
                            
                            # Desenho do Centroide e Texto
                            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                            cv2.putText(frame, "Garrafa termica azul: Caio Cavalcanti", (cx - 50, cy - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.putText(frame, f"Centro: ({cx}, {cy})", (cx - 50, cy + 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # --- LÓGICA DE SALVAMENTO AUTOMÁTICO (3 SEGUNDOS) ---
        tempo_atual = time.time()
        if tempo_atual - ultima_captura >= 500:
            timestamp = int(tempo_atual)
            nome_foto = f"captura_q24_{timestamp}.png"
            caminho_foto = os.path.join(path_output, nome_foto)
            cv2.imwrite(caminho_foto, frame)
            print(f"Foto salva: {nome_foto}")
            ultima_captura = tempo_atual

        # Exibição da captura
        cv2.imshow("Rastreamento e Captura - Lista 02", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    rastreamento_caio_com_captura()
