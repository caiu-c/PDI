import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from skimage.feature import local_binary_pattern

# --- 1. CONFIGURAÇÕES E CAMINHOS ---
PATH_GRAY = r'C:\cod_mestrado\pdi\BancoImagens_TomCinza'
PATH_OUTPUT = r'C:\cod_mestrado\pdi\Agora_vai\Lista03\Q40\results'
if not os.path.exists(PATH_OUTPUT): os.makedirs(PATH_OUTPUT)

def extract_features(path_base):
    features, labels = [], []
    classes = [d for d in sorted(os.listdir(path_base)) if os.path.isdir(os.path.join(path_base, d))]
    
    print(f"Extraindo características das classes: {classes}")
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(path_base, class_name)
        for img_file in os.listdir(class_path):
            img = cv2.imread(os.path.join(class_path, img_file), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            # LBP Uniforme (Invariante à rotação)
            radius, n_points = 3, 24
            lbp = local_binary_pattern(img, n_points, radius, method='uniform')
            
            # Histograma Normalizado
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            
            features.append(hist)
            labels.append(idx)
            
    return np.array(features), np.array(labels), classes

# --- 2. KNN MANUAL VETORIZADO ---
class ManualKNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        # Broadcasting do NumPy para calcular todas as distâncias de uma vez
        # Dist(A, B) = sqrt(sum((A-B)^2))
        preds = []
        for x in X_test:
            # Subtrai x de todos os elementos de X_train simultaneamente
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            preds.append(Counter(k_nearest_labels).most_common(1)[0][0])
        return np.array(preds)

# --- 3. MÉTRICAS E VISUALIZAÇÃO ---
def plot_confusion_matrix(cm, classes, k_val, protocol_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Matriz de Confusão - KNN (K={k_val}) - {protocol_name}')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_OUTPUT, f'cm_k{k_val}_{protocol_name}.png'))
    plt.close()

def calculate_metrics(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred): cm[t, p] += 1
    
    accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
    recalls, specificities = [], []
    
    for i in range(num_classes):
        vp = cm[i, i]
        fn = np.sum(cm[i, :]) - vp
        fp = np.sum(cm[:, i]) - vp
        vn = np.sum(cm) - (vp + fp + fn)
        
        recalls.append(vp / (vp + fn) if (vp + fn) > 0 else 0)
        specificities.append(vn / (vn + fp) if (vn + fp) > 0 else 0)
        
    return {'acc': accuracy, 'rec': np.mean(recalls), 'spec': np.mean(specificities), 'cm': cm}

# --- 4. EXPERIMENTAÇÃO ---
def run_experiment(X, y, classes):
    log_file = open(os.path.join(PATH_OUTPUT, "metricas_finais.txt"), "w")
    k_values = [1, 3, 7]
    
    for k in k_values:
        header = f"\n{'='*20} K = {k} {'='*20}\n"
        print(header); log_file.write(header)
        knn = ManualKNN(k=k)

        # LOO
        loo_preds = []
        for i in range(len(X)):
            X_t = np.delete(X, i, axis=0); y_t = np.delete(y, i)
            knn.fit(X_t, y_t)
            loo_preds.append(knn.predict([X[i]])[0])
        
        m = calculate_metrics(y, loo_preds, len(classes))
        plot_confusion_matrix(m['cm'], classes, k, "LOO")
        
        res_loo = f"[LOO] Acc: {m['acc']:.4f} | Recall: {m['rec']:.4f} | Spec: {m['spec']:.4f}\n"
        print(res_loo); log_file.write(res_loo)

        # Hold-Out (10x)
        h_acc, h_rec, h_spec = [], [], []
        for _ in range(10):
            idx = np.random.permutation(len(X))
            split = int(0.7 * len(X))
            knn.fit(X[idx[:split]], y[idx[:split]])
            p = knn.predict(X[idx[split:]])
            met = calculate_metrics(y[idx[split:]], p, len(classes))
            h_acc.append(met['acc']); h_rec.append(met['rec']); h_spec.append(met['spec'])
            
        res_ho = (f"[Hold-Out] Acc: {np.mean(h_acc):.4f} (±{np.std(h_acc):.4f}) | "
                  f"Rec: {np.mean(h_rec):.4f} | Spec: {np.mean(h_spec):.4f}\n")
        print(res_ho); log_file.write(res_ho)

    log_file.close()

if __name__ == "__main__":
    X, y, classes = extract_features(PATH_GRAY)
    if len(X) > 0: run_experiment(X, y, classes)