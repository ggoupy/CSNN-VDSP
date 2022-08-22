import random, os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from utils import load_MNIST



def main(seed=0):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    X_train, X_test, y_train, y_test = load_MNIST(seed=seed)
    X_train = X_train.reshape(-1, 28*28)
    X_test = X_test.reshape(-1, 28*28)
    
    clf = LinearSVC(random_state=seed, max_iter=10000, C=0.005)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred) 
    print(f"Accuracy : {acc}")
    return acc



if __name__ == "__main__":
    N = 5
    init_seed = 0
    recorded_acc = np.zeros(N)
    for i,seed in enumerate(range(init_seed,init_seed+N)):
        recorded_acc[i] = main(seed=seed)
    print("\n\n")
    print(f"Mean : {recorded_acc.mean()} +- {recorded_acc.std()}")
    print(recorded_acc)