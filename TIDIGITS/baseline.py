import random, os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from utils import load_encoded_TIDIGITS



def main(seed=0, trim=False):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    # Load dataset without the spike encoding step
    X_train, X_test, y_train, y_test = load_encoded_TIDIGITS(seed=seed, trim=trim, to_spike=False)
    
    # Resize to 2D vectors
    output_size = 55*40 if trim else 60*40
    X_train = X_train.reshape(-1, output_size)
    X_test = X_test.reshape(-1, output_size)

    # Readout
    clf = LinearSVC(random_state=seed, max_iter=10000, C=0.005)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred) 
    print(f"Accuracy : {acc}")
    return acc



if __name__ == "__main__":
    N = 10
    init_seed = 0
    recorded_acc = np.zeros(N)
    for i,seed in enumerate(range(init_seed,init_seed+N)):
        recorded_acc[i] = main(seed=seed)
    print("\n\n")
    print(f"Mean : {recorded_acc.mean()} +- {recorded_acc.std()}")
    print(recorded_acc)