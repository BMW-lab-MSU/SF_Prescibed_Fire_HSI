from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def evaluate_classifier(X, y, model_type='svm', test_size=0.2, random_state=42, debug_mode=False):
    """
    Train and evaluate classifier on selected bands.
    Args:
        X: (num_samples, num_bands) data matrix
        y: labels (num_samples,)
        model_type: 'svm' or 'rf'
    Returns:
        accuracy, f1
        :param random_state:
        :param y:
        :param X:
        :param model_type:
        :param test_size:
        :param use_subset:
    """

    if debug_mode:
        print("🛠️ Debug mode active: Using stratified subset")
        from src.models.balanced_sampler import stratified_subset  # or inline if already defined
        X, y = stratified_subset(X, y, n_per_class=1000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if model_type == 'svm':
        print("Starting training svm....")
        model = SVC(kernel='rbf', gamma='scale')
    elif model_type == 'rf':
        print("Starting training Random Forest....")
        model = RandomForestClassifier(n_estimators=100)
    else:
        raise ValueError("Model type must be 'svm' or 'rf'.")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    return acc, f1
