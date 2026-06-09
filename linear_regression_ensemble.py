import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class LinearRegressionEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 base_estimator=LinearRegression(), 
                 meta_estimator=RandomForestClassifier(), 
                 n_models=10, 
                 features_per_model=2,
                 class_pairs=None,
                 random_state=None):
        self.base_estimator = base_estimator
        self.meta_estimator = meta_estimator
        self.n_models = n_models
        self.features_per_model = features_per_model
        self.class_pairs = class_pairs
        self.random_state = random_state

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.models_ = []
        self.feature_sets_ = []
        self.groupings_ = []

        # Encode categorical features if any
        self.encoders_ = {}
        X = X.copy()
        self.feature_columns_ = X.columns.tolist()
        for col in X.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.encoders_[col] = le

        classes = np.unique(y)

        # If class pairs not defined, create one-vs-rest pairs
        if self.class_pairs is None:
            self.class_pairs = [([cls], list(classes[classes != cls])) for cls in classes]

        for i in range(self.n_models):
            # Choose 2 random features
            selected_features = np.random.choice(X.columns, self.features_per_model, replace=False)
            self.feature_sets_.append(selected_features)

            # Choose a random class split
            group_a, group_b = self.class_pairs[i % len(self.class_pairs)]
            y_binary = np.where(np.isin(y, group_a), 1, 0)

            X_sample, y_sample = resample(X[selected_features], y_binary, random_state=i)

            model = clone(self.base_estimator)
            model.fit(X_sample, y_sample)
            self.models_.append(model)

        # Create meta-features
        X_meta = self._generate_meta_features(X)
        self.meta_estimator_ = clone(self.meta_estimator)
        self.meta_estimator_.fit(X_meta, y)
        return self

    def _transform_input(self, X):
        X = X.copy()
        # Ensure all necessary columns are present
        missing_cols = [col for col in self.feature_columns_ if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in input: {missing_cols}")

        # Apply encoders
        for col, le in self.encoders_.items():
            if col in X.columns:
                X[col] = X[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
        return X

    def _generate_meta_features(self, X):
        X = self._transform_input(X)
        meta_features = []
        for model, features in zip(self.models_, self.feature_sets_):
            preds = model.predict(X[features])
            meta_features.append(preds)
        return np.array(meta_features).T

    def predict(self, X):
        X_meta = self._generate_meta_features(X)
        return self.meta_estimator_.predict(X_meta)

    def predict_proba(self, X):
        X_meta = self._generate_meta_features(X)
        if hasattr(self.meta_estimator_, "predict_proba"):
            return self.meta_estimator_.predict_proba(X_meta)
        else:
            raise AttributeError("Meta estimator does not support predict_proba")

# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score

    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    clf = CustomEnsembleClassifier(n_models=9, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))