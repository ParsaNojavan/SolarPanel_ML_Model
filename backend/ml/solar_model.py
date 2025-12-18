import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors


class SolarMLSystem:
    def __init__(self, sunny_threshold: float = 200):
        # ---------------- FEATURES & TARGET ----------------
        self.features = ["Hour", "Month", "TempOut", "OutHum", "WindSpeed", "Bar"]
        self.target = "SolarRad"
        self.sunny_threshold = sunny_threshold

        # ---------------- SCALER ----------------
        self.scaler = StandardScaler()

        # ---------------- REGRESSION ----------------
        self.regressor = LinearRegression()

        # ---------------- CLASSIFICATION ----------------
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )

        # ---------------- CLUSTERING ----------------
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.dbscan = DBSCAN(eps=0.8, min_samples=10)

        # ---------------- INTERNAL FLAGS ----------------
        self.is_trained = False
        self.dbscan_labels = None
        self.dbscan_training_data = None

    # ---------------- DATA LOADING ----------------
    def load_data(self, file_path: str) -> pd.DataFrame:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Only CSV or XLSX files are supported")

        # حذف ردیف‌های خالی در ستون‌های مهم
        df = df.dropna(subset=self.features + [self.target])

        # تبدیل ستون‌ها به عددی و پر کردن مقادیر گم‌شده
        df[self.features + [self.target]] = df[self.features + [self.target]].apply(pd.to_numeric, errors='coerce')
        df.fillna(df[self.features + [self.target]].mean(), inplace=True)

        return df

    # ---------------- FEATURE SCALING ----------------
    def scale_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        X = df[self.features].values
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    # ---------------- REGRESSION ----------------
    def train_regression(self, df: pd.DataFrame) -> float:
        X = self.scale_features(df, fit=True)
        y = df[self.target].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.regressor.fit(X_train, y_train)
        preds = self.regressor.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        return round(float(mse), 2)

    # ---------------- CLASSIFICATION ----------------
    def train_classifier(self, df: pd.DataFrame) -> float:
        df = df.copy()
        df["sunny"] = (df[self.target] >= self.sunny_threshold).astype(int)

        X = self.scale_features(df, fit=False)
        y = df["sunny"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.classifier.fit(X_train, y_train)
        preds = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        return round(float(accuracy), 3)

    # ---------------- CLUSTERING ----------------
    def train_clustering(self, df: pd.DataFrame) -> dict:
        X = self.scale_features(df, fit=False)
        labels = self.kmeans.fit_predict(X)
        counts = pd.Series(labels).value_counts()
        return {int(k): int(v) for k, v in counts.items()}

    # ---------------- ANOMALY DETECTION ----------------
    def train_anomaly_detector(self, df: pd.DataFrame) -> int:
        X = self.scale_features(df, fit=False)
        self.dbscan_labels = self.dbscan.fit_predict(X)
        self.dbscan_training_data = X  # ذخیره داده‌های مقیاس‌بندی شده
        return int((self.dbscan_labels == -1).sum())

    # ---------------- FULL TRAIN ----------------
    def train(self, file_path: str) -> dict:
        df = self.load_data(file_path)

        # Fit scaler on full dataset
        self.scale_features(df, fit=True)

        regression_mse = self.train_regression(df)
        classification_acc = self.train_classifier(df)
        clusters = self.train_clustering(df)
        anomalies = self.train_anomaly_detector(df)

        self.is_trained = True

        return {
            "regression_mse": float(regression_mse),
            "classification_accuracy": float(classification_acc),
            "clusters": clusters,
            "anomalies_detected": int(anomalies)
        }

    # ---------------- PREDICT ----------------
    def predict(self, input_data: dict) -> dict:
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        X = pd.DataFrame([input_data])[self.features].values
        X_scaled = self.scaler.transform(X)

        solar_rad = self.regressor.predict(X_scaled)[0]
        sunny_prob = self.classifier.predict_proba(X_scaled)[0][1]

        return {
            "solar_radiation": round(float(solar_rad), 2),
            "sunny_probability": round(float(sunny_prob), 3)
        }

    # ---------------- ANALYZE ----------------
    def analyze_point(self, input_data: dict) -> dict:
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        X = pd.DataFrame([input_data])[self.features].values
        X_scaled = self.scaler.transform(X)

        cluster = int(self.kmeans.predict(X_scaled)[0])

        # آنومالی
        if self.dbscan_labels is not None:
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(self.dbscan_training_data)
            distances, indices = nn.kneighbors(X_scaled)
            nearest_label = self.dbscan_labels[indices[0][0]]
            is_anomaly = int(nearest_label == -1)
        else:
            is_anomaly = 0

        return {
            "cluster": cluster,
            "is_anomaly": is_anomaly
        }

    # ---------------- FULL ANALYSIS ----------------
    def full_analysis(self, input_data: dict) -> dict:

        pred = self.predict(input_data)
        analysis = self.analyze_point(input_data)
        return {**pred, **analysis}
