# train_model.py  -- run once. It trains, tests, and saves everything the web page needs.
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# 1. Load data and split (stratify keeps 100-per-crop balance in train/test)
df = pd.read_csv("Crop_recommendation.csv")
X, y = df[FEATURES].values, df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Scale features (needed for the unusual-input detector)
scaler = StandardScaler().fit(X_train)
Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

# 3. Classifier
model = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr, y_train)
pred = model.predict(Xte)
print("Test accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred))
print("Confusion matrix shape:", confusion_matrix(y_test, pred).shape)

# 4. Confidence = highest predicted probability
conf = model.predict_proba(Xte).max(axis=1)
print("Confidence on test data: min %.2f, mean %.2f" % (conf.min(), conf.mean()))

# 5. Unusual-input detector
#    (a) each feature must stay inside training min..max (+10% margin)
lo, hi = X_train.min(axis=0), X_train.max(axis=0)
margin = 0.10 * (hi - lo)
lo, hi = lo - margin, hi + margin
#    (b) the COMBINATION must look like training data:
#        average distance to the 5 closest training rows must not be too big
nn = NearestNeighbors(n_neighbors=6).fit(Xtr)
train_d = nn.kneighbors(Xtr)[0][:, 1:].mean(axis=1)    # [:,1:] skips the row itself
threshold = np.percentile(train_d, 97)                  # 97% of training rows are "normal"

def is_unusual(raw):                      # raw: 2D array of original-unit values
    raw = np.asarray(raw, dtype=float)
    out_range = ((raw < lo) | (raw > hi)).any(axis=1)
    odd_combo = nn.kneighbors(scaler.transform(raw), n_neighbors=5)[0].mean(axis=1) > threshold
    return out_range | odd_combo

# 6. Check the detector
print("\nNormal test rows flagged: %.1f%%" % (100 * is_unusual(X_test).mean()))
rng = np.random.default_rng(42)
shuffled = np.column_stack([rng.permutation(X_train[:, i]) for i in range(7)])
print("Random-mixed rows flagged (each value normal, combo odd): %.1f%%" % (100 * is_unusual(shuffled).mean()))
extreme = X_test.copy(); extreme[:, 5] = 14; extreme[:, 6] = 5000
print("Extreme rows (pH 14, rain 5000) flagged: %.1f%%" % (100 * is_unusual(extreme).mean()))

# 7. Robustness: add random noise, see how often the crop stays the same
print("\nRobustness (same recommendation after noise):")
base = model.predict(Xte)
for level in [0.05, 0.10, 0.20]:
    noisy = X_test * (1 + rng.normal(0, level, X_test.shape))
    same = (model.predict(scaler.transform(noisy)) == base).mean()
    print(f"  noise {int(level*100)}%: {same*100:.1f}% unchanged")

# 8. Save everything for the web page
joblib.dump({"model": model, "scaler": scaler, "nn": nn, "threshold": threshold, "lo": lo, "hi": hi,
             "features": FEATURES}, "crop_bundle.joblib")
print("\nSaved crop_bundle.joblib")
