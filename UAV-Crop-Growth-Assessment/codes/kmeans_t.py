import cv2
import numpy as np
import time

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

# ==============================
# START TIMER
# ==============================
start_time = time.time()

# ==============================
# READ IMAGE
# ==============================
image = cv2.imread(r"C:\Users\gobik\OneDrive\Documents\UAV-Crop-Growth-Assessment\dataset\field_image.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ==============================
# RGB -> HSV
# ==============================
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

# Reshape pixels
pixels = hsv.reshape((-1, 3))

# ==============================
# K-MEANS
# ==============================
kmeans = KMeans(
    n_clusters=5,
    random_state=42
)

kmeans.fit(pixels)

labels = kmeans.labels_
colors = kmeans.cluster_centers_.astype(int)

# ==============================
# QUALITY METRICS
# ==============================
sil_score = silhouette_score(
    pixels,
    labels,
    sample_size=5000,
    random_state=42
)

db_score = davies_bouldin_score(
    pixels,
    labels
)

# ==============================
# CLUSTER ANALYSIS
# ==============================
unique, counts = np.unique(
    labels,
    return_counts=True
)

total_pixels = len(labels)

healthy = 0
mature = 0
nonveg = 0

print("\nHSV CLUSTERS\n")

for cluster_id, count in zip(unique, counts):

    H = int(colors[cluster_id][0])
    S = int(colors[cluster_id][1])
    V = int(colors[cluster_id][2])

    percentage = (count / total_pixels) * 100

    print(f"\nCluster {cluster_id}")
    print(f"HSV = [{H}, {S}, {V}]")
    print(f"Pixels = {count}")
    print(f"Percentage = {percentage:.2f}%")

    if 35 <= H <= 85:
        stage = "Healthy Green Vegetation 🌱"
        healthy += percentage

    elif 15 <= H < 35:
        stage = "Mature / Yellow-Green Vegetation 🌾"
        mature += percentage

    else:
        stage = "Dry / Non-Vegetation 🟤"
        nonveg += percentage

    print("Classification:", stage)

# ==============================
# END TIMER
# ==============================
runtime = time.time() - start_time

# ==============================
# FINAL RESULTS
# ==============================
print("\n================================")
print("K-MEANS GROWTH SUMMARY")
print("================================")

print(f"Healthy Growth: {healthy:.2f}%")
print(f"Mature Growth: {mature:.2f}%")
print(f"Non-Vegetation: {nonveg:.2f}%")

print("\n================================")
print("PERFORMANCE METRICS")
print("================================")

print(f"Runtime: {runtime:.4f} seconds")
print(f"Number of Clusters: {len(colors)}")
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Davies-Bouldin Index: {db_score:.4f}")