import numpy as np         # Thư viện tính toán số học
import pandas as pd        # Thư viện xử lý dữ liệu dạng bảng
import matplotlib.pyplot as plt  # Thư viện vẽ biểu đồ

# Tạo dữ liệu
data = {
    'Mã KV': ['KV0', 'KV1', 'KV2', 'KV3', 'KV4', 'KV5', 'KV6', 'KV7'],
    'Lưu lượng giao thông': [8000, 3000, 12000, 2000, 5000, 6000, 15000, 4000],
    'Diện tích (km2)': [5, 3, 7, 2, 5.5, 6, 8, 3]
}

df = pd.DataFrame(data)

# Chuẩn bị dữ liệu cho K-means
X = df[['Lưu lượng giao thông', 'Diện tích (km2)']].values


# PHẦN A: K-means với K=3


class KMeansCustom:
    def __init__(self, n_clusters=3, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        
    def _initialize_centroids(self, X):
        np.random.seed(self.random_state)
        # Sửa lỗi: khi n_clusters > số điểm dữ liệu, cho phép replace=True
        if self.n_clusters > len(X):
            random_indices = np.random.choice(len(X), self.n_clusters, replace=True)
        else:
            random_indices = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[random_indices]
    
    def _assign_clusters(self, X, centroids):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, centroid in enumerate(centroids):
            distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                centroids[i] = np.mean(cluster_points, axis=0)
            else:
                # Nếu cụm rỗng, gán lại centroid ngẫu nhiên
                centroids[i] = X[np.random.randint(0, len(X))]
        return centroids
    
    def fit(self, X):
        self.centroids = self._initialize_centroids(X)
        
        for _ in range(self.max_iter):
            old_centroids = self.centroids.copy()
            self.labels_ = self._assign_clusters(X, self.centroids)
            self.centroids = self._update_centroids(X, self.labels_)
            
            # Kiểm tra hội tụ
            if np.allclose(old_centroids, self.centroids):
                break
        
        # Tính inertia (WCSS) sau khi hội tụ
        self.inertia_ = 0
        for i in range(self.n_clusters):
            cluster_points = X[self.labels_ == i]
            if len(cluster_points) > 0:
                self.inertia_ += np.sum((cluster_points - self.centroids[i]) ** 2)
        
        return self

# Áp dụng K-means với K=3
print("=" * 50)
print("PHẦN A: K-MEANS VỚI K=3")
print("=" * 50)

kmeans = KMeansCustom(n_clusters=3, random_state=42)
kmeans.fit(X)

# Thêm nhãn cụm vào DataFrame
df['Cụm (K=3)'] = kmeans.labels_

# Hiển thị kết quả
print("\nKết quả phân cụm với K=3:")
print(df[['Mã KV', 'Lưu lượng giao thông', 'Diện tích (km2)', 'Cụm (K=3)']].to_string(index=False))

# Hiển thị tọa độ các centroids
print("\nTọa độ các centroids:")
for i, centroid in enumerate(kmeans.centroids):
    print(f"Cụm {i}: Lưu lượng = {centroid[0]:.2f}, Diện tích = {centroid[1]:.2f}")

# Vẽ biểu đồ phân cụm
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
for i in range(3):
    cluster_points = X[kmeans.labels_ == i]
    if len(cluster_points) > 0:
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    color=colors[i], label=f'Cụm {i}', s=100)
    
# Vẽ centroids
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
           color='black', marker='X', s=200, label='Centroids')

plt.xlabel('Lưu lượng giao thông')
plt.ylabel('Diện tích (km2)')
plt.title('K-means Phân Cụm với K=3')
plt.legend()
plt.grid(True, alpha=0.3)


# PHẦN B: Phương pháp Elbow

print("\n" + "=" * 50)
print("PHẦN B: PHƯƠNG PHÁP ELBOW")
print("=" * 50)

# Tính WCSS cho các giá trị K từ 1 đến 9
wcss = []
k_values = range(1, 10)

for k in k_values:
    if k <= len(X):  # Chỉ tính khi K <= số điểm dữ liệu
        kmeans_temp = KMeansCustom(n_clusters=k, random_state=42, max_iter=100)
        kmeans_temp.fit(X)
        wcss.append(kmeans_temp.inertia_)
    else:
        # Khi K > số điểm dữ liệu, WCSS sẽ bằng 0 (mỗi điểm là 1 cụm)
        wcss.append(0)

# Hiển thị giá trị WCSS
print("\nGiá trị WCSS cho các K:")
for k, w in zip(k_values, wcss):
    print(f"K = {k}: WCSS = {w:.2f}")

# Vẽ biểu đồ Elbow
plt.subplot(1, 2, 2)
plt.plot(k_values, wcss, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Số cụm (K)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.title('Phương Pháp Elbow')
plt.xticks(k_values)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Xác định K tối ưu bằng phương pháp góc cong 
print("\nĐộ giảm tỷ lệ WCSS (chỉ xét K từ 1 đến 7):")
optimal_k = 3  # Giá trị mặc định
max_elbow = 0

# Chỉ xét đến K=7 vì với K=8 trở lên sẽ có vấn đề
for i in range(1, min(7, len(wcss) - 1)):
    if wcss[i+1] > 0:  # Tránh chia cho 0
        # Tính góc tại điểm i (độ dốc thay đổi)
        prev_slope = wcss[i-1] - wcss[i]
        next_slope = wcss[i] - wcss[i+1]
        
        if next_slope > 0:  # Tránh trường hợp slope âm
            elbow_score = prev_slope / next_slope
            print(f"K = {i+1}: Độ giảm trước = {prev_slope:.2f}, Độ giảm sau = {next_slope:.2f}, Tỷ lệ = {elbow_score:.2f}")
            
            if elbow_score > max_elbow:
                max_elbow = elbow_score
                optimal_k = i + 1

print(f"\nK tối ưu theo phương pháp Elbow: K = {optimal_k}")

# Tính điểm gấp khủy (elbow point) bằng phương pháp khác
print("\nPhân tích bằng phương pháp tỷ lệ giảm:")
wcss_array = np.array(wcss)
reductions = []
for i in range(len(wcss_array) - 1):
    if wcss_array[i] > 0:
        reduction = (wcss_array[i] - wcss_array[i+1]) / wcss_array[i] * 100
        reductions.append(reduction)
        print(f"Từ K={i+1} sang K={i+2}: giảm {reduction:.1f}%")

# Tìm điểm mà tỷ lệ giảm giảm mạnh nhất
if len(reductions) > 1:
    reduction_drops = []
    for i in range(len(reductions) - 1):
        drop = reductions[i] - reductions[i+1]
        reduction_drops.append(drop)
    
    # Tìm K tại điểm có sự thay đổi lớn nhất
    if len(reduction_drops) > 0:
        optimal_k2 = np.argmax(reduction_drops) + 2  # +2 vì bắt đầu từ K=2
        print(f"\nK tối ưu theo phân tích tỷ lệ giảm: K = {optimal_k2}")

# Vẽ lại phân cụm với K tối ưu
print(f"\nPhân cụm với K tối ưu = {optimal_k}:")
kmeans_optimal = KMeansCustom(n_clusters=optimal_k, random_state=42)
kmeans_optimal.fit(X)

# Tạo DataFrame mới để hiển thị
df_result = df.copy()
df_result[f'Cụm (K={optimal_k})'] = kmeans_optimal.labels_
print(df_result[['Mã KV', 'Lưu lượng giao thông', 'Diện tích (km2)', f'Cụm (K={optimal_k})']].to_string(index=False))

# Vẽ biểu đồ với K tối ưu
plt.figure(figsize=(8, 6))
for i in range(optimal_k):
    cluster_points = X[kmeans_optimal.labels_ == i]
    if len(cluster_points) > 0:
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    color=colors[i], label=f'Cụm {i}', s=100)

# Vẽ centroids
plt.scatter(kmeans_optimal.centroids[:, 0], kmeans_optimal.centroids[:, 1], 
           color='black', marker='X', s=200, label='Centroids')

plt.xlabel('Lưu lượng giao thông')
plt.ylabel('Diện tích (km2)')
plt.title(f'K-means Phân Cụm với K tối ưu = {optimal_k}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n" + "=" * 50)
print("GIẢI THÍCH KẾT QUẢ:")
print("=" * 50)
print("1. Với K=3, dữ liệu được phân thành 3 cụm dựa trên lưu lượng và diện tích")
print("2. Biểu đồ Elbow giúp xác định K tối ưu:")
print("   - Khi K tăng, WCSS giảm")
print("   - Điểm 'khuỷu tay' (elbow point) là nơi WCSS giảm chậm lại đáng kể")
print("   - Thông thường, điểm này được chọn làm K tối ưu")
print("3. Với 8 điểm dữ liệu, K tối ưu thường nằm trong khoảng 2-4")