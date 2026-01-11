import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Dữ liệu đầu vào
experience = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]).reshape(-1, 1)
salary = np.array([0.0, 0.0, 0.0, 0.0, 60.0, 64.0, 55.0, 61.0, 66.0, 83.0, 93.0, 91.0, 98.0, 101.0])

# Câu 1: Tự tạo hàm KNN predictor
def knn_predictor(experience_value, k=3):
    """
    Hàm dự đoán lương dựa trên kinh nghiệm sử dụng thuật toán KNN
    experience_value: giá trị kinh nghiệm cần dự đoán
    k: số lượng neighbors (mặc định là 3)
    """
    # Tính khoảng cách Euclidean giữa điểm cần dự đoán và tất cả các điểm trong dataset
    distances = np.sqrt(np.sum((experience - experience_value) ** 2, axis=1))
    
    # Lấy chỉ số của k điểm gần nhất
    nearest_indices = np.argsort(distances)[:k]
    
    # Tính giá trị dự đoán là trung bình của k neighbors
    predicted_salary = np.mean(salary[nearest_indices])
    
    return predicted_salary

# Dự đoán với experience = 6.3 và k=3
predicted_salary_manual = knn_predictor(6.3, k=3)
print("Câu 1: Dự đoán bằng hàm tự tạo")
print(f"Kinh nghiệm: 6.3 năm => Lương dự đoán: {predicted_salary_manual:.2f}")

# Câu 2: So sánh với Scikit-learn
print("\n" + "="*50 + "\n")
print("Câu 2: So sánh với Scikit-learn")

# Tạo và huấn luyện mô hình KNN với k=3
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(experience, salary)

# Dự đoán với Scikit-learn
predicted_salary_sklearn = knn_model.predict([[6.3]])[0]
print(f"Kinh nghiệm: 6.3 năm => Lương dự đoán (Scikit-learn): {predicted_salary_sklearn:.2f}")

# So sánh kết quả
print("\nSo sánh kết quả:")
print(f"- Hàm tự tạo: {predicted_salary_manual:.2f}")
print(f"- Scikit-learn: {predicted_salary_sklearn:.2f}")
print(f"- Chênh lệch: {abs(predicted_salary_manual - predicted_salary_sklearn):.2f}")

# Kiểm tra thêm với các giá trị k khác để xác minh
print("\n" + "="*50)
print("Kiểm tra với các giá trị k khác:")

for k in [1, 3, 5, 7]:
    # Dự đoán bằng hàm tự tạo
    manual_pred = knn_predictor(6.3, k=k)
    
    # Dự đoán bằng Scikit-learn
    sklearn_model = KNeighborsRegressor(n_neighbors=k)
    sklearn_model.fit(experience, salary)
    sklearn_pred = sklearn_model.predict([[6.3]])[0]
    
    print(f"\nk = {k}:")
    print(f"  Hàm tự tạo: {manual_pred:.2f}")
    print(f"  Scikit-learn: {sklearn_pred:.2f}")
    print(f"  Khớp nhau: {abs(manual_pred - sklearn_pred) < 0.001}")

# Phân tích các neighbors khi k=3
print("\n" + "="*50)
print("Phân tích chi tiết khi k=3:")

# Tính khoảng cách cho điểm 6.3
distances = np.sqrt(np.sum((experience - 6.3) ** 2, axis=1))
nearest_indices = np.argsort(distances)[:3]

print(f"\n3 điểm gần nhất với experience=6.3:")
for i, idx in enumerate(nearest_indices):
    print(f"  {i+1}. Experience: {experience[idx][0]}, Salary: {salary[idx]}, Distance: {distances[idx]:.4f}")

print(f"\nGiá trị dự đoán = ({salary[nearest_indices[0]]} + {salary[nearest_indices[1]]} + {salary[nearest_indices[2]]}) / 3 = {predicted_salary_manual:.2f}")