---
title: "Correlation and Regression"
meta_title: ""
description: "Knowledge about Statistics"
date: 2022-04-04T05:00:00Z
image: "/images/image-placeholder.png"
categories: ["Statistics"]
author: "Minh Van"
tags: ["Statistics", "Python"]
draft: false
---

# Tổng hợp kiến thức: Regression, Polynomial, và Linear Algebra

## 1. Chuẩn hóa dữ liệu và ảnh hưởng đến học
Trong regression, đặc biệt khi dùng gradient descent hoặc polynomial features, việc scale dữ liệu là bắt buộc để đảm bảo hội tụ ổn định.

### Hai cách phổ biến
**Standardization (mean/std):**  
$$ 
x' = \frac{x - \mu}{\sigma}  
$$
- Trung tâm về 0
- Variance ≈ 1
- Gradient ổn định

**Min-max scaling:**  
$$ 
x' = \frac{x - x_{min}}{x_{max} - x_{min}}  
$$
- Dữ liệu nằm trong [0,1]
- Không zero-centered
- Nhạy với outlier

### Kết luận
Standardization thường tốt hơn trong gradient descent vì:
- Giảm lệch gradient giữa các feature
- Giúp learning rate hoạt động ổn định
---

## 2. Linear Regression dưới dạng ma trận

### Mô hình

$$  
y = Xw + b  
$$

Thường viết lại bằng cách thêm bias vào X:

$$ 
X = [1, x_1, x_2, ..., x_F]  
$$

$$ 
y = Xw  
$$

---
## 3. Closed-form solution (Normal Equation)

### Công thức

$$ 
w = (X^T X)^{-1} X^T y  
$$

### Trong code
```python
w = np.linalg.solve(X.T @ X, X.T @ y)
```

### Lưu ý
- Không dùng `np.linalg.inv` vì kém ổn định
- `solve` nhanh và chính xác hơn

### 4. Dạng thống kê:
Bạn có thể dùng công thức tay:
$$
w = \frac{\text{Cov}(x,y)}{\text{Var}(x)} 
$$$$
b = \bar{y} - w \bar{x}
$$
Code:

```python
x_mean = x_train.mean()
y_mean = y_train.mean()

w = np.sum((x_train - x_mean)*(y_train - y_mean)) / np.sum((x_train - x_mean)**2)
b = y_mean - w * x_mean
```
---
## 4. Ridge Regression (L2 Regularization)

### Hàm loss

$$  
L = |Xw - y|^2 + \lambda |w|^2  
$$

### Nghiệm
$$
w = (X^T X + \lambda I)^{-1} X^T y  
$$

### Không regularize bias

```python
I = np.eye(D)
I[0,0] = 0
```

### Ý nghĩa
- Giảm overfitting
- Ổn định khi ma trận gần singular

---
## 5. Vấn đề số học: Singular và Ill-conditioned

### Khi nào xảy ra
- Feature tương quan mạnh (đa cộng tuyến)
- Polynomial feature
- N nhỏ, D lớn

### Hệ quả

- Không invert được ( X^T X )
- Weight rất lớn
- Model không ổn định

---

## 6. Pseudo-inverse và SVD

### Công thức

$$ 
w = X^+ y  
$$

```python
w = np.linalg.pinv(X) @ y
```

### Dựa trên SVD
$$
X = U \Sigma V^T  
$$

$$
X^+ = V \Sigma^+ U^T  
$$

### Ý nghĩa
- Loại bỏ chiều yếu (singular values nhỏ)
- Ổn định hơn Ridge trong nhiều trường hợp

---
## 7. Polynomial Regression

### Ý tưởng
Thay vì:

$$ 
y = w_1 x + w_2  
$$

ta dùng:

$$
y = w^T \phi(x)  
$$

Trong đó ( $\phi(x)$ ) là polynomial features.

---
### Ví dụ với 2 biến, bậc 3

$$  
[1, x_1, x_2, x_1^2, x_1x_2, x_2^2, x_1^3, x_1^2x_2, x_1x_2^2, x_2^3]  
$$

---
### Sinh tự động bằng tổ hợp

```python
from itertools import combinations_with_replacement

comb([1.0] + x, 3)
```

→ sinh tất cả monomial bậc ≤ 3

---
### Số lượng feature

$$ 
D = \binom{F + d}{d}  
$$

Ví dụ:
- F=2, d=3 → 10
- F=5, d=3 → 56

---
## 8. Gradient Descent vs Closed-form

| Phương pháp      | Ưu điểm     | Nhược điểm                |
| ---------------- | ----------- | ------------------------- |
| Gradient Descent | mở rộng tốt | cần tuning LR             |
| Closed-form      | chính xác   | không scale tốt với D lớn |
| Ridge            | ổn định hơn | cần chọn λ                |
| SVD/pinv         | robust nhất | chậm hơn                  |

---
## 9. Hồi quy thống kê cổ điển
### Hai đường regression

1. **y on x**  
    $$  
    y = b + ax  
    $$
2. **x on y**  
    $$
    x = b' + a'y  
    $$
### Khác nhau
- y on x: minimize vertical error
- x on y: minimize horizontal error

### Quan hệ với correlation

$$  
a_{yx} = r \cdot \frac{\sigma_y}{\sigma_x}  
$$

$$
a_{xy} = r \cdot \frac{\sigma_x}{\sigma_y}  
$$

$$
a_{yx} \cdot a_{xy} = r^2  
$$

### Tính chất
- Hai đường cắt nhau tại ( $(\bar{x}, \bar{y})$ )
- Nếu ( r = 1 ): trùng nhau
- Nếu ( r = 0 ): vuông góc

