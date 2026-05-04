---
title: "Classification Model"
meta_title: ""
description: "Classify by SVM model"
date: 2022-04-04T05:00:00Z
image: "/images/image-placeholder.png"
categories: ["Machine learning"]
author: "Minh Van"
tags: ["AI", "Machine-learning"]
draft: false
---

# Introduction

## 1. Mục tiêu
- Bài toán phân loại nhị phân (binary classification)
>**Mục tiêu cốt lõi** của SVM là tìm kiếm một **siêu mặt phẳng tuyến tính tối ưu (optimal linear hyperplane)** đóng vai trò là đường biên quyết định để phân tách tất cả dữ liệu

![[../../attachments/Pasted image 20251022223912.png]]
## 2. Mở rộng với các bài toán khác

- **Dữ liệu không phân tách tuyến tính (Linearly Non-separable Cases):** 
	- Khi dữ liệu không thể phân tách tuyến tính, SVM sử dụng **Soft Margin SVM** bằng cách giới thiệu **biến lỏng (slack variables)**. 
	- Phương pháp này cho phép thư giãn các ràng buộc phân tách và tìm kiếm sự đánh đổi (trade-off) giữa việc tối đa hóa lề (margin) và giảm thiểu lỗi phân loại sai
	
- **Phân loại đa lớp (Multi-class classification):** 
	- Mặc dù ban đầu là nhị phân, SVM có thể được mở rộng để phân loại k lớp bằng các chiến lược như:
		- **One-against-the rest** (huấn luyện k mô hình SVM nhị phân) 
		- **One-against-one** (huấn luyện k(k−1)/2 mô hình SVM nhị phân)
	- Trong đó chiến lược One-against-one thường nhanh hơn trong quá trình huấn luyện

# Perceptron

## 1. Theory

### Margin of a Hyperplane

Giả sử tập dữ liệu $D$ được phân tách tuyến tính bởi một siêu phẳng $$ ( H: \mathbf{w}^T \mathbf{x} + b = 0 ) $$
Khoảng cách nhỏ nhất từ điểm đến siêu phẳng gọi là **margin**:

$$
\delta = \min_{i=1}^{n} \frac{|\mathbf{w}^T \mathbf{x}_i + b|}{\|\mathbf{w}\|}
$$

**Các trường hợp đặc biệt:**
- Nếu $\|\mathbf{w}\| = 1 :$
  $$
  \delta = \min_i |\mathbf{w}^T \mathbf{x}_i + b| = \min_i y_i (\mathbf{w}^T \mathbf{x}_i + b)
  $$
- Nếu $\min_i y_i(\mathbf{w}^T \mathbf{x}_i + b) = 1 :$
  $$
  \delta = \frac{1}{\|\mathbf{w}\|}
  $$

👉 Mục tiêu trong SVM là **tối đa hóa margin** → tương đương **tối thiểu hóa** $\|\mathbf{w}\|^2$

#### Learning Algorithm

Tìm $\mathbf{w}, b$ sao cho:

$$
s_i = y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 0, \quad \forall i
$$

**Thuật toán:**

1. Khởi tạo:
$$
   \mathbf{w}_{(0)} = 0, \quad b_{(0)} = 0, \quad t = 0
$$
2. Lặp qua từng mẫu $(\mathbf{x}_i, y_i)$:
   - Tính:
     $$
     s_i = y_i(\mathbf{w}_{(t)}^T \mathbf{x}_i + b_{(t)})
     $$
   - Nếu $s_i \ge 0$: bỏ qua (đã phân loại đúng).
   - Nếu $s_i < 0$: cập nhật theo hướng tăng $s_i$:
     $$
     \mathbf{w}_{(t+1)} = \mathbf{w}_{(t)} + y_i \mathbf{x}_i
     $$
     $$
     b_{(t+1)} = b_{(t)} + y_i
     $$
     $$
     t \leftarrow t + 1
     $$
1. Dừng khi tất cả điểm được phân loại đúng:
   $$
   s_i \ge 0, \forall i
   $$

### Convergence Theorem

**Định lý:**
Nếu tồn tại $\mathbf{w}_\star, b_\star$ sao cho $y_i(\mathbf{w}_\star^T \mathbf{x}_i + b_\star) \ge \delta > 0$,  
thì số lần cập nhật tối đa của thuật toán Perceptron là:

$$
t \le \frac{R^2(\|\mathbf{w}_\star\|^2 + b_\star^2)}{\delta^2}
$$

trong đó:
$$
R^2 = \max_i \|\mathbf{x}_i\|^2 + 1
$$
là **bán kính của tập dữ liệu**.

# Margin

![[../../attachments/Pasted image 20251023091029.png]]
> Có rất nhiều giải pháp cho SVM như hình trên
> Vậy để biết đường Hyperlane nào là tối ưu nhất, ta sẽ dùng **Margin**

## Maximize margin

![[../../attachments/Pasted image 20251023091153.png]]
> **Margin** là 2 đường song song với Hyperlane mà mở rộng ra đến khi nào sát với điểm data-point gần nhất (đường lề)

## Support Vector Machine

![[../../attachments/Pasted image 20251023092036.png]]

- Theo như hình trên, để chia được margin:
	- $\mathbf{w}^T \mathbf{x}_i + b \ge 1 \quad \text{if y = +1}$ 
	- $\mathbf{w}^T \mathbf{x}_i + b \le -1 \quad \text{if y = -1}$ 
- Và đơn giản hóa:
$$
y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1
$$
![[../../attachments/Pasted image 20251023092505.png]]

### Giải thích kĩ

> 1. Tại sao công thức margin = $\frac2 {||{\vec{w}}||^2}$
> 2. Tại sao hai lề ở hình trên lại là f = -1 và f = +1 mà không phải là số khác

Hãy bắt đầu từ đường ranh giới (decision boundary). Nó là một đường thẳng (hoặc siêu phẳng) có phương trình:

$$f(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} - b = 0$$

1. **Vấn đề của việc "Co giãn" (Scaling)**
    - Hãy xem xét phương trình trên. Nếu tôi nhân toàn bộ phương trình với 5, tôi có một phương trình mới:
        $5(\mathbf{w} \cdot \mathbf{x} - b) = 0$
        $(5\mathbf{w}) \cdot \mathbf{x} - (5b) = 0$
    - Đây là một phương trình _khác_ (vì $\mathbf{w}' = 5\mathbf{w}$ và $b' = 5b$), nhưng nó biểu diễn _cùng một đường thẳng_ trong không gian.
    
 **Quy tắc mà SVM** đặt ra là: "Chúng ta sẽ co giãn (scale) $\mathbf{w}$ và $b$ sao cho các điểm dữ liệu gần nhất với đường ranh giới (các **support vectors**) sẽ cho ra giá trị $f(\mathbf{x}) = 1$ và $f(\mathbf{x}) = -1$".

2. **Khoảng cách từ điểm M đến đường thẳng d** là: 
	d(M; d) = ![Tính khoảng cách từ một điểm đến một đường thẳng](https://vietjack.com/toan-lop-10/images/cac-cong-thuc-ve-phuong-trinh-duong-thang-a03.PNG)

Từ công thức này ta có thể thấy khoảng cách **margin** sẽ là bằng 2 lần $d(x_i, \text{hyperplane})$
Với $x_i$ là điểm gần Vector Machine nhất (f = 0)

## Lưu ý 

Trong bài toán Hard Margin ta **không thể giải theo hướng Gradient Descent**
do hàm Loss có điều kiện "**cứng**": (giả sử hàm loss như sau)
$$
Loss = \frac 1 2 ||w^2|| - \sum [y_i (w * x_i + b) - 1]
$$
Trong đó:
- $||w^2||$: Mục đích là tối đa margin (margin = $\frac 2 {||w^2||}$)
- Ràng buộc của Hard margin: $y(w*x+b) \ge 1$

> Theo như hàm Loss trên ta không thể làm cho loss về 0 để đảm bảo điều hiện **cứng** của Hard margin
> 
> ❗**Vấn đề**:
   Đây là **bài toán tối ưu có ràng buộc**.  
   Ta **không thể chỉ lấy đạo hàm = 0** vì không phải mọi $(w,b)$ đều hợp lệ  (chỉ những cái thỏa ràng buộc thôi). => Không thỏa mãn khả vi để mà gradient
>
>> Do đó ta cần 1 cách tiếp cần mới => SMO

### Code (theo Soft margin)
```Python
class HardMarginSVM:
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = np.ones(2)
        self.bias = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        # Condition is whether current prediction have same sign with true_label -> That mean correct prediciton
        # If correct -> Update weight by decrease for normalize (maximize margin task)
        # If not correct -> Update w, b with gradient descent
        for _ in range(self.n_iterations):
            for i in range(n_samples):
                condition = y[i] * (np.dot(X[i], self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (self.weights)
                else:
                    self.weights -= self.lr * (self.weights - np.dot(X[i], y[i]))
                    self.bias -= self.lr * y[i]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)
```


# SMO

Thiết lập Dạng Đối Ngẫu (Derivation of the **Dual Form**)

Để giải quyết bài toán tối ưu hóa có ràng buộc này, ta sử dụng nhân tử **Lagrange** $a_n​≥0$, một nhân tử cho mỗi ràng buộc

Hàm Lagrangian L(w,b,a) được định nghĩa là: 
$$L(w, b,a) = \frac{1}{2} \|w\|^2 - \sum_{n=1}^N a_n \{ y_n(w^T \phi(x_n) + b) - 1 \} \quad
$$

**Đạo hàm**: 
$$
\frac{\partial L}{\partial w} = 0 \quad \Rightarrow \quad w = \sum_{n=1}^N a_n y_n \phi(x_n) \quad
$$
$$\frac{\partial L}{\partial b} = 0 \quad \Rightarrow \quad \sum_{n=1}^N a_n y_n = 0 \quad
$$

Bài toán Đối Ngẫu là **tối đa hóa** hàm $\hat L (a)$: $$\tilde{L}(a) = \sum_{n=1}^N a_n - \frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N a_n a_m y_n y_m k(x_n, x_m) \quad $$
> Vậy tại vì sao đang từ Minimize Loss function mà lại thành Maximize L(a) ?

Ta có 2 mục tiêu “đối nghịch”:
- Với w,b: ta muốn **minimize** $\mathcal{L}$ để tìm nghiệm primal tốt nhất.
- Với α: ta muốn **maximize** $\mathcal{L}$ để “ép” ràng buộc trở thành chặt nhất có thể.

Điều này dẫn đến cơ chế tự nhiên:
- Đối với $w, b$: ta tìm điểm tối thiểu của Loss, cố làm ràng buộc thỏa.
- Đối với $α$: ta tìm hệ số “phạt” lớn nhất sao cho ràng buộc bị ép về phía đúng (thỏa KKT).

> => Vì thế **Dual problem là “max over α”**.

![[../../attachments/Pasted image 20251109125945.png]]

### Kernel trick

Khi chuyển sang Dual, bạn thấy xuất hiện **dot product** giữa các điểm dữ liệu:

$x_i^\top x_j$

Đây chính là chỗ “phép màu” xảy ra.  
Nếu ta thay nó bằng một hàm **kernel** $K(x_i, x_j)$, ta có thể ngầm làm việc trong **không gian đặc trưng cao hơn** mà **không cần tính trực tiếp**.

 **🔹 Ví dụ:**

Giả sử dữ liệu không tách được tuyến tính trong 2D, nhưng tách được trong không gian 3D:

$$\phi(x) = [x_1^2, \sqrt{2}x_1x_2, x_2^2]$$

→ Trong primal, ta phải thật sự tính $\phi(x)$, rất tốn.  
Nhưng trong Dual, chỉ cần thay dot product:

$$x_i^\top x_j \to K(x_i,x_j) = (\phi(x_i)^\top \phi(x_j))$$

Mà nếu ta chọn kernel “thông minh” như:

- **Linear:** $$ K(x_i,x_j) = x_i^\top x_j$$
    
- **Polynomial:** $$K(x_i,x_j) = (x_i^\top x_j + c)^d$$
- **RBF (Gaussian):** $$K(x_i,x_j) = \exp(-\frac{\|x_i - x_j\|^2}{2\sigma^2})$$
- **Sigmoid:** $$K(x_i,x_j) = \tanh(\kappa x_i^\top x_j + \theta)$$
    

→ Ta có thể tách phi tuyến **mà không bao giờ rời khỏi không gian gốc**.

## Các trường hợp khác

> Thực tế dữ liệu không đẹp để chia rõ ràng thành 2 class

![[../../attachments/Pasted image 20251023092617.png]]
=> Ta cần dùng **Soft margin**

## Soft Margin

Bài toán tối ưu hóa (Quadratic Programming - QP) được thiết lập như sau: $$\arg \min_{w,b} \quad \frac{1}{2} \|w\|^2 \quad $$ với các ràng buộc (constraints): $$t_n (w^T \phi(x_n) + b) \ge 1, \quad \text{với } n = 1, \dots, N \quad $$ Trong đó:
