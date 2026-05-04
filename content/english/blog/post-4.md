---
title: "PPO - Reinforcement learning"
meta_title: ""
description: "One of RL models"
date: 2022-04-04T05:00:00Z
image: "/images/image-placeholder.png"
categories: ["AI"]
author: "Minh Van"
tags: ["Reinforcement-learning", "AI"]
draft: false
---

> PPO được xem là một trong những thuật toán "tiêu chuẩn vàng" hiện nay nhờ sự cân bằng tuyệt vời giữa hiệu suất, độ ổn định và tính dễ sử dụng.

| **Ưu điểm (Pros)**                                                                                                                                                                                                                                                        | **Nhược điểm (Cons)**                                                                                                                                                                               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Độ ổn định cao**: PPO giới hạn sự thay đổi của chính sách trong mỗi lần cập nhật, tránh được việc cập nhật quá lớn và đột ngột khiến quá trình học bị phá vỡ. <br>=> Điều này rất quan trọng trong môi trường nơi một quyết định sai lầm có thể gây nghẽn nghiêm trọng. | **Có thể hơi chậm trong một số trường hợp**: Cơ chế "clipping" để đảm bảo ổn định đôi khi làm chậm quá trình hội tụ so với các thuật toán khác nếu bài toán đòi hỏi sự thay đổi chính sách mạnh mẽ. |
| **Hiệu quả về dữ liệu (Sample Efficiency) tốt**: Tốt hơn so với các thuật toán policy gradient cơ bản vì nó có thể tái sử dụng dữ liệu từ các lượt tương tác cũ để cập nhật nhiều lần.                                                                                    | **Tối ưu siêu tham số (Hyperparameter Tuning)**: Vẫn cần tinh chỉnh một số siêu tham số như clipping range (phạm vi cắt) để đạt hiệu suất tốt nhất.                                                 |
| **Dễ triển khai và tinh chỉnh**: So với các thuật toán phức tạp hơn như TRPO (Trust Region Policy Optimization), PPO đơn giản hơn nhiều trong việc cài đặt và gỡ lỗi.                                                                                                     |                                                                                                                                                                                                     |
| **Phù hợp với cả không gian hành động liên tục và rời rạc**: Có thể áp dụng để điều chỉnh các tham số liên tục (ví dụ: phân bổ băng thông) hoặc các quyết định rời rạc (ví dụ: chọn một trong các đường định tuyến có sẵn).                                               |                                                                                                                                                                                                     |

# Steps (2 Giai đoạn)
## 1. Giai đoạn 1: Thu thập dữ liệu (Data Collection)
#### **a. Thu thập và xử lý Trạng thái (State)**

- **Agent nhận State:** Tại mỗi bước, agent quan sát trạng thái hiện tại của môi trường, ký hiệu là $S_t$.
    
    - **Ví dụ trong mạng:** `State` có thể là một vector hoặc ma trận chứa thông tin như: băng thông hiện tại của các liên kết, độ trễ, tỷ lệ mất gói, lưu lượng truy cập tại các nút mạng, v.v.
        
- **Xử lý State (nếu cần):** Thông thường, `state` thô sẽ được tiền xử lý. Ví dụ:
    
    - **Chuẩn hóa (Normalization):** Đưa các giá trị về cùng một thang đo (ví dụ: từ 0 đến 1) để mạng neuron học tốt hơn.
        
    - **Nối các frame (Frame stacking):** Trong các môi trường động, việc nối `state` của vài bước thời gian gần nhất lại với nhau giúp agent nhận biết được các thông tin về sự thay đổi, ví dụ như "tốc độ" hay "gia tốc" của lưu lượng mạng.

#### **b. Lựa chọn Hành động (Action Selection)**

Khi đã có `state` $S_t$, agent cần quyết định hành động $A_t$ tiếp theo. Đây là nhiệm vụ của mạng **Actor**.

- **Kiến trúc Actor:** Mạng Actor là một mạng neuron nhận đầu vào là `state` $S_t$ và cho ra đầu ra là một **phân phối xác suất** trên không gian các hành động.
        
    - **Hành động liên tục (Continuous):** Nếu hành động là điều chỉnh băng thông (một giá trị số thực), Actor sẽ output các tham số của một phân phối liên tục, thường là phân phối Gaussian (Normal Distribution), tức là output ra giá trị trung bình ($\mu$) và độ lệch chuẩn ($\sigma$). Agent sau đó sẽ lấy mẫu từ phân phối $N(\mu, \sigma)$ này để ra hành động cụ thể.
        
- **Quá trình chọn:**
    
    1. Đưa `state` $S_t$ vào mạng Actor.
    2. Nhận về phân phối xác suất của các hành động.
    3. Lấy mẫu một hành động $A_t$ từ phân phối này. Việc lấy mẫu (thay vì luôn chọn hành động có xác suất cao nhất) là cực kỳ quan trọng để agent có thể **khám phá (exploration)** các chiến lược mới.
    4. Agent cũng tính toán log của xác suất thực hiện hành động đó, $log\pi(A_t|S_t)$, giá trị này sẽ được dùng trong giai đoạn tối ưu hóa.

#### **c. Tương tác và Lưu trữ**

- Agent thực hiện hành động $A_t$ trong môi trường.
- Môi trường phản hồi lại bằng:
    - Trạng thái tiếp theo $S_{t+1}$.
    - Phần thưởng (Reward) $R_t$.
    - Tín hiệu kết thúc lượt chơi (Done).
    
- Agent lưu lại bộ dữ liệu kinh nghiệm này: `(State, Action, Reward, Next_State, Log_Probability, Done)`. Quá trình này lặp lại cho đến khi thu thập đủ một lượng dữ liệu nhất định (ví dụ: 2048 bước thời gian). Toàn bộ dữ liệu này được gọi là một **batch** hoặc **trajectory**.

## 2. Giai đoạn 2: Tối ưu hóa mô hình (Model Optimization) 

Sau khi thu thập đủ một batch dữ liệu, agent sẽ tạm dừng tương tác và dùng dữ liệu này để cập nhật, cải thiện hai mạng neuron của mình: **Actor** và **Critic**.

#### **a. Xây dựng và vai trò của Critic**

- **Kiến trúc Critic:** Mạng Critic cũng là một mạng neuron, nhận đầu vào là `state` $S_t$ và output ra một giá trị duy nhất gọi là **Value** ($V(S_t)$).
    
- **Vai trò:** Value $V(S_t)$ là một **ước tính** về tổng phần thưởng chiết khấu (discounted future rewards) mà agent **kỳ vọng** sẽ nhận được khi bắt đầu từ trạng thái $S_t$ và đi theo chính sách hiện tại. Nói cách khác, Critic "phán xét" xem một trạng thái tốt hay xấu như thế nào

#### **b. Xử lý dữ liệu sau khi thu thập**

Trước khi cập nhật, agent cần tính toán một đại lượng quan trọng từ dữ liệu thô: **Advantage (Lợi thế)**.

1. **Tính Value cho mỗi state:** Cho tất cả các `state` trong batch dữ liệu đi qua mạng Critic để có được giá trị $V(S_t)$.
    
2. **Tính Advantage $A_t$**: Advantage cho biết hành động $A_t$ tại trạng thái $S_t$ đã thực hiện tốt hơn hay tệ hơn so với **kỳ vọng trung bình** tại trạng thái đó. Một cách phổ biến để tính Advantage là dùng **Generalized Advantage Estimation (GAE)**:
       $$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$
  Trong đó:
    
-    $\delta_t = R_t + \gamma V(S_{t+1}) - V(S_t)$ là "sai số thời gian" (TD Error).
        
-  $\gamma$ (gamma) là hệ số chiết khấu.
        
-  $\lambda$ (lambda) là tham số làm mịn của GAE.
        
Về cơ bản, $\hat{A}_t > 0$ nghĩa là hành động đã thực hiện tốt hơn dự kiến, và $\hat{A}_t < 0$ nghĩa là nó tệ hơn dự kiến.
  

#### **c. Cập nhật Actor và Critic**

Agent sẽ lặp lại việc cập nhật trên cùng một batch dữ liệu trong vài **epoch**. Mỗi epoch, batch dữ liệu lại được chia thành các **mini-batch** nhỏ hơn để tối ưu hóa bằng Stochastic Gradient Descent.

- **Cập nhật Critic (Value Loss):**
    
    - Mục tiêu của Critic là dự đoán Value càng chính xác càng tốt. Nó được huấn luyện để tối thiểu hóa sai số giữa dự đoán $V(S_t)$ và "mục tiêu thực tế" (tổng phần thưởng nhận được).
        
    - Hàm mất mát thường là **Mean Squared Error (MSE)**:        $$ L_{critic} = \frac{1}{N} \sum (V_{target} - V(S_t))^2$$
- **Cập nhật Actor (Policy Loss):**
    
    - Đây là phần cốt lõi của PPO. Mục tiêu là tăng xác suất của những hành động mang lại Advantage cao và giảm xác suất của những hành động có Advantage thấp.
    - PPO sử dụng một hàm mục tiêu đặc biệt có cơ chế "cắt" (clipping) để ngăn việc cập nhật chính sách quá đột ngột, giúp quá trình học ổn định.
    - Tỷ lệ xác suất: $r_t(\theta) = \frac{\pi_{\theta}(A_t|S_t)}{\pi_{\theta_{old}}(A_t|S_t)}$ (tỷ lệ giữa chính sách mới và chính sách cũ).
    - Hàm mất mát của Actor:
        $$ L_{actor} = -\mathbb{E}_t [\min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t)]$$
        
    - Trong đó $\epsilon$ (epsilon) là một siêu tham số nhỏ (ví dụ: 0.2). Hàm `clip` này đảm bảo rằng sự thay đổi của chính sách sẽ bị giới hạn trong một khoảng an toàn $[1-\epsilon, 1+\epsilon]$.
        

Sau khi hoàn thành tất cả các epoch cập nhật, agent sẽ quay lại Giai đoạn 1 để thu thập một batch dữ liệu mới với chính sách đã được cải thiện.