# Bài thực hành 3

- `Bài làm/Assignment_1/`: bài 1, LSTM 5 lớp, hidden size 256 trên UIT-VSFC.
- `Bài làm/Assignment_2/`: bài 2, GRU 5 lớp, hidden size 256 trên UIT-VSFC.
- `Bài làm/Assignment_3/`: bài 3, BiLSTM 5 lớp, hidden size 256 cho PhoNER.

Chạy từ thư mục gốc `C:\DS201\LAB3`:

```powershell
python "Bài làm\Assignment_1\assignment_1.py"
python "Bài làm\Assignment_2\assignment_2.py"
python "Bài làm\Assignment_3\assignment_3.py" --data-dir "datasets\PhoNER_COVID19\data\word"
```

## Kết quả bài 1

Lần chạy ngày 02/11/2026 sử dụng mô hình LSTM 5 lớp, hidden size 256,
batch size 32 và Adam với learning rate 0.001 trên nhãn `topic` của UIT-VSFC.

| Epoch | Training loss | Validation macro-F1 |
| ---: | ---: | ---: |
| 1 | 0.574 | 0.5936 |
| 2 | 0.417 | 0.7057 |
| 3 | 0.356 | 0.6889 |
| 4 | 0.317 | 0.7271 |
| 5 | 0.291 | 0.7452 |
| 6 | 0.261 | 0.7396 |
| **7** | **0.234** | **0.7546** |
| 8 | 0.204 | 0.7380 |
| 9 | 0.184 | 0.7488 |
| 10 | 0.170 | 0.7456 |
| 11 | 0.137 | 0.7502 |
| 12 | 0.126 | 0.7404 |

- Validation macro-F1 tốt nhất: **0.7546** tại epoch 7.
- Early stopping tại epoch 12 sau 5 epoch liên tiếp không cải thiện.
- Test macro-F1: **0.7564** (3.166 mẫu, 99 batch).
- Tổng thời gian train xấp xỉ 16 phút, không tính thời gian khởi tạo.

Loss huấn luyện tiếp tục giảm trong khi validation F1 không cải thiện sau
epoch 7, cho thấy mô hình bắt đầu overfit. Mã bài 1 hiện không lưu và khôi
phục checkpoint tốt nhất; vì vậy kết quả test trên được tính bằng
trọng số của epoch 12, không phải epoch 7.

## Kết quả bài 2

Lần chạy ngày 02/11/2026 sử dụng CPU, mô hình GRU 5 lớp, hidden size
256, batch size 32 và Adam với learning rate 0.001 trên nhãn `topic` của
UIT-VSFC.

| Epoch | Training loss | Validation macro-F1 |
| ---: | ---: | ---: |
| 1 | 0.4920 | 0.6447 |
| 2 | 0.3610 | 0.7519 |
| 3 | 0.3221 | 0.7174 |
| 4 | 0.2800 | 0.7409 |
| 5 | 0.2508 | 0.7381 |
| 6 | 0.2092 | 0.7618 |
| 7 | 0.1819 | 0.7433 |
| 8 | 0.1646 | 0.7409 |
| **9** | **0.1419** | **0.7619** |
| 10 | 0.1219 | 0.7452 |
| 11 | 0.1033 | 0.7478 |
| 12 | 0.0920 | 0.7367 |
| 13 | 0.0809 | 0.7330 |
| 14 | 0.0894 | 0.7324 |

- Validation macro-F1 tốt nhất: **0.7619** tại epoch 9.
- Early stopping tại epoch 14 sau 5 epoch liên tiếp không cải thiện.
- Test macro-F1 của checkpoint tốt nhất: **0.7641** (3.166 mẫu, 99 batch).
- Tổng thời gian train xấp xỉ 19 phút 30 giây, không tính thời gian khởi tạo.
- Checkpoint được lưu tại `Assignment_2/best_gru.pt`.

Training loss giảm mạnh trong khi validation F1 không cải thiện sau epoch
9, cho thấy mô hình bắt đầu overfit. Trước khi đánh giá test, chương
trình đã tải lại checkpoint có validation F1 tốt nhất.

## Kết quả bài 3

Lần chạy ngày 02/11/2026 sử dụng CPU, encoder BiLSTM hai chiều 5
lớp, hidden size 256, batch size 32 và Adam với learning rate 0.001 trên
PhoNER word-level.

| Epoch | Training loss | Validation entity-level micro-F1 |
| ---: | ---: | ---: |
| 1 | 0.8437 | 0.5105 |
| 2 | 0.2133 | 0.8086 |
| 3 | 0.1004 | 0.8491 |
| 4 | 0.0648 | 0.8559 |
| 5 | 0.0466 | 0.8678 |
| 6 | 0.0337 | 0.8799 |
| 7 | 0.0292 | 0.8829 |
| 8 | 0.0216 | 0.8851 |
| 9 | 0.0175 | 0.8889 |
| 10 | 0.0159 | 0.8794 |
| 11 | 0.0118 | 0.8898 |
| 12 | 0.0123 | 0.8883 |
| 13 | 0.0101 | 0.8876 |
| **14** | **0.0088** | **0.8926** |
| 15 | 0.0073 | 0.8923 |
| 16 | 0.0088 | 0.8796 |
| **17** | **0.0108** | **0.8926** |
| 18 | 0.0102 | 0.8873 |
| 19 | 0.0107 | 0.8880 |

- Validation entity-level micro-F1 cao nhất theo số hiển thị: **0.8926**
  tại epoch 14 và 17.
- Test entity-level micro-F1 của checkpoint tốt nhất: **0.8822**
  (3.000 câu, 94 batch).
- Thời gian train và validation 19 epoch xấp xỉ 1 giờ 9 phút trên CPU.
- Checkpoint được lưu tại `Assignment_3/best_bilstm_ner.pt`.

Log huấn luyện kết thúc sau epoch 19, chưa có thông báo early stopping và
chưa tự đánh giá test. Kết quả test trên được chạy riêng bằng
checkpoint tốt nhất đã lưu, không huấn luyện lại mô hình.

Bài 2 mặc định phân loại `topic`; dùng `--label-key sentiment` nếu muốn phân loại cảm xúc.

Bài 3 cần ba file CoNLL chính thức của PhoNER: `train_word.conll`, `dev_word.conll`, và `test_word.conll`. Mỗi dòng gồm token và nhãn BIO; câu cách nhau bằng dòng trống.
