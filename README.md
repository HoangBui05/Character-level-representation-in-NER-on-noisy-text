# BÁO CÁO CUỐI KỲ HỌC SÂU

## Vai trò của character-level representation trong Named Entity Recognition trên noisy text

## 1. Thông tin học phần và nhóm

| Thông tin | Nội dung |
|---|---|
| Tên môn học | Học sâu |
| Giảng viên hướng dẫn | Trần Quốc Hưng |
| Nhóm | 4 |
| Lớp | DHKL17A3HN |
| Tên đề tài | Vai trò của character-level representation trong Named Entity Recognition trên noisy text |

### Danh sách thành viên

| STT | Họ và tên | Mã sinh viên | Lớp | Nhiệm vụ |
|---:|---|---|---|---|
| 1 | Bùi Huy Hoàng | 23174600033 | DHKL17A3HN | [Nhiệm vụ] |
| 2 | Lê Anh Vũ | 23174600038 | DHKL17A3HN | [Nhiệm vụ] |
| 3 | Nguyễn Văn Tuấn Hưng | 23174600057 | DHKL17A3HN | [Nhiệm vụ] |
| 4 | Nguyễn Thị Phương Nhung | 23174600150 | DHKL17A3HN | [Nhiệm vụ] |


## 2. Tóm tắt đề tài

Đồ án khảo sát câu hỏi: **Biểu diễn ở mức ký tự có giúp mô hình nhận diện thực thể chưa xuất hiện hoặc hiếm gặp tốt hơn trên văn bản nhiễu hay không?**

Thí nghiệm sử dụng WNUT 17 và ba mô hình PyTorch huấn luyện hoàn toàn từ đầu:

1. **Baseline 1:** word embedding → BiLSTM → Softmax.
2. **Baseline 2 / ablation tắt character:** word embedding → BiLSTM → CRF.
3. **Mô hình đề xuất:** word embedding + character CNN → BiLSTM → CRF.

Hai mô hình thứ hai và thứ ba giữ cùng word encoder và CRF, khác nhau ở việc có hay không có character encoder. Vì vậy, phép so sánh này là ablation trực tiếp để khảo sát đóng góp của đặc trưng ký tự. Không sử dụng mô hình pretrained hoặc pretrained embeddings.

Trên lần chạy hiện tại, mô hình word+character đạt F1 exact-span **0,0910** trên nhóm unseen, so với **0,0441** của word-only CRF. Bootstrap 95% CI của mức chênh lệch là **[0,0265; 0,0655]**. Trên nhóm rare, F1 tăng từ **0,1538** lên **0,3243**, nhưng CI **[-0,0143; 0,3846]** còn chứa 0 và nhóm này chỉ có 22 thực thể; vì vậy chưa đủ bằng chứng để kết luận cải thiện chắc chắn.

## 3. Mô tả bộ dữ liệu

### 3.1. Nguồn và đặc điểm

- Dataset được tải bằng `datasets.load_dataset("dany0407/wnut_17")`.
- Đây là dữ liệu **WNUT 17**, dùng cho bài toán nhận diện thực thể trên văn bản xã hội/noisy text. Văn bản có thể chứa tên riêng mới, cách viết không chuẩn, chữ viết tắt, từ hiếm và ngữ cảnh ngắn.
- Dữ liệu đã được token hóa. Mỗi ví dụ có chuỗi `tokens` và chuỗi nhãn `ner_tags`; notebook giữ nguyên tokenization và căn chỉnh token–nhãn của dataset.
- Nhãn ở định dạng BIO: `O`, `B-<loại>` và `I-<loại>`. Các loại thực thể trong split train là `corporation`, `creative-work`, `group`, `location`, `person`, `product`.

### 3.2. Các split

| Split | Số câu |
|---|---:|
| Train | 3.394 |
| Validation | 1.009 |
| Test | 1.287 |

Dataset cung cấp sẵn cả ba split nên notebook không tách validation từ train. Test được dùng để báo cáo kết quả cuối; chọn checkpoint dựa trên validation.

### 3.3. Định nghĩa nhóm entity

Đếm tần suất **surface form của thực thể** trong train sau khi chuyển lowercase:

- **Unseen:** surface form chưa xuất hiện trong entity train.
- **Rare (1–2):** xuất hiện một hoặc hai lần trong entity train.
- **Frequent (3+):** xuất hiện ít nhất ba lần trong entity train.

Việc xác định các nhóm chỉ dựa trên train, không dùng nhãn test để xây vocabulary hoặc tính tần suất. Trên test có **1.079 gold entity**: **1.055 unseen**, **22 rare** và **2 frequent**. Phân bố rất lệch về unseen; support của frequent quá nhỏ để rút ra kết luận đáng tin cậy.

**Giới hạn của định nghĩa:** unseen ở đây nghĩa là *surface form entity* chưa từng xuất hiện trong train, không đồng nghĩa mọi token trong entity đều là từ OOV. Phép đo cũng không tạo các biến thể nhiễu tổng hợp để kiểm tra riêng tác động của lỗi chính tả, viết tắt hay kéo dài ký tự.

## 4. Cấu trúc notebook và diễn biến thực nghiệm

Toàn bộ mã nguồn, quy trình huấn luyện, đánh giá và phân tích lỗi nằm trong [`wnut17_character_ner.ipynb`](./wnut17_character_ner.ipynb). Các cell được tổ chức theo thứ tự:

1. **Thiết lập thí nghiệm:** import thư viện, seed, device, hyperparameter và cấu hình reproducibility.
2. **Nạp dữ liệu:** tải WNUT 17, kiểm tra features, nhãn và số lượng từng split.
3. **Chuẩn hóa dữ liệu:** ánh xạ mã nhãn sang BIO; trích xuất span và đếm tần suất surface form trên train để gán nhóm unseen/rare/frequent.
4. **Vocabulary và batching:** tạo word/character vocabulary từ train; padding chuỗi và gom batch; từ OOV ánh xạ về `<UNK>`.
5. **Định nghĩa mô hình:** xây CRF tuyến tính, word/character encoder và ba cấu hình mô hình so sánh.
6. **Huấn luyện:** train trên train, đo validation, chọn checkpoint theo entity-level validation F1; dừng sớm khi metric không cải thiện.
7. **Đánh giá trên test:** giải mã chuỗi nhãn, trích xuất span và tính precision/recall/F1 exact-span theo từng nhóm.
8. **Ablation và bootstrap:** so sánh word+character với word-only CRF bằng chênh lệch F1 và bootstrap confidence interval.
9. **Phân tích lỗi:** liệt kê các span bị bỏ sót hoặc dự đoán thừa/sai ranh giới, kèm câu chứa lỗi và nhóm entity.

## 5. Xây dựng mô hình

### 5.1. Tiền xử lý và biểu diễn đầu vào

- **Word vocabulary:** tạo từ token train; chỉ số 0 dành cho padding và từ ngoài vocabulary dùng `<UNK>`.
- **Character vocabulary:** tạo từ các ký tự xuất hiện trong token train; ký tự padding và ký tự không biết có chỉ số riêng.
- Vocabularies không được xây bằng validation/test. Các embedding được khởi tạo ngẫu nhiên và học trong quá trình train.
- Câu được padding theo độ dài lớn nhất trong batch. `pack_padded_sequence` ngăn BiLSTM xử lý phần padding ở cuối câu. Mask độ dài ký tự được dùng khi pooling đầu ra CNN.
- Nhãn padding bị loại khỏi token loss. CRF dùng độ dài thực của từng câu khi tính xác suất chuỗi và giải mã.

### 5.2. Baseline 1 — Word-BiLSTM-Softmax

1. Tra cứu vector word embedding kích thước **100** cho từng token.
2. Đưa chuỗi vector qua BiLSTM hai chiều, hidden size **64 mỗi hướng**.
3. Ghép hai hướng, áp dụng dropout **0,2**.
4. Linear classifier dự đoán phân phối nhãn BIO độc lập tại từng token.
5. Huấn luyện bằng weighted cross-entropy, bỏ qua vị trí padding.

Baseline này thể hiện cấu hình tuần tự hóa nhãn bằng BiLSTM nhưng không dùng CRF để mô hình hóa phụ thuộc nhãn giữa các token.

### 5.3. Baseline 2 và ablation — Word-BiLSTM-CRF

Encoder đầu vào vẫn chỉ dùng word embedding, sau đó qua cùng BiLSTM và linear layer. CRF học điểm chuyển tiếp giữa các nhãn, điểm bắt đầu/kết thúc chuỗi và giải mã chuỗi nhãn có tổng điểm cao nhất bằng Viterbi.

Đây là baseline mạnh hơn để so sánh character: nó giữ CRF như mô hình đề xuất, chỉ loại bỏ character CNN. Vì vậy, chênh lệch giữa baseline này và mô hình đề xuất ít bị nhiễu bởi thay đổi decoder.

### 5.4. Mô hình đề xuất — Word+Character CNN-BiLSTM-CRF

1. **Word path:** embedding kích thước 100.
2. **Character path:** embedding ký tự kích thước 30; `Conv1d` với 50 filter, kernel size 3 và padding 1; ReLU; max-pooling trên các vị trí ký tự hợp lệ để thu một vector cho mỗi token.
3. Ghép vector word 100 chiều và character 50 chiều thành biểu diễn token 150 chiều.
4. Đưa chuỗi biểu diễn token qua cùng BiLSTM hai chiều và dropout.
5. Linear layer tạo emission score; CRF tính loss chuỗi và Viterbi giải mã nhãn.

Character CNN học đặc trưng hình thái/chính tả của token, có thể chia sẻ tín hiệu giữa những từ có ký tự hoặc chuỗi con tương tự, kể cả khi toàn bộ từ chưa có trong word vocabulary.

### 5.5. Cấu hình huấn luyện

| Tham số | Giá trị |
|---|---:|
| Seed | 42 |
| Epoch tối đa | 12 |
| Batch size | 32 |
| Word embedding | 100 |
| Character embedding | 30 |
| Character CNN | 50 filter, kernel size 3 |
| BiLSTM hidden size | 64 mỗi hướng |
| Dropout | 0,2 |
| Optimizer | Adam |
| Learning rate | 0,001 |
| Gradient clipping | 5,0 |
| Early stopping patience | 4 epoch |

Để xử lý mất cân bằng giữa nhãn `O` và các nhãn entity, notebook dùng trọng số lớp tỷ lệ với căn bậc hai của nghịch đảo tần suất nhãn train. Với mô hình CRF, objective cộng CRF negative log-likelihood và **0,5 × weighted token cross-entropy**; thành phần token-level giúp cung cấp thêm tín hiệu học trong điều kiện nhãn entity thưa.

Checkpoint được chọn theo **entity exact-span F1 trên validation**; nếu bằng điểm thì ưu tiên validation loss thấp hơn. Thiết lập này phù hợp mục tiêu entity-level hơn việc chọn checkpoint chỉ dựa vào loss.

## 6. Phương pháp đánh giá và metrics

### 6.1. Entity-level exact-span

Một entity dự đoán đúng khi đồng thời khớp:

- vị trí bắt đầu;
- vị trí kết thúc;
- loại entity.

Do đó, dự đoán đúng một phần tên entity hoặc đúng span nhưng sai loại vẫn không được tính là true positive.

Từ số đếm micro trên các entity:

- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`
- `F1 = 2 × Precision × Recall / (Precision + Recall)`

Notebook báo cáo precision, recall và F1 trên toàn bộ test cũng như riêng từng nhóm unseen, rare và frequent. `support` là số gold entity trong nhóm. Với F1 nhóm, gold span và predicted span được gán nhóm dựa trên tần suất surface form trong train, rồi so khớp exact-span trong nhóm đó.

### 6.2. Paired bootstrap confidence interval

Để so sánh mô hình đề xuất với baseline 2, notebook bootstrap **theo câu** trên test, lấy mẫu có hoàn lại **1.000 lần**. Trong mỗi lần lấy mẫu, tính lại micro-F1 của cả hai mô hình trên cùng các câu và lấy hiệu:

`ΔF1 = F1(word+character) − F1(word-only CRF)`

CI 95% là hai phân vị 2,5% và 97,5% của phân phối ΔF1 bootstrap. Dùng cùng mẫu câu cho hai mô hình tạo phép so sánh paired. CI không chứa 0 là bằng chứng về khác biệt trên split test theo phương pháp này; CI chứa 0 nghĩa là chưa đủ bằng chứng, **không chứng minh hai mô hình tương đương**.

Bootstrap này lượng hóa biến thiên lấy mẫu test; nó **không** đo biến thiên do khởi tạo/training seed. Notebook hiện chỉ chạy một seed.

### 6.3. Phạm vi diễn giải

Các con số là kết quả của một lần chạy, một cấu hình và một split test. Dataset có rất ít rare/frequent entity, nên metric ở các nhóm nhỏ dao động mạnh. Chỉ nên kết luận cho cấu hình và split hiện tại, chưa khái quát thành character representation luôn cải thiện mọi bài toán NER.

## 7. Kết quả thực nghiệm

### 7.1. Exact-span precision / recall / F1

| Mô hình | Nhóm entity | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| Baseline 1: Word-BiLSTM-Softmax | Tất cả | 0,0548 | 0,0111 | 0,0185 | 1.079 |
| Baseline 1: Word-BiLSTM-Softmax | Unseen | 0,0563 | 0,0114 | 0,0189 | 1.055 |
| Baseline 1: Word-BiLSTM-Softmax | Rare (1–2) | 0,0000 | 0,0000 | 0,0000 | 22 |
| Baseline 1: Word-BiLSTM-Softmax | Frequent (3+) | 0,0000 | 0,0000 | 0,0000 | 2 |
| Baseline 2 / ablation: Word-BiLSTM-CRF | Tất cả | 0,1174 | 0,0287 | 0,0462 | 1.079 |
| Baseline 2 / ablation: Word-BiLSTM-CRF | Unseen | 0,1115 | 0,0275 | 0,0441 | 1.055 |
| Baseline 2 / ablation: Word-BiLSTM-CRF | Rare (1–2) | 0,5000 | 0,0909 | 0,1538 | 22 |
| Baseline 2 / ablation: Word-BiLSTM-CRF | Frequent (3+) | 0,0000 | 0,0000 | 0,0000 | 2 |
| Proposed: Word+CharCNN-BiLSTM-CRF | Tất cả | 0,0813 | 0,1121 | **0,0943** | 1.079 |
| Proposed: Word+CharCNN-BiLSTM-CRF | Unseen | 0,0781 | 0,1090 | **0,0910** | 1.055 |
| Proposed: Word+CharCNN-BiLSTM-CRF | Rare (1–2) | 0,4000 | 0,2727 | **0,3243** | 22 |
| Proposed: Word+CharCNN-BiLSTM-CRF | Frequent (3+) | 0,0000 | 0,0000 | 0,0000 | 2 |

### 7.2. Ablation: chênh lệch F1 của character representation

| Nhóm | F1 word+character | F1 word-only CRF | ΔF1 | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Tất cả | 0,0943 | 0,0462 | +0,0481 | [0,0277; 0,0662] |
| Unseen | 0,0910 | 0,0441 | **+0,0469** | **[0,0265; 0,0655]** |
| Rare (1–2) | 0,3243 | 0,1538 | +0,1705 | [-0,0143; 0,3846] |
| Frequent (3+) | 0,0000 | 0,0000 | 0,0000 | [0,0000; 0,0000] |

### 7.3. Nhận xét

- Trên **unseen**, F1 của mô hình có character tăng khoảng 0,0469 điểm so với ablation word-only CRF; CI bootstrap không chứa 0. Đây là bằng chứng ủng hộ character representation cho nhóm này trong thí nghiệm hiện tại.
- Trên **rare**, F1 quan sát tăng khoảng 0,1705 điểm, nhưng CI chứa 0 và support chỉ 22. Có tín hiệu tích cực, chưa đủ cơ sở để kết luận chắc chắn.
- Trên toàn bộ test, mô hình đề xuất có F1 cao hơn word-only CRF (0,0943 so với 0,0462), nhưng precision vẫn thấp (0,0813). Recall tăng không đảm bảo dự đoán entity đã chính xác.
- Baseline 1 thấp hơn baseline 2 trong lần chạy này. Điều đó cho thấy CRF giúp kết quả của cấu hình word-only hiện tại, nhưng không tách biệt tác động của CRF với các yếu tố khác nếu chỉ dựa trên một lần chạy.
- Frequent chỉ có hai gold entity và cả ba mô hình không nhận diện đúng entity nào thuộc nhóm này; đây không phải bằng chứng rằng biểu diễn ký tự không hữu ích với thực thể phổ biến.

## 8. Phân tích lỗi

Notebook lưu ví dụ lỗi vào [`wnut17_error_examples.csv`](./wnut17_error_examples.csv). Một số dạng lỗi quan sát được ở mô hình đề xuất:

1. **Sai ranh giới:** gold `Gurez sector [location]`, dự đoán chỉ `Gurez [location]`.
2. **Bỏ sót tên riêng nhiều token:** bỏ sót `Colonel Rajesh Kalia [person]`, trong khi dự đoán các đoạn chồng lấn như `Defence Spokesman Colonel Rajesh Kalia [person]` hoặc `Rajesh Kalia [person]`.
3. **Gán nhãn thực thể cho từ thường:** dự đoán `inhospitable terrain [location]` hoặc `military spokesman [location]`.
4. **Entity mới/ít ngữ cảnh:** các tên mới, viết tắt và tên tổ chức như `ART` dễ bị bỏ sót hoặc gán sai loại.

Trong file lỗi, mỗi span gold bị bỏ sót được ghi là FN và mỗi span dự đoán thừa/sai ghi là FP. Một lỗi sai ranh giới hoặc loại có thể tạo đồng thời một bản ghi FN và một bản ghi FP; số dòng lỗi do đó không phải số câu lỗi độc lập.

Các ví dụ gợi ý rằng character representation có thể cung cấp tín hiệu hữu ích để tăng recall, nhưng chưa giải quyết triệt để dự đoán thừa, phân biệt loại entity và ranh giới span. Cải thiện F1 nên được đọc cùng precision/recall và kiểm tra ví dụ cụ thể.

## 9. Kết luận

Với split test và seed hiện tại, kết quả trả lời câu hỏi nghiên cứu như sau:

> **Có bằng chứng thực nghiệm character-level representation giúp nhận diện unseen entity tốt hơn so với word-only BiLSTM-CRF trong cấu hình này.** Trên unseen, F1 tăng từ 0,0441 lên 0,0910 và bootstrap 95% CI của ΔF1 là [0,0265; 0,0655]. Với rare entity, F1 quan sát tăng nhưng khoảng tin cậy còn chứa 0 và số mẫu ít, nên chưa thể khẳng định mức cải thiện.

F1 tuyệt đối còn thấp, cho thấy cấu hình hiện tại vẫn gặp khó khăn trên WNUT 17. Các hạn chế gồm một seed, chỉ một split test, số lượng rare/frequent nhỏ, độ nhạy của phép ghép surface form, và chưa đánh giá riêng bằng các phép biến đổi nhiễu có kiểm soát. Nên chạy nhiều seed và báo cáo độ biến thiên giữa các lần huấn luyện trước khi xem kết quả là ổn định.

## 10. Cách cài đặt và chạy

### Yêu cầu

- Python 3.10+ (đã chạy với Python 3.13).
- PyTorch; CUDA được dùng tự động nếu môi trường cài bản PyTorch hỗ trợ CUDA, nếu không sẽ chạy CPU.
- Các thư viện: `datasets`, `numpy`, `pandas`, `matplotlib`.
- Có kết nối Internet lần đầu để tải dataset từ Hugging Face.

### Cài thư viện

Trong notebook, bỏ comment ở cell setup nếu cần cài bổ sung:

```python
%pip install datasets numpy pandas matplotlib
```

Cài PyTorch theo đúng phiên bản CUDA hoặc CPU tương thích với máy từ hướng dẫn cài đặt PyTorch chính thức.

### Chạy

1. Mở `wnut17_character_ner.ipynb` bằng Jupyter hoặc VS Code.
2. Chọn Python kernel có PyTorch và các thư viện trên.
3. Chạy lần lượt từ trên xuống (`Run All`).
4. Theo dõi log train, bảng metrics, bootstrap CI và ví dụ lỗi trong các cell cuối.

Lần chạy tham chiếu trong workspace dùng **Python 3.13.0, PyTorch 2.15.0+cpu**, seed 42. Không có CUDA trên môi trường đó. Do huấn luyện ba mô hình trên CPU có thể mất vài phút, có thể giảm `EPOCHS` để kiểm tra nhanh; kết quả khi đổi cấu hình sẽ không còn là kết quả báo cáo trong README.

## 11. Tệp đầu ra

- [`wnut17_character_ner.ipynb`](./wnut17_character_ner.ipynb): mã nguồn, log và kết quả của notebook.
- [`wnut17_entity_metrics.csv`](./wnut17_entity_metrics.csv): precision, recall, F1 và support theo mô hình/nhóm.
- [`wnut17_ablation_bootstrap.csv`](./wnut17_ablation_bootstrap.csv): chênh lệch F1 và bootstrap CI.
- [`wnut17_error_examples.csv`](./wnut17_error_examples.csv): ví dụ FN/FP và ngữ cảnh.
