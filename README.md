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

Mô hình thứ hai là ablation tắt character của mô hình thứ ba, cùng dùng CRF. Notebook tìm siêu tham số riêng cho từng mô hình với cùng ngân sách và cùng dãy cấu hình ứng viên; vì cấu hình được chọn có thể khác nhau, so sánh này không cô lập tuyệt đối riêng tác động của character encoder. Không sử dụng mô hình pretrained hoặc pretrained embeddings.

Notebook phiên bản tuned chọn siêu tham số bằng random search trên validation rồi huấn luyện lại từng mô hình. Trong kết quả đã lưu, word+character đạt F1 exact-span **0,1475** trên nhóm unseen, so với **0,0752** của word-only CRF; bootstrap 95% CI của mức chênh lệch là **[0,0489; 0,0974]**. Trên nhóm rare, F1 của mô hình có character là **0,0000**, thấp hơn **0,1538** của word-only CRF; nhóm này chỉ có 22 thực thể nên kết quả còn nhiều bất định.

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

**Giới hạn của định nghĩa:** unseen ở đây nghĩa là *surface form entity* chưa từng xuất hiện trong train, không đồng nghĩa mọi token trong entity đều là từ OOV. Tỷ lệ token OOV so với word vocabulary train trong lần chạy notebook là **16,7% trên validation** và **21,9% trên test**. Phép đo không tạo các biến thể nhiễu tổng hợp để kiểm tra riêng tác động của lỗi chính tả, viết tắt hay kéo dài ký tự.

## 4. Cấu trúc notebook và diễn biến thực nghiệm

Toàn bộ mã nguồn, quy trình huấn luyện, đánh giá và phân tích lỗi nằm trong [`wnut17_character_ner_tuned.ipynb`](./wnut17_character_ner_tuned.ipynb). Các cell được tổ chức theo thứ tự:

1. **Thiết lập thí nghiệm:** import thư viện, seed, device, hyperparameter và cấu hình reproducibility.
2. **Nạp dữ liệu:** tải WNUT 17, kiểm tra features, nhãn và số lượng từng split.
3. **Chuẩn hóa dữ liệu:** ánh xạ mã nhãn sang BIO; trích xuất span và đếm tần suất surface form trên train để gán nhóm unseen/rare/frequent.
4. **Vocabulary và batching:** tạo word/character vocabulary chỉ từ train; word hiếm được thay bằng `<UNK>` ngẫu nhiên trong lúc train để học embedding cho từ OOV; giới hạn token ở 30 ký tự.
5. **Định nghĩa mô hình:** xây CRF tuyến tính, word/character encoder và ba cấu hình mô hình so sánh.
6. **Tối ưu siêu tham số:** random search 8 cấu hình cho mỗi mô hình; chọn theo entity-level validation F1, hòa thì dùng validation loss thấp hơn.
7. **Huấn luyện:** huấn luyện lại từng mô hình bằng cấu hình được chọn, tối đa 30 epoch và early stopping theo validation.
8. **Đánh giá trên test:** giải mã chuỗi nhãn, trích xuất span và tính precision/recall/F1 exact-span theo từng nhóm.
9. **Ablation, bootstrap và phân tích lỗi:** so sánh word+character với word-only CRF bằng paired bootstrap theo câu; xuất bảng metrics, CI và các lỗi FN/FP.

## 5. Xây dựng mô hình

### 5.1. Tiền xử lý và biểu diễn đầu vào

- **Word vocabulary:** tạo từ token train; chỉ số 0 dành cho padding và từ ngoài vocabulary dùng `<UNK>`.
- **Character vocabulary:** tạo từ các ký tự xuất hiện trong token train; ký tự padding và ký tự không biết có chỉ số riêng.
- Vocabularies không được xây bằng validation/test. Word/character embeddings được khởi tạo ngẫu nhiên và học trong quá trình train.
- Khi train, xác suất thay một từ bằng `<UNK>` là `alpha / (count + alpha)`; cách này giúp embedding `<UNK>` nhận được tín hiệu học dù từ OOV không có trong train.
- Token dài hơn **30 ký tự** bị cắt để giới hạn bộ nhớ. Câu được padding theo độ dài lớn nhất trong batch; `pack_padded_sequence` ngăn BiLSTM xử lý phần padding ở cuối câu. Mask độ dài ký tự được dùng khi pooling đầu ra CNN.
- Nhãn padding bị loại khỏi token loss. CRF dùng độ dài thực của từng câu khi tính xác suất chuỗi và giải mã.

### 5.2. Baseline 1 — Word-BiLSTM-Softmax

1. Tra cứu vector word embedding cho từng token.
2. Đưa chuỗi vector qua BiLSTM hai chiều.
3. Ghép hai hướng, áp dụng dropout.
4. Linear classifier dự đoán phân phối nhãn BIO độc lập tại từng token.
5. Huấn luyện bằng cross-entropy, bỏ qua vị trí padding.

Baseline này thể hiện cấu hình tuần tự hóa nhãn bằng BiLSTM nhưng không dùng CRF để mô hình hóa phụ thuộc nhãn giữa các token.

### 5.3. Baseline 2 và ablation — Word-BiLSTM-CRF

Encoder đầu vào vẫn chỉ dùng word embedding, sau đó qua cùng BiLSTM và linear layer. CRF học điểm chuyển tiếp giữa các nhãn, điểm bắt đầu/kết thúc chuỗi và giải mã chuỗi nhãn có tổng điểm cao nhất bằng Viterbi.

Đây là baseline mạnh hơn để so sánh character: nó giữ CRF như mô hình đề xuất nhưng không có character CNN. Chênh lệch không bị nhiễu bởi thay đổi decoder, dù các siêu tham số tốt nhất của hai mô hình được chọn riêng và không nhất thiết giống nhau.

### 5.4. Mô hình đề xuất — Word+Character CNN-BiLSTM-CRF

1. **Word path:** embedding từ.
2. **Character path:** embedding ký tự; `Conv1d` với kernel size 3, ReLU và max-pooling trên các vị trí ký tự hợp lệ để thu một vector cho mỗi token.
3. Ghép vector word và character thành biểu diễn token.
4. Đưa chuỗi biểu diễn token qua BiLSTM hai chiều và dropout.
5. Linear layer tạo emission score; CRF tính loss chuỗi và Viterbi giải mã nhãn.

Character CNN học đặc trưng hình thái/chính tả của token, có thể chia sẻ tín hiệu giữa những từ có ký tự hoặc chuỗi con tương tự, kể cả khi toàn bộ từ chưa có trong word vocabulary.

### 5.5. Huấn luyện và tối ưu siêu tham số

Notebook đặt seed **42**, dùng Adam và gradient clipping **5,0**. Tối đa **30 epoch**, early stopping patience **8** theo validation entity-F1. Trước khi huấn luyện đầy đủ, random search thử **8 cấu hình cho mỗi mô hình** (mỗi lượt tối đa 15 epoch, patience 4); ba mô hình dùng cùng dãy cấu hình ngẫu nhiên và cùng ngân sách. Không gian tìm kiếm gồm learning rate, hidden size, dropout, word/character embedding, số filter CNN, hệ số thay từ bằng `<UNK>` và batch size. Chọn cấu hình theo validation entity-F1; nếu hòa, ưu tiên validation loss thấp hơn. Test không tham gia chọn cấu hình hoặc checkpoint.

Các cấu hình tốt nhất được chọn trong lần chạy lưu trong notebook:

| Mô hình | Learning rate | Word dim | Char dim | CNN filters | BiLSTM hidden mỗi hướng | Dropout | α `<UNK>` | Batch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Word-BiLSTM-Softmax | 0,003909 | 200 | — | — | 256 | 0,6 | 2,0 | 16 |
| Word-BiLSTM-CRF (ablation) | 0,003909 | 200 | — | — | 256 | 0,6 | 2,0 | 16 |
| Word+CharCNN-BiLSTM-CRF | 0,001388 | 50 | 50 | 30 (kernel 3) | 256 | 0,5 | 0,5 | 32 |

Softmax dùng cross-entropy; hai mô hình CRF dùng negative log-likelihood của CRF. Mỗi mô hình chỉ dùng loss tương ứng với decoder, không dùng class weight hay loss phụ. Ablation và mô hình đề xuất giữ cùng decoder/loss CRF, nhưng cấu hình siêu tham số được chọn riêng nên không phải phép so sánh chỉ thay đổi duy nhất encoder ký tự.

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
| Baseline 1: Word-BiLSTM-Softmax | Tất cả | 0,1165 | 0,0334 | 0,0519 | 1.079 |
| Baseline 1: Word-BiLSTM-Softmax | Unseen | 0,1126 | 0,0322 | 0,0501 | 1.055 |
| Baseline 1: Word-BiLSTM-Softmax | Rare (1–2) | 0,4000 | 0,0909 | 0,1481 | 22 |
| Baseline 1: Word-BiLSTM-Softmax | Frequent (3+) | 0,0000 | 0,0000 | 0,0000 | 2 |
| Baseline 2 / ablation: Word-BiLSTM-CRF | Tất cả | 0,1738 | 0,0491 | 0,0766 | 1.079 |
| Baseline 2 / ablation: Word-BiLSTM-CRF | Unseen | 0,1694 | 0,0483 | 0,0752 | 1.055 |
| Baseline 2 / ablation: Word-BiLSTM-CRF | Rare (1–2) | 0,5000 | 0,0909 | 0,1538 | 22 |
| Baseline 2 / ablation: Word-BiLSTM-CRF | Frequent (3+) | 0,0000 | 0,0000 | 0,0000 | 2 |
| Proposed: Word+CharCNN-BiLSTM-CRF | Tất cả | 0,2153 | 0,1094 | **0,1451** | 1.079 |
| Proposed: Word+CharCNN-BiLSTM-CRF | Unseen | 0,2165 | 0,1118 | **0,1475** | 1.055 |
| Proposed: Word+CharCNN-BiLSTM-CRF | Rare (1–2) | 0,0000 | 0,0000 | **0,0000** | 22 |
| Proposed: Word+CharCNN-BiLSTM-CRF | Frequent (3+) | 0,0000 | 0,0000 | 0,0000 | 2 |

### 7.2. Ablation: chênh lệch F1 của character representation

| Nhóm | F1 word+character | F1 word-only CRF | ΔF1 | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Tất cả | 0,1451 | 0,0766 | +0,0685 | [0,0439; 0,0929] |
| Unseen | 0,1475 | 0,0752 | **+0,0723** | **[0,0489; 0,0974]** |
| Rare (1–2) | 0,0000 | 0,1538 | -0,1538 | [-0,3529; 0,0000] |
| Frequent (3+) | 0,0000 | 0,0000 | 0,0000 | [0,0000; 0,0000] |

### 7.3. Nhận xét

- Trên **unseen**, F1 của mô hình có character cao hơn word-only CRF 0,0723; CI bootstrap không chứa 0. Đây là bằng chứng ủng hộ character representation cho nhóm này trong lần chạy hiện tại.
- Trên **rare**, mô hình có character không nhận diện đúng entity nào (F1 0,0000 so với 0,1538 của CRF word-only). CI có cận trên bằng 0 và support chỉ 22, nên kết quả nhóm này còn nhiều bất định.
- Trên toàn bộ test, mô hình đề xuất có F1 cao hơn word-only CRF (0,1451 so với 0,0766), đồng thời recall cao hơn (0,1094 so với 0,0491). Đây là kết quả của cấu hình đã chọn bằng validation, không đảm bảo sẽ lặp lại ở seed khác.
- Baseline 2 có F1 cao hơn baseline 1 trong lần chạy này (0,0766 so với 0,0519). Tuy nhiên, do siêu tham số được tối ưu riêng, không thể quy toàn bộ chênh lệch cho riêng việc dùng CRF.
- Frequent chỉ có hai gold entity và cả ba mô hình không nhận diện đúng entity nào thuộc nhóm này; đây không phải bằng chứng rằng biểu diễn ký tự không hữu ích với thực thể phổ biến.

## 8. Phân tích lỗi

Notebook lưu ví dụ lỗi vào [`wnut17_error_examples.csv`](./wnut17_error_examples.csv). Một số dạng lỗi quan sát được ở mô hình đề xuất:

1. **Sai ranh giới:** các span nhiều token như `Gurez sector [location]` có thể bị tách/cắt ranh giới.
2. **Bỏ sót tên riêng nhiều token:** ví dụ `Colonel Rajesh Kalia [person]`.
3. **Nhầm loại hoặc dự đoán thừa:** tên viết tắt/tổ chức như `ANI` hoặc `ART` có thể bị bỏ sót hoặc gán nhãn entity không chính xác.

Trong file lỗi, mỗi span gold bị bỏ sót được ghi là FN và mỗi span dự đoán thừa/sai ghi là FP. Một lỗi sai ranh giới hoặc loại có thể tạo đồng thời một bản ghi FN và một bản ghi FP; số dòng lỗi do đó không phải số câu lỗi độc lập.

Các ví dụ gợi ý rằng character representation đi kèm recall cao hơn trên nhóm unseen trong lần chạy này, nhưng mô hình vẫn mắc lỗi dự đoán thừa, phân biệt loại entity và ranh giới span. Cải thiện F1 nên được đọc cùng precision/recall và kiểm tra ví dụ cụ thể.

## 9. Kết luận

Với split test và seed hiện tại, kết quả trả lời câu hỏi nghiên cứu như sau:

> **Trong lần chạy này, character-level representation đi kèm F1 cao hơn trên unseen entity so với word-only BiLSTM-CRF.** F1 tăng từ 0,0752 lên 0,1475; bootstrap 95% CI của ΔF1 là [0,0489; 0,0974]. Tuy nhiên, trên rare entity, mô hình có character đạt F1 0,0000 so với 0,1538 của ablation. Không thể kết luận character luôn cải thiện kết quả; kết quả còn phụ thuộc nhóm entity và cấu hình.

Các hạn chế gồm một seed, một split test, số lượng rare/frequent nhỏ, random search giới hạn ở 8 cấu hình cho mỗi mô hình, độ nhạy của phép ghép surface form, và chưa đánh giá riêng bằng các phép biến đổi nhiễu có kiểm soát. Nên chạy nhiều seed và mở rộng tìm kiếm siêu tham số trước khi xem kết quả là ổn định.

## 10. Cách cài đặt và chạy

### Yêu cầu

- Python tương thích với phiên bản PyTorch được cài.
- PyTorch; CUDA được dùng tự động nếu môi trường cài bản PyTorch hỗ trợ CUDA, nếu không sẽ chạy CPU.
- Các thư viện: `datasets`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`.
- Có kết nối Internet lần đầu để tải dataset từ Hugging Face.

### Cài thư viện

Trong notebook, bỏ comment ở cell setup nếu cần cài bổ sung:

```python
%pip install datasets pandas matplotlib scikit-learn
```

Cài PyTorch theo đúng phiên bản CUDA hoặc CPU tương thích với máy từ hướng dẫn cài đặt PyTorch chính thức.

### Chạy

1. Mở `wnut17_character_ner_tuned.ipynb` bằng Jupyter hoặc VS Code.
2. Chọn Python kernel có PyTorch và các thư viện trên.
3. Chạy lần lượt từ trên xuống (`Run All`); lần đầu cần Internet để tải WNUT 17 từ Hugging Face.
4. Notebook mặc định chạy hyperparameter search (`RUN_TUNING = True`), sau đó train ba mô hình và tạo các file CSV ở thư mục làm việc.

Kết quả được lưu trong notebook được tạo bằng **PyTorch 2.11.0+cu130 trên NVIDIA Tesla T4**, seed 42. Trên máy không có CUDA, notebook tự chuyển sang CPU; random search và huấn luyện có thể mất nhiều thời gian. Có thể đặt `RUN_TUNING = False` hoặc giảm epoch để chạy thử, nhưng kết quả sẽ khác các số liệu được báo cáo ở đây.

## 11. Tệp đầu ra

- [`wnut17_character_ner.ipynb`](./wnut17_character_ner.ipynb): mã nguồn, log và kết quả của notebook phiên bản tuned.
- [`wnut17_entity_metrics.csv`](./wnut17_entity_metrics.csv): precision, recall, F1 và support theo mô hình/nhóm.
- [`wnut17_ablation_bootstrap.csv`](./wnut17_ablation_bootstrap.csv): chênh lệch F1 và bootstrap CI.
- [`wnut17_error_examples.csv`](./wnut17_error_examples.csv): ví dụ FN/FP và ngữ cảnh.

Khi `RUN_TUNING = True`, notebook cũng tạo `wnut17_hyperparameter_search.csv` với kết quả các lượt random search trên validation.
