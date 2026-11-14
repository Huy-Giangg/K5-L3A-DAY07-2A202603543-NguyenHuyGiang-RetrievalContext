# Báo cáo Thực nghiệm Day 7: Agentic RAG

- **Học viên:** Nguyễn Huy Giang
- **MSSV:** 2A202603543
- **Lớp:** L3A
- **Chủ đề nghiên cứu:** Báo cáo Nợ Quốc tế (*World Bank - International Debt Report 2026*)
- **Repository:** `K5-L3A-DAY07-2A202603543-NguyenHuyGiang-RetrievalContext`

---

## 1. Dữ liệu và benchmark

- **Nguồn tài liệu, bản quyền, số trang và số chunk:**
  - *Nguồn:* Tài liệu chính thức *International Debt Report 2026* do Ngân hàng Thế giới (World Bank) công bố.
  - *Bản quyền:* Creative Commons Attribution CC BY 3.0 IGO.
  - *Phạm vi xử lý:* 35 trang trọng tâm (Trang 21–55, bao gồm phần tổng quan vĩ mô *Key Takeaways* và phân tích chi tiết *Chapter 1*).
  - *Số lượng chunk:* **121 chunks** được kế thừa từ Cấu hình B tối ưu của Day 6 (chiến lược Semantic Paragraph Chunking, trung bình 742 ký tự/chunk, giữ trọn vẹn ngữ cảnh phân tích kinh tế).
- **Vector store, embedding model, phiên bản và số chiều:**
  - *Embedding model:* `sentence-transformers/all-MiniLM-L6-v2` chạy local trên CPU, kích thước vector **384 chiều**, chuẩn hóa $L_2$ (`normalize_embeddings=True`).
  - *Vector store:* `Qdrant` (chế độ persistent storage local tại `./qdrant_db`) kết hợp ma trận vector lưu trữ sẵn `data/vectors.npy` (121, 384) và `data/chunks.json`.
  - *Độ đo khoảng cách:* Cosine Similarity (tính nhanh qua tích vô hướng ma trận sau khi đã chuẩn hóa $L_2$).
- **Agent model, cách chạy và phân chia dev/test:**
  - *Agent Controller Model:* `qwen/qwen3.8-27b` thông qua Groq Cloud API (suy luận tốc độ cao ~300 tokens/s, tuân thủ nghiêm ngặt JSON schema và định dạng trích dẫn). Có thể chuyển đổi linh hoạt sang `gemini-3.5-flash` qua biến môi trường `PROVIDER` trong `.env`.
  - *Phân chia dữ liệu benchmark:* Tổng cộng **24 câu hỏi** (tuân thủ nghiêm ngặt theo [BENCHMARK.md](../../BENCHMARK.md)):
    - **Tập Dev (17 câu):** 15 câu có đáp án trích xuất từ tài liệu (`answerable: true`) và 2 câu thiếu bằng chứng (`answerable: false`).
    - **Tập Test (7 câu):** 5 câu có đáp án độc lập (`answerable: true`) và 2 câu thiếu bằng chứng (`answerable: false`).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - *Câu có đáp án:* Có `gold_source_id`, `gold_page`, và chuỗi `evidence` khớp 100% nguyên văn trong chunk. Tiêu chí chấm:
    - **Điểm 2:** Đúng nội dung, đầy đủ ý và có trích dẫn `[chunk-id]` hợp lệ nằm trong context.
    - **Điểm 1:** Đúng một phần hoặc trích dẫn chưa hoàn toàn chính xác.
    - **Điểm 0:** Trả lời sai, bị ảo giác hoặc từ chối nhầm (`false_abstention`).
  - *Câu thiếu bằng chứng (Unanswerable):* Các câu hỏi mang tính đánh lừa gần chủ đề (ví dụ: số liệu nợ của Việt Nam năm 2025, quyết định lãi suất của Fed tháng 12/2025, nợ của Nhật Bản, hoặc viện trợ EU năm 2026). Được chấm `abstention_correct = true` nếu Agent trung thực từ chối trả lời, và `false` nếu Agent bịa đặt số liệu.

---

## 2. Retrieval và thử nghiệm

### Bảng so sánh các phương pháp Retrieval trên tập Dev (15 câu có đáp án)

| Cấu hình | Phương pháp, top-k, filter, context budget | Giả thuyết / Mục tiêu thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian (median) | Chi phí |
|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Method 1)** | Dense Retrieval (`all-MiniLM-L6-v2`), top-k=3, budget=800 | Thiết lập mốc cơ sở kế thừa từ Day 6 | dev | 11/15 (73.3%) | 12/15 (80.0%) | 7/15 (46.7%) | 15.35 ms | $0.00 |
| **Phương pháp 2** | BM25 Lexical (`rank-bm25`), top-k=3, budget=800 | Bắt chính xác tên riêng, mã nước và con số kinh tế | dev | 12/15 (80.0%) | **15/15 (100%)** | 9/15 (60.0%) | **0.78 ms** | $0.00 |
| **Phương pháp 3** | Hybrid Retrieval (RRF $k=60$), top-k=3, budget=800 | Dung hòa ngữ nghĩa trừu tượng và từ khóa cứng | dev | **13/15 (86.7%)** | 14/15 (93.3%) | 9/15 (60.0%) | 15.45 ms | $0.00 |
| **Cải thiện 1** | Hybrid RRF, top-k=4, budget=1200 | Tăng budget lên 1200 để giảm rớt chunk do tràn ký tự | dev | 13/15 (86.7%) | 14/15 (93.3%) | 12/15 (80.0%) | 15.60 ms | $0.00 |
| **Cải thiện 2 (Đã chọn)** | Hybrid RRF, top-k=4, budget=1500 | Tăng budget lên 1500 để bao trọn các đoạn văn dài | dev | **13/15 (86.7%)** | **14/15 (93.3%)** | **14/15 (93.3%)** | 15.80 ms | $0.00 |
| **Baseline trên Test** | Dense Retrieval, top-k=3, budget=800 | Đo đối chứng trên tập Test chưa từng học | test | 4/5 (80.0%) | 4/5 (80.0%) | 3/5 (60.0%) | 15.10 ms | $0.00 |
| **Cấu hình chọn trên Test** | Hybrid RRF, top-k=4, budget=1500 | Đánh giá độc lập cấu hình tối ưu trên Test | test | **5/5 (100%)** | **5/5 (100%)** | **5/5 (100%)** | 15.75 ms | $0.00 |

### Phân tích hai lỗi trên Dev, nguyên nhân và lý do chọn cấu hình:
1. **Lỗi 1 (Nút thắt Context Tràn Ký tự ở câu q01 và q09):**
   - *Hiện tượng:* Ở câu hỏi `q01` và `q09`, retriever đã đưa đoạn văn chứa đáp án vào Top-3, tuy nhiên `Context Evidence Rate` bị rớt xuống chỉ còn 46.7% - 60%.
   - *Nguyên nhân:* Hàm `pack_context` với ngưỡng mặc định `budget=800` ký tự đã loại bỏ chunk thứ 2 và thứ 3 khi chunk đầu tiên đã dài xấp xỉ 600–700 ký tự.
   - *Cách thử và khắc phục:* Tăng dần context budget từ 800 lên 1200 rồi 1500 ký tự. Tại mức 1500 ký tự, `Context Evidence Rate` tăng vọt từ **46.7% lên 93.3%**.
2. **Lỗi 2 (Bắt sót thực thể tên riêng khi chỉ dùng Dense đơn thuần ở câu q05):**
   - *Hiện tượng:* Với câu hỏi *"Which borrower led the increase in borrowing levels in 2025 among MICs other than China?"*, mô hình Dense đơn thuần có xu hướng xếp các đoạn chung về MICs lên trước đoạn chứa từ khóa *"Ukraine"*.
   - *Nguyên nhân:* Dense embedding nắm ngữ nghĩa tổng thể nhưng trọng số chú ý cho từ khóa đơn lẻ thấp hơn mô hình từ khóa.
   - *Cách thử và khắc phục:* Áp dụng **BM25 kết hợp RRF (Hybrid)**. BM25 cho điểm số vượt trội với token *"Ukraine"*, kéo chunk này lên vị trí Top-1 ngay lập tức.
- **Lý do chốt cấu hình:** Cấu hình **Hybrid RRF + Budget 1500 + Top-4** đạt sự cân bằng hoàn hảo: Hit@1 cao nhất (86.7%), Context Evidence Rate đạt 93.3%, đồng thời độ trễ chỉ ~15ms và chi phí retrieval bằng $0.00.

---

## 3. Agent và đánh giá

- **Kiến trúc Agent:**
  - *Tool điều phối:* Tool `search(query: str, filters: dict | None = None, top_k: int = 4)` trả về danh sách các chunk tương thích kèm điểm số và metadata.
  - *Số lượt gọi & Điều kiện dừng:* Tối đa `max_searches = 3`. Nếu sau 3 lượt tìm kiếm vẫn không đủ bằng chứng, Agent tự động kích hoạt điều kiện dừng an toàn và từ chối trả lời (`status: "abstain"`).
  - *Cách chọn Context & Trích dẫn:* Context được nén và kiểm tra trùng lặp qua hàm `pack_context` với ngân sách 1500 ký tự. Agent bắt buộc phải gắn trích dẫn `[chunk-id]` ngay tại câu văn chứa khẳng định và liệt kê danh sách tương ứng trong trường `citations`.

### Bảng kết quả tổng hợp của Agent (Chạy với `starter/agent_metrics.py`)

| Cấu hình / Split | Tool use | Số lượt gọi (tb) | Grounded answer (0 / 1 / 2) và Trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm (`false_abstain`) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dev - Baseline (Dense, 800, top-3)** | 17/17 (100%) | 1.8 lượt | 9 / 0 / 6 — **Mean: 0.80 / 2.0** | 6/6 (100%) | 6/6 (100%) | **2/2 (100%)** | 9/15 (60.0%) |
| **Dev - Cải tiến (Hybrid, 1500, top-4)** | 17/17 (100%) | 1.6 lượt | 0 / 1 / 14 — **Mean: 1.93 / 2.0** | 14/16 (87.5%) | 15/15 (100%) | **2/2 (100%)** | **0/15 (0.0%)** |
| **Test - Baseline (Dense, 800, top-3)** | 7/7 (100%) | 1.7 lượt | 2 / 0 / 3 — **Mean: 1.20 / 2.0** | 3/3 (100%) | 3/3 (100%) | **2/2 (100%)** | 2/5 (40.0%) |
| **Test - Cấu hình chọn (Hybrid, 1500, top-4)** | **7/7 (100%)** | **1.4 lượt** | 0 / 0 / 5 — **Mean: 2.00 / 2.0** | **5/5 (100%)** | **5/5 (100%)** | **2/2 (100%)** | **0/5 (0.0%)** |

### Trích xuất các ca điển hình từ Trace:

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và Citation | Nhãn & Lý do |
|---|---|---|---|---|
| **q02** | Có đáp án (Dev) | `search("total external borrowing level of LMICs 2025")` -> 4 chunks (Top-1: `worldbank-idr-2026_p27_para_001`) | *"The total external borrowing level of LMICs reached a record $9.4 trillion at the end of 2025 [worldbank-idr-2026_p27_para_001]."* | **Điểm 2**: Đáp án chính xác tuyệt đối, trích dẫn đúng ID và số liệu. |
| **q16** | Có đáp án (Test) | `search("cumulative external debt stock of MICs 2025 excluding China")` -> 4 chunks (Top-1: `worldbank-idr-2026_p31_para_001`) | *"The total dollar value of the external debt stock of MICs in 2025, excluding China, was $6.9 trillion [worldbank-idr-2026_p31_para_001]."* | **Điểm 2**: Trả lời đúng $6.9 trillion kèm citation chuẩn xác. |
| **q_unans_dev_01** | Thiếu bằng chứng (Dev) | `search("Vietnam external debt stock 2025")` -> 4 chunks không có số liệu cụ thể của Việt Nam | *"Không đủ bằng chứng để trả lời."* `citations: []` | **Abstention đúng (True)**: Agent không bịa số liệu, trung thực từ chối. |
| **q_unans_test_01** | Thiếu bằng chứng (Test) | `search("total external debt Japan 2025")` -> 4 chunks không có Nhật Bản (vì là nước phát triển) | *"Không đủ bằng chứng để trả lời."* `citations: []` | **Abstention đúng (True)**: Nhận diện chính xác câu hỏi nằm ngoài phạm vi tài liệu. |

- **Đường dẫn artifact thực nghiệm:**
  - Kết quả Retrieval: [`results/retrieval_comparison.json`](../../results/retrieval_comparison.json)
  - Trace Dev Baseline: [`results/agent-dev-baseline/traces.json`](../../results/agent-dev-baseline/traces.json)
  - Trace Dev Cải tiến: [`results/agent-dev-improved/traces.json`](../../results/agent-dev-improved/traces.json)
  - Trace Test Cấu hình chọn: [`results/agent-test-improved/traces.json`](../../results/agent-test-improved/traces.json)
- **Lệnh tái hiện thực nghiệm:**
  ```bash
  # 1. Chạy so sánh 3 phương pháp Retrieval trên tập Dev:
  python src/benchmark_retrievers.py

  # 2. Chạy Agent Baseline trên Dev:
  python src/run_agent_benchmark.py --split dev --retriever dense --budget 800 --top-k 3 --output-dir results/agent-dev-baseline

  # 3. Chạy Agent Cấu hình Cải tiến trên Dev:
  python src/run_agent_benchmark.py --split dev --retriever hybrid --budget 1500 --top-k 4 --output-dir results/agent-dev-improved

  # 4. Đánh giá kiểm thử Cấu hình chốt trên Test:
  python src/run_agent_benchmark.py --split test --retriever hybrid --budget 1500 --top-k 4 --output-dir results/agent-test-improved
  ```

---

## 4. Kết luận và bài học kinh nghiệm

1. **Hiệu quả của Hybrid Retrieval trong phân tích tài chính/kinh tế:**
   - Việc chỉ dựa vào Dense Embedding dễ làm trôi các mã nước viết tắt (IDA, LMIC) hoặc tên riêng quốc gia (Ukraine). Kết hợp Dense + BM25 bằng Reciprocal Rank Fusion (RRF) giúp nâng Hit@1 từ **73.3% lên 86.7%**, đưa đúng đoạn văn chứa số liệu lên vị trí ưu tiên số 1.
2. **Tác động quyết định của Context Budget:**
   - Ban đầu, Agent có tỷ lệ từ chối nhầm (`false_abstention`) cao tới 60% không phải do model kém mà do giới hạn `budget=800` ký tự làm rơi rụng bằng chứng. Khi điều chỉnh ngân sách lên 1500 ký tự, điểm số trung bình vọt từ **0.80 lên 1.93 / 2.0**, và trên tập Test đạt điểm tối đa **2.00 / 2.0 (100%)**.
3. **Tính trung thực và chống ảo giác (Hallucination Prevention):**
   - Nhờ ràng buộc chặt chẽ trong System Prompt và Controller Protocol, Agent đạt tỷ lệ **100% từ chối đúng đối với các câu hỏi ngoài phạm vi** (không hề bịa đặt số liệu nợ của Việt Nam hay Nhật Bản).
