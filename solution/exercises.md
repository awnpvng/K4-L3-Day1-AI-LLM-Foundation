# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Ở temperature 0.0–1.0, các phản hồi vẫn tập trung vào cùng chủ đề và khá mạch lạc, nhưng cách diễn đạt, ví dụ và mức độ chi tiết có thể khác nhau. Khi tăng lên 1.5, phản hồi thường đa dạng và khó đoán hơn, đồng thời có thể xuất hiện nội dung lan man hoặc kém mạch lạc. Nhìn chung, temperature càng cao thì tính sáng tạo tăng nhưng độ ổn định và khả năng kiểm soát đầu ra giảm.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Temperature thấp, khoảng 0.0–0.3 với chatbot hỗ trợ khách hàng vì trả lời cần sự nhất quán, đúng chính sách/thông tin sản phẩm và hạn chế tối đa việc "bịa thông tin" (hallucination).
> Temperature thấp giúp model ưu tiên các từ có xác suất cao nhất, giảm biến thiên giữa các lần trả lời cho cùng một câu hỏi — quan trọng khi nhiều khách hàng hỏi cùng một vấn đề và kỳ vọng nhận được câu trả lời giống nhau, đáng tin cậy.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Với 10.500.000 token đầu ra/ngày, GPT-4o tốn khoảng $105.00/ngày (≈$3,150/tháng) trong khi GPT-4o-mini chỉ tốn khoảng $6.30/ngày (≈$189/tháng) — GPT-4o đắt hơn mini khoảng 16.7 lần cho cùng workload này (tính theo bảng giá output trong `template.py`).GPT-4o xứng đáng khi tác vụ đòi hỏi suy luận phức tạp, độ chính xác cao, ảnh hưởng trực tiếp đến quyết định quan trọng — ví dụ tư vấn pháp lý/tài chính, sinh code phức tạp, hoặc phân tích dữ liệu nhiều bước. Ngược lại, mini phù hợp cho các tác vụ đơn giản, khối lượng lớn — ví dụ phân loại ý định (intent classification), trả lời FAQ, tóm tắt ngắn — nơi tốc độ và chi phí quan trọng hơn độ sâu suy luận.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Phản hồi theo persona giáo viên tiểu học dùng câu ngắn, từ vựng đơn giản và ví dụ đời thường, chẳng hạn ví blockchain như một cuốn sổ mà nhiều người cùng giữ một bản giống nhau. Phản hồi theo persona chuyên gia tài chính dùng các thuật ngữ chuyên môn như "sổ cái phân tán (DLT)" và "mạng ngang hàng (P2P)", đồng thời trình bày sâu hơn. Vì vậy, system prompt định hình vai trò, giọng điệu, mức độ chuyên môn và cách lựa chọn ví dụ của model, dù câu hỏi người dùng không thay đổi.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn 113 từ, ước lượng "số từ / 0.75" cho ra khoảng 151 token, còn `tiktoken` của GPT-4o đếm được 150 token. Chênh lệch là khoảng 0,66%, gần như trùng khớp trong đoạn văn cụ thể này. Tuy vậy, tiếng Việt thường tốn nhiều token hơn tiếng Anh cùng độ dài vì các từ và ký tự có dấu thường bị bộ mã hóa tách thành nhiều token con; công thức "số từ / 0.75" được ước lượng chủ yếu theo tiếng Anh nên chỉ mang tính tham khảo.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi có người dùng đang chờ trực tiếp trên giao diện tương tác (chatbot, trợ lý coding) và câu trả lời khá dài: hiển thị từng phần ngay khi model sinh ra giúp giảm cảm giác chờ đợi (perceived latency), người dùng có thể bắt đầu đọc trong khi phần còn lại vẫn đang sinh. Ngược lại, non-streaming phù hợp hơn khi kết quả cần được xử lý như một khối hoàn chỉnh trước khi dùng — ví dụ output có cấu trúc JSON để parse, function calling, hoặc các tác vụ chạy nền/batch không có người dùng theo dõi trực tiếp — vì lúc đó việc hiển thị từng chunk không mang lại lợi ích UX nào mà còn làm code xử lý phức tạp hơn.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff tăng dần thời gian chờ sau mỗi lần thất bại, giúp giảm áp lực lên server đang quá tải một cách từ từ thay vì gây thêm một đợt request dồn dập ngay khi vừa lỗi. Nếu dùng delay cố định giống nhau cho hàng nghìn client, tất cả sẽ đồng loạt retry cùng lúc sau đúng 1 giây — tạo ra hiện tượng "thundering herd": server vừa mới quá tải lại ngay lập tức nhận thêm một làn sóng request đồng thời, dễ tiếp tục lỗi hoặc sập hẳn, và chu kỳ lỗi-retry-lỗi cứ lặp lại. Backoff theo cấp số nhân (thường kết hợp thêm jitter — độ trễ ngẫu nhiên) giúp giãn request ra theo thời gian, cho server cơ hội phục hồi thực sự.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Persona: *"Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt."* Từ "ngắn gọn" được đưa vào có chủ đích: vì `history` chỉ giữ 3 lượt hội thoại gần nhất (6 message), câu trả lời càng dài càng nhanh chiếm hết ngân sách token của history/context và làm tăng chi phí mỗi lượt gọi API — yêu cầu ngắn gọn giúp trợ lý tập trung vào ý chính. Việc chỉ định rõ "bằng tiếng Việt" đảm bảo tính nhất quán ngôn ngữ: nếu không chỉ định, model có thể tự chuyển sang tiếng Anh giữa chừng khi gặp thuật ngữ kỹ thuật, gây trải nghiệm khó chịu cho học viên đang hỏi bằng tiếng Việt.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là `history` chỉ giữ 3 lượt gần nhất (`history[-6:]`) và không có bộ nhớ dài hạn giữa các phiên chat — hễ đóng chương trình là mất hết ngữ cảnh, và ngay trong một phiên dài, các chi tiết từ đầu buổi cũng bị "quên" khi vượt quá 3 lượt. Cải thiện đề xuất: thêm một bước tóm tắt định kỳ (summarization) — mỗi khi history sắp vượt quá giới hạn, gọi model tóm tắt các lượt cũ thành một đoạn ngắn 2-3 câu, lưu đoạn tóm tắt đó vào đầu system prompt (hoặc một biến `summary` riêng) thay vì xoá hẳn. Cách này giữ được ngữ cảnh quan trọng xuyên suốt phiên chat dài mà không làm history phình to, đồng thời có thể lưu `summary` này ra file/DB theo từng người dùng để phục hồi ngữ cảnh ở phiên chat lần sau.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026