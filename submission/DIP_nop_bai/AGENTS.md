# Project Behavioral Directives & Standards

## Mandatory Skill Engagement (`.agents/skills/`)
Đối với mọi tác vụ hoặc prompt từ người dùng trong workspace này, AI Assistant **BẮT BUỘC** phải chủ động kiểm tra, kích hoạt và tuân thủ nghiêm ngặt các kỹ năng tương ứng trong `.agents/skills/` trước khi lên kế hoạch hoặc sinh mã:

### 1. Quy tắc sinh mã đầy đủ (Full Code Generation)
- Luôn kích hoạt và tuân thủ `full-output-enforcement`:
  - **TUYỆT ĐỐI KHÔNG** dùng placeholder (ví dụ: `// TODO`, `/* keep existing code */`, `...`).
  - Phải sinh code hoàn chỉnh, đầy đủ ngữ cảnh và sẵn sàng chạy được.

### 2. Tiêu chuẩn Giao diện & Thiết kế (Frontend, UI/UX & Motion)
- Khi thực hiện bất kỳ công việc nào liên quan đến Web / Mobile / Dashboard / UI:
  - Tham khảo và áp dụng các skill thiết kế cao cấp: `design-taste-frontend`, `gpt-taste`, `high-end-visual-design`, `minimalist-ui`, hoặc `industrial-brutalist-ui` tùy thuộc vào ngữ cảnh sản phẩm.
  - Chống mọi khuôn mẫu "AI slop" (gradient tím nhạt generic, card lồng card vô tội vạ, typography thô thiển, padding cẩu thả).
  - Với các dự án nâng cấp / sửa giao diện có sẵn, luôn áp dụng `redesign-existing-projects` để audit trước khi sửa, tránh đứt gãy tính năng.

### 3. Thiết kế Thị giác & Ảnh minh họa (Visual & Image Assets)
- Khi cần tạo concept, brand identity, layout mockup hoặc sinh ảnh UI:
  - Áp dụng `imagegen-frontend-web`, `imagegen-frontend-mobile`, `image-to-code`, `brandkit`.
  - Tuân thủ quy tắc 1 section / 1 ảnh rõ ràng, không ghép nén nhiều section vào một tấm ảnh gây giảm chất lượng.

### 4. Quy trình thực hiện với Skills
1. **Tra cứu trước khi làm**: Trước khi trả lời hoặc thực thi prompt, xác định ngay skill nào trong `.agents/skills/` (hoặc các global skills) liên quan trực tiếp đến tác vụ.
2. **Đọc `SKILL.md`**: Đọc tệp hướng dẫn `SKILL.md` của skill đó để nạp đầy đủ quy tắc, checklists và anti-patterns.
3. **Tuân thủ toàn bộ**: Thực hiện tác vụ bám sát 100% hướng dẫn của skill đã kích hoạt.
