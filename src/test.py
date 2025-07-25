def create_prompt(flag, context, question):
    if flag == 1:
        prompt_template = f"""<|im_start|>system
Bạn là một trợ lý AI hữu ích tên là TUNGLLAMA. Hãy chỉ dựa vào thông tin trong ngữ cảnh bên dưới để trả lời câu hỏi một cách chính xác, ngắn gọn và không lặp lại. Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy trả lời "Tôi không tìm thấy thông tin trong ngữ cảnh".
<|im_end|>
<|im_start|>user
Ngữ cảnh:
{context}

Câu hỏi:
{question}
<|im_end|>
<|im_start|>assistant
"""
        return prompt_template
    elif flag == 0:
        prompt_template = f"""<|im_start|>system
Bạn là một trợ lí AI tên là TungLlama. Hãy trả lời người dùng một cách chính xác.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
        return prompt_template

print(create_prompt(1, "tung ten to", "gioi"))