# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaModel
from llama import Custom_LlamaModel

tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca")
model = LlamaModel.from_pretrained("beomi/KoAlpaca")


prompt = "### 질문: 안녕, 너는 누구니?\n\n### 답변: "
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model(inputs.input_ids)
print(generate_ids['last_hidden_state'])


# answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# print(answer)
# # "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."


# # get_input_embeddings