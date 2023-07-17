# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXModel
from llama import Custom_LlamaModel

tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")
model = GPTNeoXModel.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")


prompt = "### 질문: 안녕, 너는 누구니?\n\n### 답변: "
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
embedding = model(inputs.input_ids)['last_hidden_state']
print(embedding)
print(embedding.shape)


# answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# print(answer)
# # "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."


# # get_input_embeddings