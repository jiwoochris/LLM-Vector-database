import openai

# 보안을 위해 api_key.txt는 따로 보관
api_key_file = "api_key.txt"

with open(api_key_file, 'r') as file:
    api_token = file.read()
    
    openai.api_key = api_token
            

LLM_prompt =  """
너는 법률 자문을 위한 챗봇이야. 사용자의 질문에 대한 답변을 해 줘야해.

사용자의 질문 : "몸이 아파서 회사에 안나갔는데, 무단결근이라고 해고당했습니다. 이게 맞나요?"

아래 문서를 바탕으로 사용자의 질문에 대답해줘.

문서 : 
"
질문 : 몸이 아파 일주일 동안 회사에 출근하지 않았는데, 회사에서 무단결근을 이유로 해고하였습니다. 정당한 해고 인가요?
답변 : 취업규칙이나 단체협약에서 무단결근을 해고사유나 징계사유로 정하고 있는 경우에 이러한 취업규칙이나 단체협약은 특별한 사정이 없는 한 무효라고 할 수 없으므로 근로자를 무단결근을 이유로 해고하는 것은 정당한 해고에 해당합니다.
"
"""

LLM_prompt = "몸이 아파서 회사에 안나갔는데, 무단결근이라고 해고당했습니다. 이게 맞나요?"

response = openai.Completion.create(
    model="text-davinci-003",
    prompt=LLM_prompt,
    max_tokens=516,
    temperature=0.7,
    top_p=0.9,
    n=1,
    stream=False,
    logprobs=None
)


print(response)
print("->")
print(response.choices[0].text.strip())

answer1 = response.choices[0].text.strip()