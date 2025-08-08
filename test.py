from openai import OpenAI
import time

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio")

prompt = """
Hello how are you doing!
"""

prompt_messages = [
  {
    "role": "user",
    "content": prompt}
]

start_time_perf = time.perf_counter()
stream = client.chat.completions.create(
  model="josiefied-qwen3-0.6b-abliterated-v1",
  messages=prompt_messages,
  temperature=0.7,
  top_p=0.95,
  stream=True,
)

result = ""

for chunk in stream:
  data = chunk.choices[0].delta.content
  if data is not None:
   result += data
end_time_perf = time.perf_counter()



print(result)
print(end_time_perf - start_time_perf)