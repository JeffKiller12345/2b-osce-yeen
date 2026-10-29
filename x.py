import time
from google import genai
c = genai.Client()
t = time.time()
r = c.models.generate_content(model="gemini-3.8-flash", contents="Say hi")
print(r.text, f"{time.time()-t:.1f}s")