# ai_engine/nvidia_client.py
import os
import requests
from flask import session, has_request_context

# Use environment variable for NVIDIA API key; no hardcoded fallback.
DEFAULT_NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY')

def call_llm(system_instruction: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.2, **kwargs) -> str:
    """
    Invoke configured LLM (Nvidia Llama 3.1) using the session or environment keys.
    """
    nvidia_key = None
    
    if has_request_context():
        nvidia_key = session.get('nvidia_api_key')
        
    if not nvidia_key:
        nvidia_key = DEFAULT_NVIDIA_API_KEY

    # Invoke Nvidia API
    if nvidia_key:
        try:
            url = "https://integrate.api.nvidia.com/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {nvidia_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "meta/llama-3.1-8b-instruct",
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            res = requests.post(url, headers=headers, json=payload, timeout=30)
            if res.status_code == 200:
                res_json = res.json()
                return res_json['choices'][0]['message']['content']
            else:
                print(f"Nvidia API returned code {res.status_code}: {res.text}")
                return f"⚠️ Nvidia API error (Code {res.status_code})"
        except Exception as e:
            print(f"Exception calling Nvidia API in call_llm: {e}")
            return f"⚠️ Error calling Nvidia API: {e}"

    return "⚠️ No configured LLM API Key (Nvidia) was found."
