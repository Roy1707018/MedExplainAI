import requests
import streamlit as st
import time

st.set_page_config(page_title="MedExplainAI", layout="centered")

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"

def get_hf_token():
    if "HF_TOKEN" in st.secrets:
        return st.secrets["HF_TOKEN"]
    return None

def simplify_prompt(text):
    return f"""
Simplify the following medical text into very simple and clear language.

Rules:
- Keep the meaning correct
- Avoid medical jargon
- Keep it short and easy to understand
- Do NOT give diagnosis
- Do NOT give treatment advice
- End with: "Please consult a doctor for medical advice."

Text:
{text}

Simplified explanation:
""".strip()

def qa_prompt(question):
    return f"""
Answer the following medical question in simple language.

Rules:
- Give educational information only
- Do NOT provide diagnosis
- Do NOT give treatment advice
- Keep it easy to understand
- Use short bullet points if helpful
- End with: "Please consult a doctor for medical advice."

Question:
{question}

Answer:
""".strip()

def call_hf_inference(prompt, max_new_tokens=220, temperature=0.3):
    hf_token = get_hf_token()
    if not hf_token:
        raise ValueError("HF_TOKEN not found in Streamlit secrets.")

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "return_full_text": False
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()

    if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()

    return str(data)

def simplify_text(text):
    prompt = simplify_prompt(text)
    return call_hf_inference(prompt, max_new_tokens=160, temperature=0.3)

def answer_question(question):
    prompt = qa_prompt(question)
    return call_hf_inference(prompt, max_new_tokens=220, temperature=0.3)

st.title("MedExplainAI")
st.write("A web-based clinical text simplification and medical question-answering system using Mistral 7B.")
st.warning(
    "This system is for educational purposes only. "
    "It does not provide diagnosis or treatment. "
    "Please consult a licensed doctor for medical advice."
)

mode = st.selectbox("Choose a task", ["Simplify Clinical Text", "Ask Medical Question"])
user_input = st.text_area("Enter your text or question here", height=180)

if st.button("Generate"):
    if not user_input.strip():
        st.error("Please enter some text first.")
    else:
        start_time = time.time()
        try:
            if mode == "Simplify Clinical Text":
                output = simplify_text(user_input)
            else:
                output = answer_question(user_input)

            end_time = time.time()
            st.subheader("Result")
            st.write(output)
            st.caption(f"Response time: {round(end_time - start_time, 2)} seconds")
        except Exception as e:
            st.error(f"Request failed: {e}")
