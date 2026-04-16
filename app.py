import time
import requests
import streamlit as st

st.set_page_config(page_title="MedExplainAI", layout="centered")

API_URL = "https://router.huggingface.co/v1/chat/completions"

# Safer deployable choice because provider support is visible on the model page
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai"

def get_hf_token():
    if "HF_TOKEN" in st.secrets:
        return st.secrets["HF_TOKEN"]
    return None

def simplify_messages(text):
    return [
        {
            "role": "system",
            "content": (
                "You are a medical text simplification assistant. "
                "Rewrite clinical text in simple, patient-friendly language. "
                "Do not diagnose. Do not give treatment advice. "
                "Keep the meaning correct and the wording easy to understand. "
                "End with: Please consult a doctor for medical advice."
            ),
        },
        {
            "role": "user",
            "content": f"Simplify this medical text:\n\n{text}"
        },
    ]

def qa_messages(question):
    return [
        {
            "role": "system",
            "content": (
                "You are a medical educational assistant. "
                "Answer in simple language. "
                "Give educational information only. "
                "Do not diagnose. Do not give treatment instructions. "
                "End with: Please consult a doctor for medical advice."
            ),
        },
        {
            "role": "user",
            "content": question
        },
    ]

def call_hf_chat(messages, max_tokens=220, temperature=0.3):
    hf_token = get_hf_token()
    if not hf_token:
        raise ValueError("HF_TOKEN not found in Streamlit secrets.")

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)

    # Helpful debugging if HF returns a 4xx/5xx
    if not response.ok:
        raise RuntimeError(f"{response.status_code} {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()

def simplify_text(text):
    return call_hf_chat(simplify_messages(text), max_tokens=180, temperature=0.2)

def answer_question(question):
    return call_hf_chat(qa_messages(question), max_tokens=240, temperature=0.3)

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
