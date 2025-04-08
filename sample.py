import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# モデル名を設定
model_name = "モデル名" # 例: "google/gemma-7b"

# トークナイザーとモデルのロード（@spaces.GPUデコレータの内側で行う）
tokenizer = None
model = None

@gr.spaces.GPU
def generate_response(prompt):
    global tokenizer
    global model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda") # GPUにモデルを移動

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda") # 入力もGPUに移動
    output = model.generate(input_ids, max_length=200, num_return_sequences=1)
    response = tokenizer.decode(output, skip_special_tokens=True)
    return response

iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Prompt"),
    outputs=gr.Textbox(label="Response")
)

iface.launch()

# gradio>=4.0
# transformers
# torch