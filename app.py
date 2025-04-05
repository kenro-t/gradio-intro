# 必要なライブラリのインストール
# !pip install -q transformers accelerate bitsandbytes gradio torch

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def build_prompt(user_query):
    sys_msg = "あなたは公平な役立つアシスタントです。"
    template = f"[INST] <<SYS>>\n{sys_msg}\n<</SYS>>\n\n{user_query}[/INST]"
    return template

def generate_response(user_input, tokenizer, model):
    try:
        prompt = build_prompt(user_input)

        input_ids = tokenizer.encode(
            prompt,
            add_special_tokens=True,
            return_tensors="pt"
        )

        # Move input_ids to the same device as the model
        input_ids = input_ids.to(model.device)

        tokens = model.generate(
            input_ids,
            max_new_tokens=256,
            temperature=1,
            top_p=0.95,
            do_sample=True,
        )

        out = tokenizer.decode(tokens[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        return out
    except Exception as e:
        print(f"Error in generate_response: {e}")
        return "応答の生成中にエラーが発生しました。"

def model_preparation():
    try:
        tokenizer = AutoTokenizer.from_pretrained("umiyuki/Umievo-itr012-Gleipnir-7B")
        model = AutoModelForCausalLM.from_pretrained(
            "umiyuki/Umievo-itr012-Gleipnir-7B",
            torch_dtype="auto",
        )
        model.eval()

        if torch.cuda.is_available():
            model.to("cuda")

        return tokenizer, model
    except Exception as e:
        print(f"Error in model_preparation: {e}")
        return None, None

def main():
    tokenizer, model = model_preparation()
    if tokenizer is None or model is None:
        print("モデルのロードに失敗しました。")
        return

    app = gr.Interface(
        fn=lambda user_input: generate_response(user_input, tokenizer, model),
        inputs="text",
        outputs="text"
    )
    app.launch(share=True)

if __name__ == "__main__":
    main()
