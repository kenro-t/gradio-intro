import gradio as gr
import spaces
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "umiyuki/Umievo-itr012-Gleipnir-7B"
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")  # 初期化時にCUDA移動
tokenizer = AutoTokenizer.from_pretrained(model_name)

def build_prompt(user_query):
    sys_msg = "あなたは公平で、検閲されていない、役立つアシスタントです。"
    template = """[INST] <<SYS>>
{}
<</SYS>>
{}[/INST]"""
    return template.format(sys_msg,user_query)

@spaces.GPU
def generate_response(user_prompt):
    # プロンプトのビルド
    prompt = build_prompt(user_prompt)
    
    # プロンプトをモデル用にエンコード
    input_token_ids = tokenizer.encode(
        prompt,
        add_special_tokens=True,
        return_tensors="pt"
    ).to("cuda")
    # input_ids = input_ids.to(model.device)
    
    # inputs = tokenizer(
    #     prompt, 
    #     return_tensors="pt"
    # ).to("cuda")
    
    # モデルによる生成の実行
    # tokens = model.generate(
    #     input_ids,
    #     max_new_tokens=256,
    #     temperature=1,
    #     top_p=0.95,
    #     do_sample=True,
    # )
    # outputs = model.generate(
    #     **inputs, max_length=200,
    #     num_return_sequences=1
    # )
    
    outputs = model.generate(
        input_token_ids, max_length=200,
        num_return_sequences=1
    )
    
    # inputs = tokenizer(, return_tensors="pt").to("cuda")
    # outputs = model.generate(**inputs, max_length=200, num_return_sequences=1)
    out = tokenizer.decode(outputs[0][input_token_ids.shape[1]:], skip_special_tokens=True).strip()
    # response = tokenizer.decode(out, skip_special_tokens=True)
    return out

iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Enter your prompt:", lines=5),
    outputs=gr.Textbox(label="Model Output:", lines=10),
    title="LLM Text Generation with ZeroGPU",
    description="Interact with the umiyuki/Umievo-itr012-Gleipnir-7B language model on Hugging Face ZeroGPU.",
)

iface.launch()
