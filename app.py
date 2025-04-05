import gradio as gr

def greet(name):
    return f"Hello {name}!"

with gr.Blocks() as demo:
    gr.Markdown("# Gradio Test Interface")
    with gr.Row():
        name_input = gr.Textbox(label="名前を入力")
        output = gr.Textbox(label="結果")
    greet_btn = gr.Button("実行")
    greet_btn.click(fn=greet, inputs=name_input, outputs=output)

if __name__ == "__main__":
    demo.launch()
