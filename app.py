# DeepTube IQ - YouTube Video Question Answering App

import gradio as gr
from utils.yt_processor import create_db_from_youtube_video_url, get_response_from_query

# Global DB instance
db = None

# Define function for Gradio

def process_video(video_url, question):
    global db
    try:
        if db is None:
            db = create_db_from_youtube_video_url(video_url)
        response, _ = get_response_from_query(db, question)
        return response
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Gradio UI
description = """
## 🤖 DeepTube IQ
Ask intelligent questions about any YouTube video using the power of LLMs & AI.
"""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(description)
    with gr.Row():
        video_url = gr.Textbox(label="YouTube Video URL", placeholder="Enter full YouTube link")
    with gr.Row():
        question = gr.Textbox(label="Ask a Question", placeholder="e.g. What is this video about?")
    with gr.Row():
        answer = gr.Textbox(label="Answer", lines=5)
    with gr.Row():
        submit_btn = gr.Button("Get Answer 🧠")
        submit_btn.click(fn=process_video, inputs=[video_url, question], outputs=answer)

# Launch the app
demo.launch()
