import gradio as gr
from backend import process_song

def run_pipeline(audio_file, target_lang):
    if audio_file is None:
        return None, None, "Please upload an audio file.", ""

    print(f"Processing {audio_file} to {target_lang}")
    final_mix, vocals, original_lyrics, translated_lyrics = process_song(audio_file, target_lang)

    return final_mix, vocals, original_lyrics, translated_lyrics

# Define the interface
with gr.Blocks(title="Song Translator PoC") as demo:
    gr.Markdown("# 🎵 AI Song Translator")
    gr.Markdown("Upload a song, choose a language, and get a translated version (spoken/sung)!")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Upload Song", type="filepath")
            target_lang = gr.Dropdown(
                label="Target Language",
                choices=["es", "fr", "de", "it", "pt", "nl", "ru", "ja", "ko", "zh", "en"],
                value="es"
            )
            submit_btn = gr.Button("Translate & Sing", variant="primary")

        with gr.Column():
            final_output = gr.Audio(label="Final Translated Mix")
            vocals_output = gr.Audio(label="Extracted Vocals")

    with gr.Row():
        original_text = gr.Textbox(label="Original Lyrics", lines=10)
        translated_text = gr.Textbox(label="Translated Lyrics", lines=10)

    submit_btn.click(
        fn=run_pipeline,
        inputs=[audio_input, target_lang],
        outputs=[final_output, vocals_output, original_text, translated_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
