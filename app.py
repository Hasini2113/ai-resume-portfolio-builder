import gradio as gr
from transformers import pipeline

# Load model
generator = pipeline(
    "text-generation",
    model="gpt2-medium"
)



# Resume generation function
def generate_resume(name, education, skills, projects, role):
    prompt = f"""
Resume for {role}

Name: {name}

Professional Summary:
Motivated {education} student with strong skills in {skills}, seeking an entry-level {role} position.

Skills:
{skills}

Projects:
{projects}

Education:
{education}

--- END OF RESUME ---
"""

    output = generator(
        prompt,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.5,
        eos_token_id=generator.tokenizer.eos_token_id
    )

    text = output[0]["generated_text"]

    # 🔴 HARD STOP: cut anything after END OF RESUME
    text = text.split("--- END OF RESUME ---")[0]

    return text.strip()




# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 AI Resume Builder")

    name = gr.Textbox(label="Name")
    education = gr.Textbox(label="Education")
    skills = gr.Textbox(label="Skills")
    projects = gr.Textbox(label="Projects")
    role = gr.Textbox(label="Target Job Role")

    btn = gr.Button("Generate Resume")
    output = gr.Textbox(lines=15, label="Generated Resume")

    btn.click(
        generate_resume,
        inputs=[name, education, skills, projects, role],
        outputs=output
    )

demo.launch(ssr_mode=False)

