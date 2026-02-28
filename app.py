import gradio as gr
from logpt.inference import LogAnalyzer, QUERY_PROMPTS
import transformers

TASK_MAP = {
    "Summary": "summary",
    "Root Cause Analysis": "root_cause",
    "Action Items": "action_items",
}

DESCRIPTION = """
<div style="text-align: center;">
<h1>LoGPT — AI Log Analyzer</h1>
<p>A fine-tuned GPT-2 model built for secure, on-device log analysis. Your logs never leave your machine, no external API calls, and instant insights completely under your control.</p>
<p>Upload a <code>.log</code> or <code>.txt</code> file (or paste logs), select an analysis type, and get intelligent summaries, root cause analysis, or action items in seconds.</p>
</div>
"""

THEME = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#eaf5f2", c100="#d4ebe5", c200="#a9d7cb", c300="#7ec3b1",
        c400="#74aa9c", c500="#74aa9c", c600="#5e9487", c700="#4a7a6e",
        c800="#376055", c900="#24463c", c950="#162c25",
    ),
    neutral_hue="gray",
    font=gr.themes.GoogleFont("Inter"),
    text_size="md",
).set(
    # force light mode colors
    body_background_fill="#ffffff",
    block_background_fill="#ffffff",
    block_border_color="#e5e7eb",
    panel_background_fill="#ffffff",
    background_fill_primary="#ffffff",
    background_fill_secondary="#f3f4f6",
    input_background_fill="#f3f4f6",
    # text
    body_text_color="#1f2937",
    block_label_text_color="#374151",
    block_title_text_color="#1f2937",
    # buttons
    button_primary_background_fill="#74aa9c",
    button_primary_background_fill_hover="#5e9487",
    button_primary_text_color="#ffffff",
    button_primary_border_color="#74aa9c",
    button_secondary_background_fill="#74aa9c",
    button_secondary_background_fill_hover="#5e9487",
    button_secondary_text_color="#ffffff",
    button_secondary_border_color="#74aa9c",
)

CUSTOM_CSS = """
/* nav button sizing */
.nav-btn { min-width: 48px !important; max-width: 56px !important; }
/* disabled nav buttons */
.nav-btn:disabled, .nav-btn[disabled] {
    background-color: #d1d5db !important; border-color: #d1d5db !important;
    color: #9ca3af !important; cursor: default !important;
}
/* status centered */
.status-text { text-align: center !important; }
.status-text p { text-align: center !important; }
/* output scrollable */
.output-box { max-height: 500px; overflow-y: auto; }
"""

print("Loading LoGPT model...")
analyzer = LogAnalyzer()
print(f"Model ready on {analyzer.device}")


def read_input(file, pasted_text):
    if file is not None:
        # Gradio 5+ returns a filepath string; older versions return an object with .name
        path = file if isinstance(file, str) else file.name
        print(f"[LoGPT] Reading file: {path}")
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        return content
    if pasted_text and pasted_text.strip(): 
        return pasted_text.strip()
    return None


def make_state(log_text="", task="", chunks_data=None, current=0, total=0, generating=False):
    return {
        "log_text": log_text,
        "task": task,
        "chunks_data": chunks_data or [],
        "current": current,
        "total": total,
        "generating": generating,
    }


def format_status(current, total):
    return f"📄 Chunk {current}/{total}"


def render_chunk(state):
    """Render the current chunk from state."""
    if not state or not state["chunks_data"]:
        return ""
    idx = state["current"]
    total = state["total"]
    text = state["chunks_data"][idx]
    return f"### Chunk {idx + 1}/{total}\n\n{text}"


# Analyze first chunk with streaming, store result in state
def run_analysis(file, pasted_text, task_label):
    try:
        log_text = read_input(file, pasted_text)
        if not log_text:
            state = make_state()
            yield "⚠️ Please upload a file or paste log content.", "", state, gr.update(interactive=False), gr.update(interactive=False)
            return

        task = TASK_MAP[task_label]
        chunks = analyzer.processor.chunk(log_text)
        total = len(chunks)
        print(f"[LoGPT] Starting analysis: {total} chunks, task={task}")
        state = make_state(log_text=log_text, task=task, chunks_data=[""] * total, current=0, total=total, generating=True)

        status = f"⏳ Analyzing chunk 1/{total}..."
        partial = ""
        for partial in analyzer.generate_stream(chunks[0], task):
            state["chunks_data"][0] = partial
            yield f"### Chunk 1/{total}\n\n{partial}", status, state, gr.update(interactive=False), gr.update(interactive=False)

        state["chunks_data"][0] = partial
        state["generating"] = False
        has_next = total > 1
        yield (
            f"### Chunk 1/{total}\n\n{partial}",
            format_status(1, total),
            state,
            gr.update(interactive=False),
            gr.update(interactive=has_next),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        state = make_state()
        yield f"**Error:** {e}", "", state, gr.update(interactive=False), gr.update(interactive=False)


# Navigate to the next chunk — generate if not yet analyzed, otherwise just display
def go_next(state):
    if not state or state["current"] >= state["total"] - 1:
        yield render_chunk(state), format_status(state["current"] + 1, state["total"]), state, gr.update(), gr.update()
        return

    state["current"] += 1
    idx = state["current"]
    total = state["total"]

    if not state["chunks_data"][idx]:
        try:
            chunks = analyzer.processor.chunk(state["log_text"])
            status = f"⏳ Analyzing chunk {idx + 1}/{total}..."
            partial = ""
            for partial in analyzer.generate_stream(chunks[idx], state["task"]):
                state["chunks_data"][idx] = partial
                yield f"### Chunk {idx + 1}/{total}\n\n{partial}", status, state, gr.update(interactive=False), gr.update(interactive=False)
            state["chunks_data"][idx] = partial
        except Exception as e:
            import traceback
            traceback.print_exc()
            state["chunks_data"][idx] = f"**Error:** {e}"

    has_prev = idx > 0
    has_next = idx < total - 1
    yield (
        render_chunk(state),
        format_status(idx + 1, total),
        state,
        gr.update(interactive=has_prev),
        gr.update(interactive=has_next),
    )


# Navigate to the previous chunk — always instant since it was already generated
def go_prev(state):
    if not state or state["current"] <= 0:
        return render_chunk(state), format_status(1, state.get("total", 0)), state, gr.update(), gr.update()

    state["current"] -= 1
    idx = state["current"]
    total = state["total"]
    has_prev = idx > 0
    has_next = idx < total - 1
    return (
        render_chunk(state),
        format_status(idx + 1, total),
        state,
        gr.update(interactive=has_prev),
        gr.update(interactive=has_next),
    )


with gr.Blocks(title="LoGPT") as app:
    gr.Markdown(DESCRIPTION)
    session_state = gr.State(make_state())

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload log file", file_types=[".log", ".txt"])
            paste_input = gr.Textbox(label="Or paste logs here", lines=8, placeholder="Paste raw log content...", elem_classes="paste-box")
            task_dropdown = gr.Dropdown(
                choices=list(TASK_MAP.keys()),
                value="Summary",
                label="Analysis Type",
            )
            analyze_btn = gr.Button("🔍 Analyze", variant="primary")

        with gr.Column(scale=2):
            with gr.Row():
                prev_btn = gr.Button("◀", interactive=False, elem_classes="nav-btn", scale=0)
                status_bar = gr.Markdown(value="", elem_classes="status-text")
                next_btn = gr.Button("▶", interactive=False, elem_classes="nav-btn", scale=0)
            output_area = gr.Markdown(label="Results", value="", elem_classes="output-box")

    analyze_btn.click(
        fn=run_analysis,
        inputs=[file_input, paste_input, task_dropdown],
        outputs=[output_area, status_bar, session_state, prev_btn, next_btn],
    )

    next_btn.click(
        fn=go_next,
        inputs=[session_state],
        outputs=[output_area, status_bar, session_state, prev_btn, next_btn],
    )

    prev_btn.click(
        fn=go_prev,
        inputs=[session_state],
        outputs=[output_area, status_bar, session_state, prev_btn, next_btn],
    )


if __name__ == "__main__":
    app.launch(theme=THEME, css=CUSTOM_CSS, ssr_mode=False)
