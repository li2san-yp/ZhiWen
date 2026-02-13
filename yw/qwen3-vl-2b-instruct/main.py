from flask import Flask, request, jsonify, send_from_directory, Response
from transformers import AutoProcessor, TextIteratorStreamer
import torch
import traceback
import logging
import base64
import io
from PIL import Image
from threading import Thread

app = Flask(__name__)

# Model load (load once at startup)
model_path = "./Qwen3-VL-2B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from transformers import Qwen3VLForConditionalGeneration  # type: ignore
    _MODEL_CLASS = Qwen3VLForConditionalGeneration
except Exception:
    from transformers import AutoModelForImageTextToText
    _MODEL_CLASS = AutoModelForImageTextToText

logging.info("Loading model...")
model = _MODEL_CLASS.from_pretrained(
    model_path,
    device_map="cuda",
    dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=False,
)
model = model.to(device) # type: ignore
processor = AutoProcessor.from_pretrained(model_path)

model.eval()
logging.info("Model loaded successfully.")


MAX_IMAGE_PIXELS = 1024 * 1024 * 8
MAX_IMAGE_EDGE = 1024


def _resize_image(image: Image.Image) -> Image.Image:
    if image.width * image.height > MAX_IMAGE_PIXELS or max(image.width, image.height) > MAX_IMAGE_EDGE:
        image = image.copy()
        image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.LANCZOS) # type: ignore
    return image


def _decode_data_url_image(data_url: str):
    if not isinstance(data_url, str):
        return None
    if not data_url.startswith("data:"):
        return None
    try:
        header, b64 = data_url.split(",", 1)
        if "base64" not in header:
            return None
        image_bytes = base64.b64decode(b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return _resize_image(image)
    except Exception:
        return None


def _normalize_messages(data):
    if isinstance(data, dict) and "messages" in data:
        messages = data.get("messages", [])
    elif isinstance(data, dict) and "prompt" in data:
        messages = [
            {"role": "user", "content": [{"type": "text", "text": data.get("prompt", "")}]}  # compat
        ]
    else:
        return None

    normalized = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts = []
        if isinstance(content, str):
            if content.strip():
                parts.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image":
                    image = _decode_data_url_image(item.get("image"))  # type: ignore
                    if image is not None:
                        parts.append({"type": "image", "image": image})
                elif item.get("type") == "text":
                    text = item.get("text", "")
                    if isinstance(text, str) and text.strip():
                        parts.append({"type": "text", "text": text})
        if parts:
            normalized.append({"role": role, "content": parts})

    return normalized


def _run_chat(messages, max_new_tokens=None, temperature=0.7, top_p=0.9):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        gen_kwargs = dict(
            **inputs,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens
        generated_ids = model.generate(**gen_kwargs) # type: ignore

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    result = output_text[0] if output_text else ""
    # Best-effort GPU memory cleanup for large images
    try:
        del inputs, generated_ids, generated_ids_trimmed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return result


def _run_chat_stream(messages, max_new_tokens=None, temperature=0.7, top_p=0.9):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_special_tokens=True,
        decode_kwargs={"clean_up_tokenization_spaces": False},
    )

    generation_kwargs = dict(
        **inputs,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        streamer=streamer,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    if max_new_tokens is not None:
        generation_kwargs["max_new_tokens"] = max_new_tokens

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for text in streamer:
        if text:
            yield text

    # Best-effort GPU memory cleanup for large images
    try:
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        messages = _normalize_messages(data)
        if not messages:
            return jsonify({"error": "Missing 'messages' in request body"}), 400

        max_new_tokens = data.get("max_new_tokens")
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)

        output = _run_chat(messages, max_new_tokens, temperature, top_p)
        return jsonify({"output": output})

    except Exception as e:
        error_msg = str(e)
        logging.error(error_msg)
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    try:
        data = request.get_json()
        messages = _normalize_messages(data)
        if not messages:
            return jsonify({"error": "Missing 'messages' in request body"}), 400

        max_new_tokens = data.get("max_new_tokens")
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)

        def generate_stream():
            try:
                for chunk in _run_chat_stream(messages, max_new_tokens, temperature, top_p):
                    yield chunk
            except Exception as e:
                err = f"\n[STREAM_ERROR] {str(e)}"
                yield err

        return Response(generate_stream(), mimetype="text/plain")

    except Exception as e:
        error_msg = str(e)
        logging.error(error_msg)
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500


# Backward-compatible endpoint
@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        if not data or "prompt" not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400

        messages = _normalize_messages(data)
        max_new_tokens = data.get("max_new_tokens")
        temperature = data.get("temperature", 0.7)
        top_p = data.get("top_p", 0.9)

        output = _run_chat(messages, max_new_tokens, temperature, top_p)
        return jsonify({"generated_text": output})

    except Exception as e:
        error_msg = str(e)
        logging.error(error_msg)
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "Qwen3-VL-4B-Instruct", "device": device})


@app.route("/", methods=["GET"])
def default_index():
    return send_from_directory(".", "index.html")

@app.route("/styles.css")
def styles():
    return send_from_directory(".", "styles.css")

if __name__ == "__main__":
    logging.basicConfig(
        filename="file.log",
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    app.run(host="0.0.0.0", port=4543, debug=False)
