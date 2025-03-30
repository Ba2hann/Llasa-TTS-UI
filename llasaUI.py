import gradio as gr
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import soundfile as sf
import numpy as np
import os
import tempfile
import librosa
import traceback

try:
    from optimum.quanto import QuantizedModelForCausalLM, qint4, qfloat8, qint8 # Import types
    print("Successfully imported optimum-quanto.")
    QUANTO_AVAILABLE = True
    # Choose quantization type here (e.g., qint4 for 4-bit, qint8 for 8-bit)
    QUANTIZATION_TYPE = qint4
    QUANTIZATION_TYPE_STR = "qint4"
except ImportError:
    print("Warning: optimum-quanto library not found. Quantization disabled.")
    print("Install it using: pip install optimum-quanto")
    QUANTO_AVAILABLE = False
    QUANTIZATION_TYPE = None
    QUANTIZATION_TYPE_STR = "None (optimum-quanto not installed)"

FLASH_ATTENTION_AVAILABLE = False
try:
    import flash_attn
    FLASH_ATTENTION_AVAILABLE = True
    print("Flash Attention 2 available.")
except ImportError:
    print("Flash Attention 2 not found. Install with 'pip install flash-attn --no-build-isolation' for potential speedup.")

LLASA_MODEL_NAME = 'HKUSTAudio/Llasa-3B'
XCODEC_MODEL_NAME = 'HKUSTAudio/xcodec2'
WHISPER_MODEL_NAME = "openai/whisper-large-v3-turbo"
TARGET_SAMPLE_RATE = 16000
MAX_GENERATION_LENGTH = 2048

print("Checking for CUDA availability...")
llasa_model = None # Initialize
llasa_tokenizer = None
codec_model = None
whisper_pipeline = None

if torch.cuda.is_available():
    DEVICE = "cuda:0"
    BASE_MODEL_DTYPE = torch.float16 # Load base model in fp16 before quantizing
    print(f"CUDA found. Using device: {DEVICE}")

    # 1. Load LLaSA Tokenizer (always needed)
    try:
        print(f"Loading LLaSA tokenizer ({LLASA_MODEL_NAME})...")
        llasa_tokenizer = AutoTokenizer.from_pretrained(LLASA_MODEL_NAME)
        print("LLaSA tokenizer loaded successfully.")
    except Exception as e:
        print(f"FATAL: Failed to load LLaSA tokenizer: {e}")
        # Exit or handle appropriately if tokenizer is critical

    # 2. Load Base LLaSA Model (FP16) - Try with Flash Attention
    base_llasa_model = None
    load_kwargs = {
        "torch_dtype": BASE_MODEL_DTYPE,
        # Load model fully to device before potential quantization
        # "device_map": "auto", # Avoid device_map with manual .to() and quantization
    }
    attn_implementation_used = "sdpa" # Default in newer transformers
    if FLASH_ATTENTION_AVAILABLE:
        # Try loading with Flash Attention 2 if available
        load_kwargs["attn_implementation"] = "flash_attention_2"
        attn_implementation_used = "flash_attention_2"
        print("Attempting to load LLaSA with Flash Attention 2...")
    else:
        # Recommend SDPA if available (usually default in recent transformers)
        # Check if `attn_implementation` is supported by the loaded transformers version
        # For simplicity, we'll rely on the default behavior if flash_attn isn't installed.
        # You could explicitly set "sdpa" or "eager" if needed.
        print("Flash Attention 2 not available, using default attention mechanism (likely SDPA).")


    try:
        print(f"Loading base LLaSA model ({LLASA_MODEL_NAME}) in {BASE_MODEL_DTYPE} with attn_implementation='{attn_implementation_used}'...")
        base_llasa_model = AutoModelForCausalLM.from_pretrained(
            LLASA_MODEL_NAME,
            **load_kwargs
        )
        print(f"Moving base LLaSA model to {DEVICE}...")
        base_llasa_model = base_llasa_model.to(DEVICE)
        print("Base LLaSA model loaded and moved successfully.")
        llasa_model = base_llasa_model # Set as default in case quantization fails
    except Exception as e_load:
        # If Flash Attention fails, maybe try default
        if attn_implementation_used == "flash_attention_2":
             print(f"Warning: Failed to load LLaSA with Flash Attention 2: {e_load}")
             print("Retrying with default attention mechanism...")
             load_kwargs.pop("attn_implementation", None) # Remove flash attn kwarg
             attn_implementation_used = "default (likely SDPA)"
             try:
                 base_llasa_model = AutoModelForCausalLM.from_pretrained(
                     LLASA_MODEL_NAME,
                     **load_kwargs
                 ).to(DEVICE)
                 print("Base LLaSA model loaded successfully with default attention.")
                 llasa_model = base_llasa_model
             except Exception as e_retry:
                 print(f"FATAL: Failed to load base LLaSA model even with default attention: {e_retry}")
                 base_llasa_model = None
        else:
            # Failed even without flash attention
             print(f"FATAL: Failed to load base LLaSA model: {e_load}")
             base_llasa_model = None # Ensure it's None if loading failed


    # 3. Attempt Quantization
    if base_llasa_model is not None and QUANTO_AVAILABLE and QUANTIZATION_TYPE is not None:
        try:
            print(f"Attempting to quantize LLaSA model using optimum-quanto ({QUANTIZATION_TYPE_STR})...")
            # Exclude lm_head - IMPORTANT for generative models
            # Verify 'lm_head' is the correct name for LLaSA's output layer if issues arise
            llasa_model = QuantizedModelForCausalLM.quantize(
                base_llasa_model,
                weights=QUANTIZATION_TYPE,
                exclude='lm_head' # Common practice, adjust if LLaSA uses a different name
            )
            print(f"LLaSA model quantized successfully ({QUANTIZATION_TYPE_STR}).")
            # Optional: Delete base model if quantization successful and VRAM is tight
            del base_llasa_model
            torch.cuda.empty_cache()
            print("Base FP16 model deleted to save VRAM.")
        except Exception as e_quant:
            print(f"Warning: optimum-quanto quantization failed: {e_quant}")
            print("Falling back to use the original FP16 LLaSA model.")
            llasa_model = base_llasa_model # Ensure we fall back correctly
    elif base_llasa_model is not None:
         print(f"Info: Using original FP16 LLaSA model (Quantization skipped or failed). Attention: {attn_implementation_used}.")
         llasa_model = base_llasa_model # Use the loaded fp16 model
    else:
         print("Info: LLaSA model could not be loaded.")
         llasa_model = None # Explicitly set to None

    # 4. Load XCodec2
    try:
        print(f"Loading XCodec2 model ({XCODEC_MODEL_NAME})...")
        from xcodec2.modeling_xcodec2 import XCodec2Model
        codec_model = XCodec2Model.from_pretrained(XCODEC_MODEL_NAME)
        codec_model = codec_model.to(DEVICE) # Move to device
        codec_model = codec_model.eval()
        print("XCodec2 model loaded successfully.")
    except ImportError:
        print("ERROR: xcodec2 library not found. Please install it.")
        codec_model = None
    except Exception as e:
        print(f"FATAL: Failed to load XCodec2 model: {e}")
        codec_model = None

    # 5. Load Whisper
    try:
        print(f"Loading Whisper model ({WHISPER_MODEL_NAME}) for transcription...")
        whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            WHISPER_MODEL_NAME,
            torch_dtype=torch.float16, # Whisper usually runs well in FP16
            device=DEVICE,
            model_kwargs={"attn_implementation": "sdpa"} # Use SDPA for whisper too if available
        )
        print("Whisper model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load Whisper model: {e}. Automatic transcription disabled.")
        whisper_pipeline = None

else:
    # CPU Loading (Quantization typically less beneficial/supported on CPU)
    DEVICE = "cpu"
    attn_implementation_used = "eager (CPU default)" # No flash/sdpa on cpu usually
    print("WARNING: CUDA not found. Using CPU. Performance will be slower. Quantization skipped.")
    try:
        print(f"Loading LLaSA tokenizer ({LLASA_MODEL_NAME})...")
        llasa_tokenizer = AutoTokenizer.from_pretrained(LLASA_MODEL_NAME)
        print("LLaSA tokenizer loaded.")
        print(f"Loading LLaSA model ({LLASA_MODEL_NAME}) on CPU (FP32)...")
        # Note: attn_implementation might not be applicable or beneficial on CPU
        llasa_model = AutoModelForCausalLM.from_pretrained(LLASA_MODEL_NAME)
        print("LLaSA model loaded successfully on CPU.")
    except Exception as e:
        print(f"FATAL: Failed to load LLaSA model on CPU: {e}")
        llasa_model = None
        llasa_tokenizer = None

    try:
        print(f"Loading XCodec2 model ({XCODEC_MODEL_NAME}) on CPU...")
        from xcodec2.modeling_xcodec2 import XCodec2Model
        codec_model = XCodec2Model.from_pretrained(XCODEC_MODEL_NAME).eval()
        print("XCodec2 model loaded successfully on CPU.")
    except ImportError:
        print("ERROR: xcodec2 library not found. Please install it.")
        codec_model = None
    except Exception as e:
        print(f"FATAL: Failed to load XCodec2 model on CPU: {e}")
        codec_model = None

    try:
        print(f"Loading Whisper model ({WHISPER_MODEL_NAME}) on CPU...")
        whisper_pipeline = pipeline("automatic-speech-recognition", WHISPER_MODEL_NAME, device=DEVICE)
        print("Whisper model loaded successfully on CPU.")
    except Exception as e:
        print(f"ERROR: Failed to load Whisper model on CPU: {e}. Automatic transcription disabled.")
        whisper_pipeline = None

# Set models to evaluation mode (important after quantization too)
if llasa_model:
    llasa_model.eval()
if codec_model:
    codec_model.eval()

def preprocess_audio(audio_path, target_sr=TARGET_SAMPLE_RATE):
    """Loads, converts to mono, resamples, and returns path to temp WAV and tensor."""
    if not audio_path:
        return None, None, "Error: No audio file provided."

    processed_path = None # Initialize in case of early error
    try:
        wav, sr = librosa.load(audio_path, sr=None, mono=False)

        if wav.ndim > 1:
            # Smart mono conversion based on shape
            if wav.shape[0] == 1: wav = wav[0]
            elif wav.shape[1] == 1: wav = wav[:, 0]
            else: wav = librosa.to_mono(wav) # Average if truly stereo

        if sr != target_sr:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)

        wav = wav.astype(np.float32)

        # Create a temporary file for the processed audio
        # Use context manager for safer file handling
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            processed_path = tmpfile.name
        sf.write(processed_path, wav, target_sr) # Write after closing handle in 'with'

        # Return path AND the tensor
        return processed_path, torch.from_numpy(wav).float().unsqueeze(0), None

    except Exception as e:
        error_msg = f"Error processing audio file '{os.path.basename(audio_path)}': {e}\n{traceback.format_exc()}"
        print(error_msg)
        # Clean up temp file if it was created before the error
        if processed_path and os.path.exists(processed_path):
            try: os.remove(processed_path)
            except OSError: pass
        return None, None, error_msg

def ids_to_speech_tokens(speech_ids):
    """Converts integer speech IDs to LLaSA token format <|s_ID|>."""
    return [f"<|s_{int(speech_id)}|>" for speech_id in speech_ids]

def extract_speech_ids(speech_tokens_str):
    """Extracts integer speech IDs from LLaSA token format <|s_ID|>."""
    speech_ids = []
    import re
    # Improved regex to handle spaces sometimes inserted by tokenizers
    pattern = re.compile(r'<\|\s*s_(\d+)\s*\|>')
    # If input is a list of strings, join them first. If it's one string, use it directly.
    full_str = "".join(speech_tokens_str) if isinstance(speech_tokens_str, list) else speech_tokens_str
    matches = pattern.findall(full_str)
    for num_str in matches:
        try:
            speech_ids.append(int(num_str))
        except ValueError:
            print(f"Warning: Could not parse integer from match: {num_str}")
    return speech_ids



def _llasa_generate_core(model, input_ids, **generation_kwargs):
    """Helper for the actual model.generate call."""
    # The inference_mode context will be applied outside this function
    outputs = model.generate(
        input_ids,
        **generation_kwargs
    )
    return outputs

def transcribe_audio(ref_audio_path):
    """
    Processes audio and runs Whisper transcription.
    Returns transcription text and status message.
    """
    if not ref_audio_path:
        return "Please upload a reference audio file first.", "Status: Idle"
    if not whisper_pipeline:
        return "Whisper model not available. Cannot transcribe automatically.", "Status: Error"

    status = "Processing audio for transcription..."
    yield "", status # Clear previous transcription, update status

    processed_ref_path, _, error = preprocess_audio(ref_audio_path) # Don't need tensor here

    if error:
        # No need to yield here, just return
        return f"Audio Error: {error}", "Status: Error"
    if not processed_ref_path:
        return "Audio processing failed silently.", "Status: Error" # Should have error msg

    transcription_text = ""
    try:
        status = "Transcribing using Whisper..."
        yield transcription_text, status

        # Run whisper on the *processed* 16khz mono wav
        # Explicitly set return_timestamps=False for clarity
        with torch.inference_mode(): # Use inference mode for Whisper pipeline
            result = whisper_pipeline(
                processed_ref_path,
                chunk_length_s=30,
                batch_size=8,
                return_timestamps=False # Explicitly false
            )
        transcription_text = result["text"].strip()

        if not transcription_text:
            transcription_text = ""
            status = "Transcription complete (No speech detected or empty)."
        else:
            status = "Transcription complete."

        # Yield final result *before* finally block
        yield transcription_text, status

    except Exception as e:
        error_msg = f"Whisper Error: {e}\n{traceback.format_exc()}"
        print(error_msg)
        status = "Status: Whisper Error"
        # Yield error status, potentially with partial transcription
        yield transcription_text, status
    finally:
        # Clean up the temporary processed file used for transcription
        if processed_ref_path and os.path.exists(processed_ref_path):
            try:
                os.remove(processed_ref_path)
                print(f"Cleaned up temp transcription file: {processed_ref_path}")
            except OSError as e:
                print(f"Warning: Could not delete temp transcription file {processed_ref_path}: {e}")


def generate_speech(
    ref_audio_path,
    prompt_text,
    target_text,
    temperature,
    repetition_penalty,
    top_k
    ):
    output_wav_path = None
    processed_ref_path = None

    try:
        if not ref_audio_path:
            return "Status: Error - Please upload a reference audio file.", None
        if not prompt_text or not prompt_text.strip():
            return "Status: Error - Reference transcription is missing. Please type or use the 'Transcribe' button.", None
        if not target_text or not target_text.strip():
            return "Status: Error - Please enter the target text to synthesize.", None
        if not llasa_model or not llasa_tokenizer or not codec_model:
             # Be more specific about which model failed if possible
             missing = []
             if not llasa_model: missing.append("LLaSA")
             if not llasa_tokenizer: missing.append("LLaSA Tokenizer")
             if not codec_model: missing.append("XCodec2")
             return f"Status: Error - Core models not loaded ({', '.join(missing)}).", None

        prompt_text = prompt_text.strip()
        target_text = target_text.strip()

        status_update = "Processing reference audio for generation..."
        yield status_update, None # Update UI

        # Preprocess audio outside inference_mode as it uses librosa (CPU)
        processed_ref_path, ref_wav_tensor, error = preprocess_audio(ref_audio_path)
        if error:
            return f"Audio Processing Error: {error}", None
        if ref_wav_tensor is None or processed_ref_path is None:
             return "Audio Processing Error: Failed to load/process audio.", None

        with torch.no_grad(): # Reverted from torch.inference_mode()
            # --- LLaSA Input Preparation ---
            status_update += "\nEncoding reference audio..."
            yield status_update, None

            input_text = prompt_text + " " + target_text

            ref_wav_tensor = ref_wav_tensor.to(DEVICE)

            # Encode Reference Audio
            vq_codes = codec_model.encode_code(input_waveform=ref_wav_tensor)
            print(f"Prompt VQ Code Shape: {vq_codes.shape}")
            try:
                vq_codes_prompt = vq_codes[0, 0, :]
            except IndexError:
                 if vq_codes.dim() == 1: vq_codes_prompt = vq_codes
                 elif vq_codes.dim() == 2 and vq_codes.shape[0] == 1: vq_codes_prompt = vq_codes[0]
                 else: raise ValueError(f"Cannot extract VQ codes from shape: {vq_codes.shape}")

            speech_ids_prefix_tokens = ids_to_speech_tokens(vq_codes_prompt.cpu().numpy())
            formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"


            chat = [
                {"role": "user", "content": "Convert the text to speech:" + formatted_text},
                {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix_tokens)}
            ]

            # Tokenization happens on CPU usually, but keep inside inference_mode just in case
            input_ids = llasa_tokenizer.apply_chat_template(
                chat, tokenize=True, return_tensors='pt', add_generation_prompt=False,
            ).to(DEVICE)

            prompt_length = input_ids.shape[1]
            if prompt_length >= MAX_GENERATION_LENGTH:
                 return f"ERROR: Input length ({prompt_length}) too long.", None

            # Speech Generation
            status_update += "\nGenerating speech tokens..."
            yield status_update, None

            speech_end_id = llasa_tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

            generation_kwargs = {
                "max_length": MAX_GENERATION_LENGTH,
                "eos_token_id": speech_end_id,
                "do_sample": temperature > 0.0,
                "temperature": max(temperature, 1e-6), # Ensure temperature is positive for sampling
                "top_k": int(top_k) if top_k > 0 else None,
                "repetition_penalty": repetition_penalty,
                "pad_token_id": llasa_tokenizer.eos_token_id if llasa_tokenizer.eos_token_id is not None else llasa_tokenizer.pad_token_id if llasa_tokenizer.pad_token_id is not None else 50256 # Safe fallback pad
            }
            # Filter out None values to avoid issues with generate
            generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}


            print(f"Calling generate with kwargs: {generation_kwargs}")

            outputs = _llasa_generate_core(
                 llasa_model, input_ids, **generation_kwargs
            )

            # Post-processing
            status_update += "\nDecoding speech tokens..."
            yield status_update, None

            generation_start_idx = input_ids.shape[1]
            generated_ids = outputs[0][generation_start_idx:]

            eos_indices = torch.where(generated_ids == speech_end_id)[0]
            if len(eos_indices) > 0:
                generated_ids = generated_ids[:eos_indices[0]] # Trim at the first EOS token

            # Decoding tokens to text is usually fast, keep inside inference_mode
            generated_tokens_list = llasa_tokenizer.batch_decode(generated_ids.unsqueeze(0))
            speech_tokens_int = extract_speech_ids(generated_tokens_list[0]) # extract_speech_ids is CPU string processing

            if not speech_tokens_int:
                status_update += "\nWarning: No valid speech tokens generated."
                yield status_update, None
                # Don't return here, let finally block clean up temp file
                # return status_update, None # Old behavior
            else:
                # Prepare tensor for codec decoding
                speech_tokens_tensor = torch.tensor(speech_tokens_int, dtype=torch.long, device=DEVICE).unsqueeze(0).unsqueeze(0) #[1, 1, T]

                # Decode speech tokens to audio using codec_model
                gen_wav = codec_model.decode_code(speech_tokens_tensor)
                gen_wav_np = gen_wav.squeeze().cpu().numpy() # Squeeze and move to CPU

                # Save output using a context manager for the temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_gen_file:
                    output_wav_path = tmp_gen_file.name
                sf.write(output_wav_path, gen_wav_np, TARGET_SAMPLE_RATE)

                status_update += f"\nGenerated audio: {os.path.basename(output_wav_path)}"
                yield status_update, output_wav_path

            # If no speech tokens were generated, yield the status and None for audio path
            if not speech_tokens_int:
                 yield status_update, None


    except Exception as e:
        error_msg = f"Generation Error: {e}\n{traceback.format_exc()}"
        print(error_msg)
        if output_wav_path and os.path.exists(output_wav_path):
             try: os.remove(output_wav_path)
             except OSError: pass
        # Yield the error message and None for the audio path
        yield f"Generation Failed: {error_msg}", None # Use yield here too

    finally:
        # Ensure temp file cleanup happens regardless of success/failure/yields
        if processed_ref_path and os.path.exists(processed_ref_path):
            try:
                os.remove(processed_ref_path)
                print(f"Cleaned up temp generation ref file: {processed_ref_path}")
            except OSError as e:
                print(f"Warning: Could not delete temp generation ref file {processed_ref_path}: {e}")


# Gradio

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
        # LLaSA Speech Synthesis with Speech Prompt
        1. Upload a reference audio (`.wav`, `.mp3`, etc.).
        2. Click **Transcribe Reference Audio** (optional, uses Whisper) or type the transcription manually.
        3. Edit the transcription if needed.
        4. Enter the **Target Text**.
        5. Adjust hyperparameters and click **Generate Speech**.
        *(LLaSA Model: {LLASA_MODEL_NAME}, Quantization: {QUANTIZATION_TYPE_STR}, Attention: {attn_implementation_used})*
        """
    ) # Added quantization & attention status to title

    with gr.Row():
        with gr.Column(scale=1):
            ref_audio = gr.Audio(label="1. Reference Audio", type="filepath")
            transcribe_btn = gr.Button("2. Transcribe Reference Audio (Optional)")
            ref_transcription = gr.Textbox(label="3. Reference Transcription", info="Edit Whisper's output or enter manually.", lines=4, interactive=True)
            target_text = gr.Textbox(label="4. Target Text", info="The text you want the model to speak.", lines=5)

            with gr.Accordion("Generation Hyperparameters", open=False):
                temperature = gr.Slider(minimum=0.0, maximum=1.5, value=0.8, step=0.01, label="Temperature (0=greedy)")
                repetition_penalty = gr.Slider(minimum=1.0, maximum=2.0, value=1.1, step=0.01, label="Repetition Penalty")
                top_k = gr.Slider(minimum=0, maximum=200, value=50, step=1, label="Top-K Sampling (0=disable)")

            generate_btn = gr.Button("5. Generate Speech", variant="primary")

        with gr.Column(scale=1):
            status_output = gr.Textbox(label="Status / Logs", interactive=False, lines=10)
            generated_audio = gr.Audio(label="Generated Speech Output", type="filepath")

    # --- Button Click Actions ---
    # Use yield for transcribe_audio as it updates status mid-process
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[ref_audio],
        outputs=[ref_transcription, status_output]
    )

    # Use yield for generate_speech as it updates status mid-process
    generate_btn.click(
        fn=generate_speech,
        inputs=[
            ref_audio, ref_transcription, target_text,
            temperature, repetition_penalty, top_k
        ],
        outputs=[ status_output, generated_audio ]
    )


# Launch
if __name__ == "__main__":
    if llasa_model and codec_model and llasa_tokenizer:
        print("Launching Gradio Interface...")
        demo.queue().launch(debug=False, share=False) # Disable debug for cleaner logs unless needed
    else:
        print("Essential models (LLaSA, Tokenizer, or XCodec2) failed to load. Gradio interface will not launch.")
        print("Please check model paths, dependencies (transformers, optimum-quanto, xcodec2, librosa, soundfile), and CUDA setup.")
        # Also mention flash-attn if trying to use it
        if not FLASH_ATTENTION_AVAILABLE and torch.cuda.is_available():
             print("Consider installing 'flash-attn' for potential speed improvements on compatible GPUs.")