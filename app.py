import os
import sys
import subprocess
import re
import time
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from pydub import AudioSegment

# ---------- UTF-8 CHO WINDOWS ----------
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"

# ---------- FFMPEG PATH ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] = BASE_DIR + os.pathsep + os.environ.get("PATH", "")

try:
    subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    print("✅ FFmpeg đã sẵn sàng")
except Exception:
    print("❌ FFmpeg KHÔNG hoạt động")
    sys.exit(1)


# ============================================================
#  LOAD MODELS
# ============================================================
print("⏳ Đang tải models…")

# --- 1. faster-whisper base (tốt cho tiếng Anh) ---
from faster_whisper import WhisperModel
print("  → faster-whisper 'base'…")
fw_model = WhisperModel("base", device="cpu", compute_type="int8")
print("  ✅ faster-whisper sẵn sàng")

# --- 2. PhoWhisper base (tốt cho tiếng Việt) ---
import torch
import librosa
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor

DEVICE = torch.device("cpu")
PHO_MODEL_NAME = "vinai/PhoWhisper-base"

print(f"  → {PHO_MODEL_NAME}…")
pho_processor = WhisperProcessor.from_pretrained(PHO_MODEL_NAME)
pho_model = WhisperForConditionalGeneration.from_pretrained(PHO_MODEL_NAME).to(DEVICE)
pho_model.eval()
print("  ✅ PhoWhisper sẵn sàng")

# --- 3. MarianMT (dịch en<->vi) ---
from transformers import MarianMTModel, MarianTokenizer
MT_MODELS = {}
for src, tgt, name in [
    ('en', 'vi', 'Helsinki-NLP/opus-mt-en-vi'),
    ('vi', 'en', 'Helsinki-NLP/opus-mt-vi-en'),
]:
    print(f"  → {name}…")
    tok = MarianTokenizer.from_pretrained(name)
    mdl = MarianMTModel.from_pretrained(name).to(DEVICE)
    mdl.eval()
    MT_MODELS[(src, tgt)] = (tok, mdl)
    print(f"  ✅ {src}→{tgt} sẵn sàng")

print("✅ Hoàn tất tải models\n")


app = Flask(__name__)
CORS(app)

LANG_LABEL = {'en': 'English', 'vi': 'Tiếng Việt'}

FILLERS = {
    'en': ['um', 'uh', 'er', 'ah', 'like', 'you know', 'so', 'well', 'actually', 'basically'],
    'vi': ['ờ', 'à', 'ừ', 'ơ', 'thì', 'là', 'mà', 'kiểu', 'nói chung', 'đại loại', 'kiểu như'],
}

# Tập ký tự có dấu tiếng Việt
VN_DIACRITICS = set(
    'ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩị'
    'óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ'
)

# Từ tiếng Anh phổ biến
EN_COMMON = {
    'the','is','are','am','was','were','a','an','and','or','but',
    'my','your','his','her','their','our','its','name','hello','hi',
    'today','tomorrow','yesterday','this','that','these','those',
    'it','he','she','they','we','you','i','me','him','us','them',
    'what','where','when','why','how','who','which','can','could',
    'will','would','shall','should','may','might','must','do','does',
    'did','have','has','had','be','been','being','in','on','at','to',
    'for','with','about','from','by','of','up','down','good','bad',
    'yes','no','not','very','so','just','now','then','here','there',
    'go','come','get','make','take','see','look','want','need','like',
    'love','know','think','say','tell','ask','work','play'
}


def count_vn_diacritics(text):
    return sum(1 for ch in text.lower() if ch in VN_DIACRITICS)


def en_word_hits(text):
    words = set(re.findall(r'\b[a-z]+\b', text.lower()))
    return len(words & EN_COMMON)


# ============================================================
#  NHẬN DẠNG
# ============================================================
def transcribe_fw(wav_path):
    """faster-whisper base — dùng cho tiếng Anh + detect ngôn ngữ."""
    segments, info = fw_model.transcribe(
        wav_path,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text, (info.language or 'en'), getattr(info, 'language_probability', 0.0)


def transcribe_pho(wav_path):
    """PhoWhisper base — dùng cho tiếng Việt."""
    audio, _ = librosa.load(wav_path, sr=16000)
    if len(audio) == 0:
        return ''
    inputs = pho_processor(
        audio, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(DEVICE)

    with torch.no_grad():
        ids = pho_model.generate(
            inputs,
            max_length=448,
            num_beams=3,
            language="vi",
            task="transcribe",
        )
    return pho_processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def transcribe_with_whisper(wav_path):
    """
    Chiến lược:
      1. Chạy faster-whisper trước (nhanh, tốt cho tiếng Anh)
      2. Nếu chắc chắn là tiếng Anh → trả về luôn
      3. Ngược lại → chạy PhoWhisper (tốt cho tiếng Việt) → trả về
    """
    t0 = time.time()

    # --- Bước 1: faster-whisper ---
    fw_text, fw_lang, fw_prob = transcribe_fw(wav_path)
    fw_diac = count_vn_diacritics(fw_text)
    fw_en_hits = en_word_hits(fw_text)

    print(f"  ⏱️  faster-whisper {time.time()-t0:.2f}s | lang={fw_lang} "
          f"(p={fw_prob:.2f}) | diac={fw_diac} | en_hits={fw_en_hits}")
    print(f"       text: {fw_text!r}")

    # --- Bước 2: Nếu chắc chắn tiếng Anh, dùng luôn ---
    is_english = (
        (fw_lang == 'en' and fw_prob > 0.5 and fw_diac == 0)
        or (fw_en_hits >= 2 and fw_diac == 0 and len(fw_text.split()) >= 2)
    )
    if is_english:
        print(f"  ✅ → Dùng faster-whisper (English)")
        return 'en', fw_text

    # --- Bước 3: Không phải tiếng Anh → dùng PhoWhisper ---
    t1 = time.time()
    try:
        pho_text = transcribe_pho(wav_path)
    except Exception as e:
        print(f"  ❌ PhoWhisper error: {e}")
        pho_text = ''
    t_pho = time.time() - t1

    pho_diac = count_vn_diacritics(pho_text)
    print(f"  ⏱️  PhoWhisper {t_pho:.2f}s | diac={pho_diac}")
    print(f"       text: {pho_text!r}")

    # Nếu PhoWhisper trả về rỗng → fallback faster-whisper
    if not pho_text or len(pho_text) < 2:
        print(f"  ⚠️  PhoWhisper rỗng → fallback faster-whisper")
        return fw_lang if fw_lang in ('en', 'vi') else 'vi', fw_text

    # Nếu PhoWhisper có dấu tiếng Việt → chắc chắn VI
    if pho_diac > 0:
        print(f"  ✅ → Dùng PhoWhisper (Vietnamese)")
        return 'vi', pho_text

    # Trường hợp lưỡng lự: PhoWhisper không có dấu nhưng faster-whisper cũng không
    # Nếu faster-whisper có nhiều từ tiếng Anh hơn → chọn EN
    if fw_en_hits >= 2 and fw_diac == 0:
        print(f"  ✅ → Dùng faster-whisper (English heuristic)")
        return 'en', fw_text

    # Mặc định: tiếng Việt (vì UI chủ yếu dùng VI)
    print(f"  ✅ → Dùng PhoWhisper (default VI)")
    return 'vi', pho_text


# ============================================================
#  DỊCH
# ============================================================
def translate_text(text, source_lang):
    if not text or not text.strip():
        return ''
    target_lang = 'vi' if source_lang == 'en' else 'en'
    key = (source_lang, target_lang)
    if key not in MT_MODELS:
        return f'(Model {source_lang}→{target_lang} chưa tải được)'

    try:
        tokenizer, model = MT_MODELS[key]
        inputs = tokenizer(
            [text], return_tensors="pt",
            padding=True, truncation=True, max_length=512,
        ).to(DEVICE)

        with torch.no_grad():
            generated = model.generate(
                **inputs, max_length=512,
                num_beams=4, early_stopping=True,
            )
        return tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    except Exception as e:
        print(f"⚠️ Translation error: {e}")
        return f"(Lỗi dịch: {e})"


# ============================================================
#  PHÂN TÍCH BÀI NÓI
# ============================================================
def count_fillers(text, lang='en'):
    fillers = FILLERS.get(lang, FILLERS['en'])
    words = re.findall(r'\b\w+\b', text.lower())
    count = 0
    for filler in fillers:
        if ' ' in filler:
            count += text.lower().count(filler)
        else:
            count += words.count(filler)
    return count


def analyze_transcript(transcript, duration_sec, lang='en'):
    if not transcript or not transcript.strip():
        return {
            'text': '', 'transcript': '(No speech detected)', 'lang': lang,
            'metrics': {'wpm': 0, 'word_count': 0, 'unique_words': 0,
                        'filler_count': 0, 'sentence_count': 0,
                        'avg_sentence_len': 0, 'duration_sec': round(duration_sec, 1)},
            'feedback': [{'type': 'bad', 'text': 'Không nhận dạng được giọng nói.'}]
        }

    words = re.findall(r'\b\w+\b', transcript)
    word_count = len(words)
    unique_words = len(set(w.lower() for w in words))
    sentences = [s.strip() for s in re.split(r'[.!?。！？]+', transcript) if s.strip()]
    sentence_count = len(sentences)
    avg_sentence_len = word_count / sentence_count if sentence_count else 0
    filler_count = count_fillers(transcript, lang)
    wpm = (word_count / duration_sec) * 60 if duration_sec > 0 else 0

    metrics = {
        'wpm': round(wpm, 1), 'word_count': word_count,
        'unique_words': unique_words, 'filler_count': filler_count,
        'sentence_count': sentence_count,
        'avg_sentence_len': round(avg_sentence_len, 1),
        'duration_sec': round(duration_sec, 1),
    }

    feedback = []
    if wpm < 80:
        feedback.append({'type': 'warn', 'text': f'Tốc độ nói chậm ({wpm:.0f} WPM). Nên đạt 120–160 WPM.'})
    elif wpm > 200:
        feedback.append({'type': 'warn', 'text': f'Tốc độ nói nhanh ({wpm:.0f} WPM). Hãy nói chậm lại.'})
    else:
        feedback.append({'type': 'good', 'text': f'Tốc độ nói tốt ({wpm:.0f} WPM).'})

    if filler_count > 5:
        feedback.append({'type': 'warn', 'text': f'Dùng {filler_count} từ đệm. Hãy giảm bớt.'})
    else:
        feedback.append({'type': 'good', 'text': f'Ít từ đệm ({filler_count}) – tốt!'})

    if unique_words < 15:
        feedback.append({'type': 'warn', 'text': f'Vốn từ hạn chế ({unique_words} từ khác nhau).'})
    else:
        feedback.append({'type': 'good', 'text': f'Vốn từ phong phú ({unique_words} từ khác nhau).'})

    if sentence_count < 3:
        feedback.append({'type': 'info', 'text': 'Nên chia thành nhiều câu để mạch lạc hơn.'})
    else:
        feedback.append({'type': 'good', 'text': f'Cấu trúc tốt – {sentence_count} câu.'})

    return {
        'text': transcript, 'transcript': transcript,
        'lang': lang, 'metrics': metrics, 'feedback': feedback,
    }


# ============================================================
#  ROUTES
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')


def _process_audio(audio_file):
    debug_dir = "debug"
    os.makedirs(debug_dir, exist_ok=True)
    webm_path = os.path.join(debug_dir, "uploaded.webm")
    wav_path = os.path.join(debug_dir, "converted.wav")

    audio_file.save(webm_path)
    print(f"💾 WebM saved: {os.path.getsize(webm_path)} bytes")

    try:
        audio = AudioSegment.from_file(webm_path, format="webm")
        print(f"✅ Audio: {len(audio)/1000:.2f}s")
        audio.export(wav_path, format="wav")
    except Exception as e:
        return {'error': f'Conversion failed: {e}'}, 400

    try:
        detected_lang, transcript = transcribe_with_whisper(wav_path)
    except Exception as e:
        print(f"❌ ASR error: {e}")
        return {'error': f'ASR error: {e}'}, 500

    print(f"📝 [{detected_lang}] {transcript}")

    translation = translate_text(transcript, detected_lang)
    target_lang = 'vi' if detected_lang == 'en' else 'en'
    print(f"🌐 [{target_lang}] {translation}")

    duration_sec = len(audio) / 1000.0
    result = analyze_transcript(transcript, duration_sec, lang=detected_lang)

    result.update({
        'text': transcript,
        'lang': detected_lang,
        'langLabel': LANG_LABEL.get(detected_lang, detected_lang),
        'translation': translation,
        'targetLang': target_lang,
        'targetLangLabel': LANG_LABEL.get(target_lang, target_lang),
    })
    return result, 200


@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    result, status = _process_audio(audio_file)
    return jsonify(result), status


@app.route('/analyze', methods=['POST'])
def analyze_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    result, status = _process_audio(audio_file)
    return jsonify(result), status


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)