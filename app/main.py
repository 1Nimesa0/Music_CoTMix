"""
FastAPI Serving Application for Cross-Environment Music Genre Recognition.
Provides high-performance, low-latency inference endpoints and an interactive,
production-grade Web UI modeled after the functional audio classification interface.
"""

import os
import psutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
from app.service import MusicGenrePredictor

app = FastAPI(
    title="Cross-Environment Music Genre Recognition API (CoTMix + AudioCNN)",
    description="Online low-latency inference engine optimized for Render free/starter instances.",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        backbone = os.getenv("BACKBONE", "cnn").lower()
        conf_thresh = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))
        predictor = MusicGenrePredictor(
            backbone=backbone,
            confidence_threshold=conf_thresh
        )
        print(f"[API Startup] Loaded {predictor.model_version} with backbone '{predictor.backbone_name}' on {predictor.device}.")
    return predictor


@app.on_event("startup")
def startup_event():
    get_predictor()


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Returns system operational metrics, RAM usage, and active backbone."""
    p = get_predictor()
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    return {
        "status": "ok",
        "model_version": p.model_version,
        "backbone": p.backbone_name,
        "device": str(p.device),
        "ram_usage_mb": round(mem, 1),
        "render_optimized": True,
        "classes": p.class_names
    }


@app.post("/predict", tags=["Inference"])
async def predict_genre(file: UploadFile = File(...)):
    """Predicts music genre from user-uploaded audio file (WAV, MP3, OGG, FLAC, WEBM, M4A)."""
    p = get_predictor()

    allowed_exts = (".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm", ".aac", ".opus", ".wma")
    filename = (file.filename or "audio.wav").lower()
    if not any(filename.endswith(ext) for ext in allowed_exts):
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng file không hỗ trợ. Vui lòng tải lên một trong: {allowed_exts}"
        )

    try:
        audio_bytes = await file.read()
        if len(audio_bytes) < 100:
            raise ValueError("Tệp âm thanh quá ngắn hoặc không có dữ liệu.")
        result = await run_in_threadpool(p.predict, audio_bytes)
        result["filename"] = file.filename or "uploaded_audio.wav"
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi suy luận: {str(e)}")


@app.get("/samples/{genre}", tags=["Presets"])
def get_sample_audio(genre: str):
    """Streams a bundled audio sample for the requested genre."""
    genre_dir = os.path.join("data", "audio", genre.lower())
    if not os.path.isdir(genre_dir):
        raise HTTPException(status_code=404, detail="Thể loại mẫu không tồn tại")

    files = [f for f in os.listdir(genre_dir) if f.endswith(".wav")]
    if not files:
        raise HTTPException(status_code=404, detail="Không tìm thấy file mẫu")

    sample_path = os.path.join(genre_dir, files[0])
    return FileResponse(sample_path, media_type="audio/wav", filename=f"{genre}_sample.wav")


@app.post("/sample-predict/{genre}", tags=["Presets"])
async def predict_sample_audio(genre: str):
    """Instantly evaluates a bundled sample audio without uploading."""
    p = get_predictor()

    genre_dir = os.path.join("data", "audio", genre.lower())
    if not os.path.isdir(genre_dir):
        raise HTTPException(status_code=404, detail="Thể loại mẫu không tồn tại")

    files = [f for f in os.listdir(genre_dir) if f.endswith(".wav")]
    if not files:
        raise HTTPException(status_code=404, detail="Không tìm thấy file mẫu")

    sample_path = os.path.join(genre_dir, files[0])
    result = await run_in_threadpool(p.predict, sample_path)
    result["sample_genre"] = genre
    result["filename"] = files[0]
    result["audio_url"] = f"/samples/{genre.lower()}"
    return JSONResponse(content=result)


@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
def serve_ui():
    """Serves the complete, interactive music classification dashboard."""
    html_content = """<!DOCTYPE html>
<html lang="vi" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Music DOMAIN AI - Nhận dạng thể loại âm nhạc xuyên môi trường</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        darkbg: '#080d1a',
                        cardbg: '#0f172a',
                        cardinner: '#15203b',
                        sidebarbg: '#090e1d',
                        brandBlue: '#2563eb',
                        brandCyan: '#06b6d4',
                        borderNavy: '#1e2d4d',
                        borderHover: '#2b3f6c'
                    },
                    fontFamily: {
                        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
                        mono: ['JetBrains Mono', 'monospace']
                    }
                }
            }
        }
    </script>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #080d1a;
            color: #f1f5f9;
        }
        /* Custom Glowing Scrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #080d1a; }
        ::-webkit-scrollbar-thumb { background: #1e2d4d; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #2563eb; }
        
        .card-box {
            background: #0f172a;
            border: 1px solid #1e2d4d;
            border-radius: 1rem;
            box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
        }
        .card-box:hover {
            border-color: #2b3f6c;
        }
        .glow-blue-bar {
            box-shadow: 0 0 10px rgba(37, 99, 235, 0.7);
        }
        .timeline-dot {
            width: 4px;
            height: 12px;
            background: #1e2d4d;
            border-radius: 2px;
            transition: all 0.15s ease;
        }
        .timeline-dot.active {
            background: #2563eb;
            height: 18px;
            box-shadow: 0 0 6px #2563eb;
        }
    </style>
</head>
<body class="min-h-screen flex antialiased select-none">
    <!-- NON-BLOCKING TOAST NOTIFICATIONS -->
    <div id="toast-container" class="fixed top-5 right-5 z-50 flex flex-col space-y-3 pointer-events-none max-w-sm sm:max-w-md w-full px-4"></div>

    <!-- LEFT SIDEBAR -->
    <aside class="w-64 border-r border-[#17233f] bg-[#090e1d] flex flex-col justify-between shrink-0 fixed inset-y-0 left-0 z-30">
        <div>
            <!-- Brand Logo -->
            <div class="p-5 flex items-center space-x-3.5 border-b border-[#17233f]">
                <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-blue-600 to-cyan-500 flex items-center justify-center text-white shadow-lg shadow-blue-600/30">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3"></path></svg>
                </div>
                <div>
                    <h1 class="text-base font-bold text-white tracking-tight leading-none">Music DOMAIN AI</h1>
                    <p class="text-[11px] text-blue-400 font-medium mt-1">Nhận dạng xuyên môi trường</p>
                </div>
            </div>

            <!-- Navigation Workspace -->
            <div class="px-3.5 py-5">
                <nav class="space-y-1.5">
                    <button onclick="switchTab('predict')" id="nav-predict" class="w-full flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-xs font-semibold text-white bg-blue-600 shadow-lg shadow-blue-600/30 transition">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"></path></svg>
                        <span>Trang chủ / Dự đoán</span>
                    </button>
                    
                    <button onclick="switchTab('domain')" id="nav-domain" class="w-full flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-xs font-medium text-slate-400 hover:text-white hover:bg-slate-800/60 transition">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"></path></svg>
                        <span>Kiểm tra xuyên môi trường</span>
                    </button>

                    <button onclick="switchTab('history')" id="nav-history" class="w-full flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-xs font-medium text-slate-400 hover:text-white hover:bg-slate-800/60 transition">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                        <span>Lịch sử dự đoán</span>
                    </button>

                    <button onclick="switchTab('settings')" id="nav-settings" class="w-full flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-xs font-medium text-slate-400 hover:text-white hover:bg-slate-800/60 transition">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
                        <span>Cài đặt & Telemetry</span>
                    </button>
                </nav>
            </div>
        </div>

        <!-- Security / Privacy Bottom Badge -->
        <div class="p-4 border-t border-[#17233f]">
            <div class="bg-[#121c33] p-3 rounded-xl border border-[#1d2d4f]">
                <div class="flex items-center space-x-2 text-emerald-400 text-xs font-semibold mb-1">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path></svg>
                    <span>Bảo mật dữ liệu</span>
                </div>
                <p class="text-[11px] text-slate-400 leading-relaxed">
                    Âm thanh được xử lý trực tiếp trên RAM và không lưu trữ trên máy chủ nếu bạn không yêu cầu.
                </p>
            </div>
        </div>
    </aside>

    <!-- MAIN APP CONTAINER -->
    <main class="ml-64 flex-1 min-h-screen px-8 py-7 max-w-7xl">

        <!-- TOP BAR (Title & Profile) -->
        <header class="flex items-center justify-between pb-6 border-b border-[#17233f]">
            <div class="flex items-center space-x-3.5">
                <div class="w-10 h-10 rounded-xl bg-blue-600/20 border border-blue-500/40 text-blue-400 flex items-center justify-center font-bold">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 100-6 3 3 0 000 6z"></path></svg>
                </div>
                <div>
                    <h1 class="text-xl font-bold text-white tracking-tight">Dự đoán thể loại âm nhạc xuyên môi trường</h1>
                    <p class="text-xs text-slate-400 mt-0.5">Tải lên hoặc ghi âm đoạn nhạc để AI phân tích và dự đoán thể loại của bạn.</p>
                </div>
            </div>

            <!-- Right Profile Badge -->
            <div class="flex items-center space-x-3">
                <div class="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-[#121c33] border border-[#1e2d4d] text-xs font-medium text-slate-300">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                    <span>Người dùng (Demo)</span>
                </div>
            </div>
        </header>

        <!-- TAB 1: PREDICTION & INPUT (CHÍNH) -->
        <div id="tab-predict" class="tab-pane mt-6 space-y-6">

            <!-- TWO MAIN CARDS: INPUT & PREDICTION RESULT -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">

                <!-- LEFT CARD: FUNCTION BUTTONS & INPUT -->
                <div class="card-box p-6 flex flex-col justify-between space-y-5">
                    <div>
                        <!-- TWO FUNCTION TOGGLE BUTTONS -->
                        <div class="grid grid-cols-2 gap-3 p-1 bg-[#121c33] rounded-xl border border-[#1e2d4d]">
                            <button id="mode-record-btn" onclick="setMode('record')" class="flex items-center justify-center space-x-2 py-2.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-white transition">
                                <svg class="w-4 h-4 text-rose-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 100-6 3 3 0 000 6z"></path></svg>
                                <span>Ghi âm trực tiếp</span>
                            </button>
                            <button id="mode-upload-btn" onclick="setMode('upload')" class="flex items-center justify-center space-x-2 py-2.5 rounded-lg text-xs font-semibold text-white bg-blue-600 shadow-sm transition">
                                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path></svg>
                                <span>Tải lên file âm thanh</span>
                            </button>
                        </div>

                        <!-- DROPZONE AREA (UPLOAD MODE) -->
                        <div id="upload-panel" class="mt-5">
                            <div id="drop-zone" onclick="triggerFileInput()" class="border-2 border-dashed border-[#24355a] hover:border-blue-500 bg-[#121c33]/50 hover:bg-[#121c33] rounded-2xl p-7 text-center cursor-pointer transition flex flex-col items-center justify-center space-y-3">
                                <div class="w-12 h-12 rounded-full bg-blue-600/10 text-blue-400 flex items-center justify-center border border-blue-500/20">
                                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
                                </div>
                                <div>
                                    <p id="drop-title" class="text-sm font-semibold text-slate-200">Kéo thả hoặc bấm để chọn file</p>
                                    <p class="text-[11px] text-slate-400 mt-1">MP3, WAV, M4A, OGG, FLAC, WEBM &bull; Tối đa 15 MB</p>
                                </div>
                                <div id="selected-file-pill" class="hidden items-center space-x-2 px-3 py-1 rounded-full bg-blue-950/80 border border-blue-600 text-blue-300 text-xs">
                                    <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                                    <span id="selected-file-name" class="font-medium truncate max-w-[200px]">audio.wav</span>
                                </div>
                            </div>
                            <input type="file" id="real-file-input" accept="audio/*" class="hidden">
                        </div>

                        <!-- RECORD PANEL (LIVE MIC MODE) -->
                        <div id="record-panel" class="hidden mt-5">
                            <div class="border-2 border-dashed border-[#24355a] bg-[#121c33]/50 rounded-2xl p-7 text-center flex flex-col items-center justify-center space-y-3">
                                <div id="mic-pulse-ring" class="w-14 h-14 rounded-full bg-rose-600/20 text-rose-400 flex items-center justify-center border border-rose-500/40">
                                    <svg class="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 100-6 3 3 0 000 6z"></path></svg>
                                </div>
                                <div>
                                    <p id="record-status-label" class="text-sm font-semibold text-slate-200">Sẵn sàng ghi âm giọng hát / bài nhạc</p>
                                    <p id="record-timer" class="text-lg font-bold text-rose-400 font-mono mt-1">00:00</p>
                                </div>
                                <div class="flex items-center space-x-2 pt-1">
                                    <button id="record-action-btn" onclick="toggleRecordState()" class="px-5 py-2.5 rounded-xl text-xs font-semibold bg-rose-600 hover:bg-rose-500 text-white shadow-lg shadow-rose-600/30 transition flex items-center space-x-1.5">
                                        <span id="record-btn-text">Bắt đầu ghi âm</span>
                                    </button>
                                </div>
                            </div>
                        </div>

                        <!-- AUDIO PREVIEW BAR -->
                        <div class="mt-5 p-3.5 bg-[#121c33] rounded-xl border border-[#1e2d4d] flex items-center space-x-3.5">
                            <button id="main-play-btn" onclick="togglePlay()" class="w-9 h-9 rounded-full bg-blue-600/20 hover:bg-blue-600/40 text-blue-400 border border-blue-500/30 flex items-center justify-center shrink-0 transition">
                                <svg id="main-play-svg" class="w-4 h-4 ml-0.5" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                            </button>

                            <!-- Waveform / Timeline dots -->
                            <div class="flex-1 flex items-center space-x-1 h-5 overflow-hidden" id="timeline-dots-container">
                                <!-- Generated dynamically -->
                            </div>

                            <span id="player-time-display" class="text-xs text-slate-400 font-mono shrink-0 font-medium">0:00 / 0:00</span>
                        </div>

                        <!-- PRESETS QUICK-TEST BUTTONS -->
                        <div class="mt-4 flex flex-wrap items-center gap-2">
                            <span class="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mr-1">Mẫu thử:</span>
                            <button onclick="pickPreset('jazz')" class="preset-btn px-2.5 py-1 rounded-lg text-xs bg-[#121c33] hover:bg-blue-950 text-slate-300 hover:text-blue-300 border border-[#1e2d4d] hover:border-blue-600 transition">🎷 Jazz</button>
                            <button onclick="pickPreset('blues')" class="preset-btn px-2.5 py-1 rounded-lg text-xs bg-[#121c33] hover:bg-blue-950 text-slate-300 hover:text-blue-300 border border-[#1e2d4d] hover:border-blue-600 transition">🎸 Blues</button>
                            <button onclick="pickPreset('classical')" class="preset-btn px-2.5 py-1 rounded-lg text-xs bg-[#121c33] hover:bg-blue-950 text-slate-300 hover:text-blue-300 border border-[#1e2d4d] hover:border-blue-600 transition">🎻 Classical</button>
                            <button onclick="pickPreset('rock')" class="preset-btn px-2.5 py-1 rounded-lg text-xs bg-[#121c33] hover:bg-blue-950 text-slate-300 hover:text-blue-300 border border-[#1e2d4d] hover:border-blue-600 transition">⚡ Rock</button>
                            <button onclick="pickPreset('pop')" class="preset-btn px-2.5 py-1 rounded-lg text-xs bg-[#121c33] hover:bg-blue-950 text-slate-300 hover:text-blue-300 border border-[#1e2d4d] hover:border-blue-600 transition">🎧 Pop</button>
                        </div>
                    </div>

                    <!-- BIG PREDICT BUTTON (✦ Dự đoán thể loại) -->
                    <button id="run-predict-btn" onclick="executePrediction()" class="w-full py-3.5 rounded-xl font-bold text-sm text-white bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 shadow-lg shadow-blue-600/30 flex items-center justify-center space-x-2 transition transform active:scale-[0.99]">
                        <span id="predict-btn-icon">✦</span>
                        <span id="predict-btn-text">Dự đoán thể loại âm nhạc</span>
                        <div id="predict-spinner" class="hidden w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    </button>
                </div>

                <!-- RIGHT CARD: KẾT QUẢ DỰ ĐOÁN (Results) -->
                <div class="card-box p-6 flex flex-col justify-between space-y-6">
                    <div>
                        <!-- Header -->
                        <div class="flex items-center justify-between pb-3 border-b border-[#17233f]">
                            <div class="flex items-center space-x-2 text-blue-400 font-bold text-xs uppercase tracking-wider">
                                <span>✦</span>
                                <h3 class="text-sm font-bold text-white tracking-tight">Kết quả dự đoán</h3>
                            </div>
                            <span id="status-pill-badge" class="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-slate-800 text-slate-400 border border-slate-700">
                                Chờ dự đoán
                            </span>
                        </div>

                        <!-- Predicted Genre Hero Block -->
                        <div class="mt-6 flex items-center space-x-4 p-4 rounded-2xl bg-[#121c33] border border-[#1e2d4d]">
                            <div id="genre-avatar-box" class="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-600/20 to-purple-600/20 text-3xl flex items-center justify-center border border-blue-500/30 shrink-0">
                                🎵
                            </div>
                            <div>
                                <span class="text-[10px] font-bold text-slate-400 uppercase tracking-wider">THỂ LOẠI DỰ ĐOÁN</span>
                                <h2 id="result-genre-title" class="text-3xl font-extrabold text-white mt-0.5">--</h2>
                                <div class="mt-1 flex items-center space-x-2">
                                    <span id="result-conf-pill" class="text-xs font-bold text-slate-400 font-mono bg-slate-800/80 px-2 py-0.5 rounded-md border border-slate-700">
                                        Độ tin cậy: —
                                    </span>
                                    <span id="result-sublabel" class="text-xs text-slate-400">Chờ dữ liệu phân tích</span>
                                </div>
                            </div>
                        </div>

                        <!-- Probability Distribution (Xác suất cho từng thể loại) -->
                        <div class="mt-6">
                            <div class="flex items-center justify-between text-xs font-semibold text-slate-300 mb-3">
                                <span>Xác suất cho từng thể loại</span>
                                <span class="text-[10px] text-blue-400 font-mono">Ranked Softmax</span>
                            </div>

                            <!-- Initial waiting placeholder -->
                            <div id="probability-placeholder" class="text-center py-7 px-4 rounded-xl border border-dashed border-[#1e2d4d] bg-[#121c33]/40">
                                <p class="text-xs text-slate-300 font-medium">Chưa có kết quả dự đoán</p>
                                <p class="text-[11px] text-slate-500 mt-1">Vui lòng tải tệp hoặc ghi âm rồi bấm <span class="text-blue-400 font-semibold">"✦ Dự đoán thể loại âm nhạc"</span> để xem kết quả.</p>
                            </div>

                            <!-- Filled dynamically after predict -->
                            <div id="probability-list" class="hidden space-y-3.5"></div>
                        </div>
                    </div>

                    <!-- Domain Shift & Telemetry Tag -->
                    <div class="pt-4 border-t border-[#17233f] flex items-center justify-between text-xs text-slate-400">
                        <div class="flex items-center space-x-1.5">
                            <span id="domain-dot" class="w-2 h-2 rounded-full bg-slate-500"></span>
                            <span>CoTMix Adaptation: <strong id="domain-text" class="text-slate-400">Sẵn sàng</strong></span>
                        </div>
                        <span id="result-latency" class="font-mono text-slate-500">-- ms</span>
                    </div>
                </div>

            </div>

            <!-- BOTTOM CARD: LỊCH SỬ DỰ ĐOÁN (Table) -->
            <div class="card-box p-6 space-y-4">
                <div class="flex items-center justify-between pb-3 border-b border-[#17233f]">
                    <div class="flex items-center space-x-2 text-white font-bold text-sm">
                        <svg class="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                        <span>Lịch sử dự đoán</span>
                    </div>
                    <button onclick="clearHistory()" class="text-xs font-semibold text-slate-400 hover:text-rose-400 transition flex items-center space-x-1">
                        <span>Xoá lịch sử</span>
                    </button>
                </div>

                <div class="overflow-x-auto">
                    <table class="w-full text-left text-xs">
                        <thead class="text-slate-400 border-b border-[#17233f] font-semibold">
                            <tr>
                                <th class="py-2.5 px-3">Thời gian</th>
                                <th class="py-2.5 px-3">Tệp âm thanh</th>
                                <th class="py-2.5 px-3">Thể loại dự đoán</th>
                                <th class="py-2.5 px-3">Độ tin cậy</th>
                                <th class="py-2.5 px-3">Phân phối xác suất</th>
                                <th class="py-2.5 px-3">Độ trễ</th>
                            </tr>
                        </thead>
                        <tbody id="history-table-body" class="divide-y divide-[#17233f] text-slate-300">
                            <tr id="empty-history-row">
                                <td colspan="6" class="text-center py-6 text-slate-500">
                                    Chưa có dự đoán nào. Hãy tải lên tệp âm thanh hoặc bấm "Dự đoán thể loại âm nhạc".
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>

                <p class="text-[11px] text-slate-500 pt-2 border-t border-[#17233f]">
                    Giao diện theo thiết kế "Music Domain AI" &bull; Mô hình CoTMix-AudioCNN tối ưu hóa cho Render.com.
                </p>
            </div>

        </div>

        <!-- TAB 2: DOMAIN SHIFT PIPELINE (KIỂM TRA XUYÊN MÔI TRƯỜNG) -->
        <div id="tab-domain" class="tab-pane hidden mt-6 space-y-6">
            <div class="card-box p-6 space-y-6">
                <div>
                    <h2 class="text-lg font-bold text-white">Kiểm tra & Mô phỏng Thích nghi Xuyên Môi Trường</h2>
                    <p class="text-xs text-slate-400 mt-1">
                        Đánh giá độ bền vững của mô hình khi chuyển đổi từ nguồn phòng thu (Studio Clean) sang các thiết bị thực tế có tạp âm và nén méo dải tần.
                    </p>
                </div>

                <!-- Visual Pipeline Diagram -->
                <div class="grid grid-cols-1 md:grid-cols-5 items-center gap-3">
                    <div class="md:col-span-2 bg-[#121c33] p-4 rounded-xl border border-blue-900/50">
                        <div class="flex items-center space-x-2 text-blue-400 text-xs font-bold">
                            <span class="w-2 h-2 rounded-full bg-blue-400"></span>
                            <span>Nguồn &bull; Studio</span>
                        </div>
                        <h4 class="text-white font-bold text-sm mt-1">Âm thanh sạch (Clean)</h4>
                        <p class="text-xs text-slate-400 mt-1">Chất lượng phòng thu số, không lẫn tạp âm, tỉ số SNR cao.</p>
                    </div>

                    <div class="md:col-span-1 flex justify-center py-2">
                        <div class="w-8 h-8 rounded-full bg-blue-600/20 text-blue-400 border border-blue-500/40 flex items-center justify-center">
                            ➔
                        </div>
                    </div>

                    <div class="md:col-span-2 bg-[#1c142b] p-4 rounded-xl border border-purple-900/50">
                        <div class="flex items-center space-x-2 text-purple-400 text-xs font-bold">
                            <span class="w-2 h-2 rounded-full bg-purple-400"></span>
                            <span>Đích &bull; Thiết bị thực tế</span>
                        </div>
                        <h4 class="text-white font-bold text-sm mt-1">Nhiễu nền / Nén âm</h4>
                        <p class="text-xs text-slate-400 mt-1">Micro smartphone dải hẹp, tiếng ồn đường phố, quán cafe, dội phòng.</p>
                    </div>
                </div>

                <!-- Domain Filter Simulator Buttons -->
                <div>
                    <h3 class="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3">Thử Nghiệm Bộ Lọc Nhiễu Âm Học</h3>
                    <div class="grid grid-cols-1 md:grid-cols-4 gap-3">
                        <button onclick="applyFilter('clean')" id="filter-btn-clean" class="p-3.5 rounded-xl border border-blue-500 bg-blue-950/40 text-left transition">
                            <div class="text-xs font-bold text-blue-400">1. Clean Studio</div>
                            <p class="text-[11px] text-slate-400 mt-0.5">Không hiệu ứng méo tiếng.</p>
                        </button>
                        <button onclick="applyFilter('phone')" id="filter-btn-phone" class="p-3.5 rounded-xl border border-[#1e2d4d] bg-[#121c33] text-left hover:border-slate-600 transition">
                            <div class="text-xs font-bold text-purple-400">2. Loa điện thoại</div>
                            <p class="text-[11px] text-slate-400 mt-0.5">Lọc thông dải hẹp 400Hz - 3.5kHz.</p>
                        </button>
                        <button onclick="applyFilter('cafe')" id="filter-btn-cafe" class="p-3.5 rounded-xl border border-[#1e2d4d] bg-[#121c33] text-left hover:border-slate-600 transition">
                            <div class="text-xs font-bold text-amber-400">3. Quán Cafe</div>
                            <p class="text-[11px] text-slate-400 mt-0.5">Thêm nhiễu nền dải trầm trung.</p>
                        </button>
                        <button onclick="applyFilter('reverb')" id="filter-btn-reverb" class="p-3.5 rounded-xl border border-[#1e2d4d] bg-[#121c33] text-left hover:border-slate-600 transition">
                            <div class="text-xs font-bold text-emerald-400">4. Vang vọng phòng</div>
                            <p class="text-[11px] text-slate-400 mt-0.5">Mô phỏng phản xạ âm học kín.</p>
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 3: SETTINGS & TELEMETRY -->
        <div id="tab-settings" class="tab-pane hidden mt-6 space-y-6">
            <div class="card-box p-6 space-y-6">
                <div>
                    <h2 class="text-lg font-bold text-white">Cấu hình hệ thống & Telemetry Render</h2>
                    <p class="text-xs text-slate-400 mt-1">Thông số hoạt động và tình trạng bộ nhớ RAM trên máy chủ Render.com.</p>
                </div>

                <div class="max-w-xl space-y-4">
                    <div>
                        <label class="block text-xs font-semibold text-slate-300 mb-1">Mô hình AI Đang Phục Vụ (Backbone)</label>
                        <div class="p-3 bg-[#121c33] rounded-xl border border-[#1e2d4d] flex items-center justify-between text-xs">
                            <div>
                                <span class="font-bold text-white">AudioCNN (CoTMix Head)</span>
                                <p class="text-[11px] text-emerald-400 mt-0.5">Tối ưu hoàn hảo cho Render Free Tier (&lt;40MB RAM)</p>
                            </div>
                            <span class="px-2.5 py-1 rounded-full bg-emerald-950 text-emerald-400 border border-emerald-800 text-[10px] font-bold">Active</span>
                        </div>
                    </div>

                    <div class="p-4 bg-[#121c33] rounded-xl border border-[#1e2d4d] space-y-2 text-xs">
                        <div class="flex justify-between text-slate-400">
                            <span>Render RAM Usage:</span>
                            <span id="telemetry-ram-val" class="font-mono text-emerald-400 font-bold">-- MB / 512 MB</span>
                        </div>
                        <div class="flex justify-between text-slate-400">
                            <span>OpenAPI Swagger Docs:</span>
                            <a href="/docs" target="_blank" class="text-blue-400 underline font-mono">/docs</a>
                        </div>
                        <div class="flex justify-between text-slate-400">
                            <span>Health Check Endpoint:</span>
                            <a href="/health" target="_blank" class="text-blue-400 underline font-mono">/health</a>
                        </div>
                    </div>
                </div>
            </div>
        </div>

    </main>

    <!-- CORE HTML5 AUDIO ELEMENT -->
    <audio id="global-audio-player" class="hidden"></audio>

    <!-- SCRIPT APPLICATION -->
    <script>
        // State variables
        let currentMode = 'upload'; // 'upload' or 'record'
        let currentAudioUrl = null;
        let currentLoadedGenre = null;
        let currentFileObj = null;
        let isPlaying = false;
        let isRecording = false;

        // Web Audio PCM Recorder Variables
        let audioContext = null;
        let micStream = null;
        let mediaStreamSource = null;
        let scriptProcessor = null;
        let pcmBuffer = [];
        let recordInterval = null;
        let recordSeconds = 0;

        const audioPlayer = document.getElementById('global-audio-player');
        const playBtn = document.getElementById('main-play-btn');
        const playSvg = document.getElementById('main-play-svg');
        const timeDisplay = document.getElementById('player-time-display');
        const dotsContainer = document.getElementById('timeline-dots-container');
        const fileInput = document.getElementById('real-file-input');

        // Genre Emoji Map
        const genreEmojis = {
            'Jazz': '🎷',
            'Blues': '🎸',
            'Classical': '🎻',
            'Rock': '⚡',
            'Pop': '🎧',
            'Hiphop': '🎤',
            'Metal': '🤘',
            'Disco': '🕺',
            'Reggae': '🌴',
            'Country': '🪕'
        };

        // Initialize page without prefilling Jazz
        window.addEventListener('DOMContentLoaded', () => {
            initTimelineDots(32);
            loadHistoryFromStorage();
            fetchTelemetry();
        });

        // Initialize Timeline Dots
        function initTimelineDots(count) {
            dotsContainer.innerHTML = '';
            for (let i = 0; i < count; i++) {
                const dot = document.createElement('div');
                dot.className = 'timeline-dot cursor-pointer';
                const h = Math.floor(Math.sin(i * 0.25) * 8 + 10);
                dot.style.height = h + 'px';
                dot.onclick = () => seekAudio(i / count);
                dotsContainer.appendChild(dot);
            }
        }

        function updateTimelineProgress(percent) {
            const dots = dotsContainer.children;
            const activeIdx = Math.floor(percent * dots.length);
            for (let i = 0; i < dots.length; i++) {
                if (i <= activeIdx) {
                    dots[i].classList.add('active');
                } else {
                    dots[i].classList.remove('active');
                }
            }
        }

        // Mode Switcher (Upload vs Record)
        function setMode(mode) {
            currentMode = mode;
            const recBtn = document.getElementById('mode-record-btn');
            const upBtn = document.getElementById('mode-upload-btn');
            const upPanel = document.getElementById('upload-panel');
            const recPanel = document.getElementById('record-panel');

            if (mode === 'upload') {
                upBtn.className = 'flex items-center justify-center space-x-2 py-2.5 rounded-lg text-xs font-semibold text-white bg-blue-600 shadow-sm transition';
                recBtn.className = 'flex items-center justify-center space-x-2 py-2.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-white transition';
                upPanel.classList.remove('hidden');
                recPanel.classList.add('hidden');
            } else {
                recBtn.className = 'flex items-center justify-center space-x-2 py-2.5 rounded-lg text-xs font-semibold text-white bg-blue-600 shadow-sm transition';
                upBtn.className = 'flex items-center justify-center space-x-2 py-2.5 rounded-lg text-xs font-semibold text-slate-400 hover:text-white transition';
                recPanel.classList.remove('hidden');
                upPanel.classList.add('hidden');
            }
        }

        // Tab Switcher
        function switchTab(tabId) {
            document.querySelectorAll('.tab-pane').forEach(el => el.classList.add('hidden'));
            const target = document.getElementById('tab-' + tabId);
            if (target) target.classList.remove('hidden');

            const tabBtns = ['predict', 'domain', 'history', 'settings'];
            tabBtns.forEach(id => {
                const btn = document.getElementById('nav-' + id);
                if (btn) {
                    if (id === tabId) {
                        btn.className = 'w-full flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-xs font-semibold text-white bg-blue-600 shadow-lg shadow-blue-600/30 transition';
                    } else {
                        btn.className = 'w-full flex items-center space-x-3 px-3.5 py-2.5 rounded-xl text-xs font-medium text-slate-400 hover:text-white hover:bg-slate-800/60 transition';
                    }
                }
            });
        }

        // File Selection
        function triggerFileInput() {
            fileInput.click();
        }

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                const file = fileInput.files[0];
                handleAudioFile(file);
            }
        });

        // Drag & Drop
        const dropZone = document.getElementById('drop-zone');
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('border-blue-500'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('border-blue-500'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('border-blue-500');
            if (e.dataTransfer.files.length > 0) {
                handleAudioFile(e.dataTransfer.files[0]);
            }
        });

        function handleAudioFile(file) {
            currentFileObj = file;
            currentLoadedGenre = null; // Prioritize uploaded file
            const url = URL.createObjectURL(file);
            loadAudio(url, file.name);

            // Show selected pill
            document.getElementById('selected-file-name').innerText = file.name;
            document.getElementById('selected-file-pill').classList.remove('hidden');
            document.getElementById('selected-file-pill').classList.add('flex');
            document.getElementById('drop-title').innerText = 'Đã chọn tệp: ' + file.name;
        }

        // Load Audio into HTML5 player
        function loadAudio(url, title) {
            audioPlayer.src = url;
            currentAudioUrl = url;

            audioPlayer.onloadedmetadata = () => {
                const dur = Math.floor(audioPlayer.duration || 0);
                const m = Math.floor(dur / 60);
                const s = String(dur % 60).padStart(2, '0');
                timeDisplay.innerText = `0:00 / ${m}:${s}`;
                updateTimelineProgress(0);
            };

            audioPlayer.ontimeupdate = () => {
                const cur = Math.floor(audioPlayer.currentTime || 0);
                const dur = Math.floor(audioPlayer.duration || 1);
                const cm = Math.floor(cur / 60);
                const cs = String(cur % 60).padStart(2, '0');
                const dm = Math.floor(dur / 60);
                const ds = String(dur % 60).padStart(2, '0');
                timeDisplay.innerText = `${cm}:${cs} / ${dm}:${ds}`;
                updateTimelineProgress(dur > 0 ? (cur / dur) : 0);
            };

            audioPlayer.onended = () => {
                isPlaying = false;
                playSvg.innerHTML = '<path d="M8 5v14l11-7z"/>';
                updateTimelineProgress(0);
            };
        }

        // Play / Pause Toggle
        function togglePlay() {
            if (!audioPlayer.src) {
                showToast('Chưa có tệp âm thanh nào được nạp. Hãy chọn một file, bấm mẫu thử hoặc ghi âm trước.', 'warning');
                return;
            }
            if (isPlaying) {
                audioPlayer.pause();
                isPlaying = false;
                playSvg.innerHTML = '<path d="M8 5v14l11-7z"/>';
            } else {
                audioPlayer.play().catch(e => console.log('Autoplay policy:', e));
                isPlaying = true;
                playSvg.innerHTML = '<path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>';
            }
        }

        function seekAudio(percent) {
            if (audioPlayer.duration) {
                audioPlayer.currentTime = percent * audioPlayer.duration;
            }
        }

        // Preset Sample Picker
        function pickPreset(genre) {
            setMode('upload');
            currentFileObj = null; // Clear manual file, use sample preset
            currentLoadedGenre = genre;
            document.getElementById('selected-file-pill').classList.add('hidden');
            document.getElementById('drop-title').innerText = 'Mẫu thử: ' + genre.toUpperCase();
            loadAudio('/samples/' + genre, genre + '_sample.wav');
        }

        // --- NATIVE 16KHz PCM WAV RECORDER (Guarantees 100% genuine WAV bytes) ---
        async function toggleRecordState() {
            const btn = document.getElementById('record-action-btn');
            const btnText = document.getElementById('record-btn-text');
            const timerEl = document.getElementById('record-timer');
            const statusEl = document.getElementById('record-status-label');
            const pulseRing = document.getElementById('mic-pulse-ring');

            if (!isRecording) {
                try {
                    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    const AudioCtx = window.AudioContext || window.webkitAudioContext;
                    audioContext = new AudioCtx();
                    mediaStreamSource = audioContext.createMediaStreamSource(micStream);
                    scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
                    pcmBuffer = [];
                    recordSeconds = 0;

                    scriptProcessor.onaudioprocess = (e) => {
                        if (!isRecording) return;
                        const channelData = e.inputBuffer.getChannelData(0);
                        pcmBuffer.push(new Float32Array(channelData));
                    };

                    mediaStreamSource.connect(scriptProcessor);
                    scriptProcessor.connect(audioContext.destination);

                    isRecording = true;
                    statusEl.innerText = 'Đang thu âm... Hãy hát hoặc mở nhạc gần mic';
                    btnText.innerText = 'Dừng & Lưu ghi âm';
                    btn.className = 'px-5 py-2.5 rounded-xl text-xs font-semibold bg-slate-700 hover:bg-slate-600 text-white shadow-lg transition flex items-center space-x-1.5';
                    pulseRing.classList.add('animate-pulse');

                    recordInterval = setInterval(() => {
                        recordSeconds++;
                        const m = String(Math.floor(recordSeconds / 60)).padStart(2, '0');
                        const s = String(recordSeconds % 60).padStart(2, '0');
                        timerEl.innerText = `${m}:${s}`;
                        if (recordSeconds >= 15) {
                            toggleRecordState();
                        }
                    }, 1000);

                } catch (err) {
                    showToast('Không thể truy cập microphone: ' + err.message, 'error');
                }
            } else {
                // Stop recording & finalize PCM WAV
                isRecording = false;
                clearInterval(recordInterval);
                pulseRing.classList.remove('animate-pulse');

                if (micStream) {
                    micStream.getTracks().forEach(t => t.stop());
                }
                if (scriptProcessor) scriptProcessor.disconnect();
                if (mediaStreamSource) mediaStreamSource.disconnect();

                // Merge PCM chunks
                let totalLength = 0;
                for (let i = 0; i < pcmBuffer.length; i++) totalLength += pcmBuffer[i].length;
                const mergedFloat32 = new Float32Array(totalLength);
                let offset = 0;
                for (let i = 0; i < pcmBuffer.length; i++) {
                    mergedFloat32.set(pcmBuffer[i], offset);
                    offset += pcmBuffer[i].length;
                }

                // Downsample to 16000 Hz if needed
                const inRate = audioContext ? audioContext.sampleRate : 44100;
                const targetRate = 16000;
                const finalPcm = downsamplePCM(mergedFloat32, inRate, targetRate);

                // Encode to real RIFF/WAVE PCM 16-bit
                const wavBlob = encodeWAV16Bit(finalPcm, targetRate);
                currentFileObj = new File([wavBlob], 'live_mic_record.wav', { type: 'audio/wav' });
                currentLoadedGenre = null;

                const audioUrl = URL.createObjectURL(wavBlob);
                loadAudio(audioUrl, 'Bản ghi âm');

                statusEl.innerText = 'Đã ghi âm xong (sẵn sàng dự đoán)';
                btnText.innerText = 'Bắt đầu ghi âm mới';
                btn.className = 'px-5 py-2.5 rounded-xl text-xs font-semibold bg-rose-600 hover:bg-rose-500 text-white shadow-lg transition flex items-center space-x-1.5';
            }
        }

        function downsamplePCM(buffer, fromRate, toRate) {
            if (fromRate === toRate) return buffer;
            const ratio = fromRate / toRate;
            const newLen = Math.round(buffer.length / ratio);
            const result = new Float32Array(newLen);
            let offsetResult = 0;
            let offsetBuffer = 0;
            while (offsetResult < result.length) {
                const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
                let accum = 0, count = 0;
                for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
                    accum += buffer[i];
                    count++;
                }
                result[offsetResult] = count > 0 ? (accum / count) : 0;
                offsetResult++;
                offsetBuffer = nextOffsetBuffer;
            }
            return result;
        }

        function encodeWAV16Bit(samples, sampleRate) {
            const buffer = new ArrayBuffer(44 + samples.length * 2);
            const view = new DataView(buffer);

            function writeStr(offset, str) {
                for (let i = 0; i < str.length; i++) {
                    view.setUint8(offset + i, str.charCodeAt(i));
                }
            }

            // RIFF identifier
            writeStr(0, 'RIFF');
            view.setUint32(4, 36 + samples.length * 2, true);
            writeStr(8, 'WAVE');

            // fmt sub-chunk
            writeStr(12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true); // Raw PCM
            view.setUint16(22, 1, true); // Mono 1 channel
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * 2, true); // Byte rate
            view.setUint16(32, 2, true); // Block align
            view.setUint16(34, 16, true); // 16 bits

            // data sub-chunk
            writeStr(36, 'data');
            view.setUint32(40, samples.length * 2, true);

            // Write 16-bit PCM integer samples
            let ptr = 44;
            for (let i = 0; i < samples.length; i++, ptr += 2) {
                let s = Math.max(-1, Math.min(1, samples[i]));
                view.setInt16(ptr, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
            }

            return new Blob([buffer], { type: 'audio/wav' });
        }

        // Non-blocking Toast Notification System
        function showToast(message, type = 'info', duration = 5000) {
            const container = document.getElementById('toast-container');
            if (!container) return;

            const toast = document.createElement('div');
            const isError = type === 'error';
            const isWarning = type === 'warning';
            const isSuccess = type === 'success';

            const bgClass = isError 
                ? 'bg-rose-950/95 border-rose-700/80 text-rose-200' 
                : isWarning 
                    ? 'bg-amber-950/95 border-amber-700/80 text-amber-200'
                    : isSuccess 
                        ? 'bg-emerald-950/95 border-emerald-700/80 text-emerald-200'
                        : 'bg-slate-900/95 border-blue-600/80 text-slate-200';

            const icon = isError ? '⚠️' : isWarning ? '⚡' : isSuccess ? '✅' : 'ℹ️';

            toast.className = `pointer-events-auto flex items-start space-x-3 px-4 py-3 rounded-xl border shadow-2xl backdrop-blur-md transition-all duration-300 transform translate-y-2 opacity-0 w-full ${bgClass}`;
            toast.innerHTML = `
                <span class="text-base shrink-0 mt-0.5">${icon}</span>
                <div class="flex-1 text-xs leading-relaxed">${message}</div>
                <button onclick="this.parentElement.remove()" class="text-slate-400 hover:text-white shrink-0 ml-2 text-base leading-none">&times;</button>
            `;

            container.appendChild(toast);
            requestAnimationFrame(() => {
                toast.classList.remove('translate-y-2', 'opacity-0');
            });

            if (duration > 0) {
                setTimeout(() => {
                    toast.classList.add('opacity-0', '-translate-y-2');
                    setTimeout(() => toast.remove(), 300);
                }, duration);
            }
        }

        let isPredicting = false;

        // Execute Prediction with Automatic Cold-Start Recovery
        async function executePrediction() {
            if (isPredicting) return;

            if (!currentFileObj && !currentLoadedGenre) {
                showToast('Vui lòng chọn một tệp âm thanh, ghi âm qua mic hoặc chọn mẫu thử trước khi bấm dự đoán.', 'warning');
                return;
            }

            const runBtn = document.getElementById('run-predict-btn');
            const btnText = document.getElementById('predict-btn-text');
            const spinner = document.getElementById('predict-spinner');
            const btnIcon = document.getElementById('predict-btn-icon');
            const statusPill = document.getElementById('status-pill-badge');

            isPredicting = true;
            if (runBtn) {
                runBtn.disabled = true;
                runBtn.classList.add('opacity-75', 'cursor-not-allowed');
            }
            btnText.innerText = 'Đang phân tích âm học...';
            spinner.classList.remove('hidden');
            btnIcon.classList.add('hidden');
            statusPill.className = 'px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-blue-950/60 text-blue-400 border border-blue-700 animate-pulse';
            statusPill.innerText = 'Đang suy luận';

            const maxRetries = 2;
            let lastErr = null;

            try {
                for (let attempt = 0; attempt <= maxRetries; attempt++) {
                    try {
                        let data = null;
                        if (currentFileObj) {
                            const formData = new FormData();
                            formData.append('file', currentFileObj);
                            const res = await fetch('/predict', { method: 'POST', body: formData });
                            if (res.status === 502 || res.status === 503 || res.status === 504) {
                                if (attempt < maxRetries) {
                                    btnText.innerText = `Đang khởi động máy chủ (${attempt + 1}/${maxRetries})...`;
                                    await new Promise(r => setTimeout(r, 2000));
                                    continue;
                                }
                            }
                            if (!res.ok) {
                                const errData = await res.json().catch(() => ({}));
                                throw new Error(errData.detail || 'Máy chủ trả về mã lỗi ' + res.status);
                            }
                            data = await res.json();
                        } else if (currentLoadedGenre) {
                            const genreKey = currentLoadedGenre.toLowerCase();
                            const res = await fetch('/sample-predict/' + genreKey, { method: 'POST' });
                            if (res.status === 502 || res.status === 503 || res.status === 504) {
                                if (attempt < maxRetries) {
                                    btnText.innerText = `Đang khởi động máy chủ (${attempt + 1}/${maxRetries})...`;
                                    await new Promise(r => setTimeout(r, 2000));
                                    continue;
                                }
                            }
                            if (!res.ok) {
                                const errData = await res.json().catch(() => ({}));
                                throw new Error(errData.detail || 'Máy chủ trả về mã lỗi ' + res.status);
                            }
                            data = await res.json();
                        }

                        if (!data || !data.predicted_genre) {
                            throw new Error('Dữ liệu kết quả không hợp lệ.');
                        }

                        renderPredictionResult(data);
                        addHistoryRecord(data);

                        statusPill.className = 'px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-950/60 text-emerald-400 border border-emerald-800';
                        statusPill.innerText = 'Phân tích xong';
                        lastErr = null;
                        showToast(`Phân tích thành công: Thể loại <b>${data.predicted_genre}</b> (${((data.confidence || 0) * 100).toFixed(1)}%)`, 'success', 3500);
                        break;
                    } catch (err) {
                        lastErr = err;
                        if (attempt < maxRetries && (err.message.includes('502') || err.message.includes('503') || err.message.includes('Failed to fetch'))) {
                            btnText.innerText = `Đang kết nối lại (${attempt + 1}/${maxRetries})...`;
                            await new Promise(r => setTimeout(r, 2000));
                            continue;
                        }
                        break;
                    }
                }

                if (lastErr) {
                    showToast(`<b>Lỗi dự đoán:</b> ${lastErr.message}<br/><span class="text-[11px] text-slate-300/80 mt-1 block">💡 Gợi ý: Trên gói Render miễn phí, nếu máy chủ vừa được đánh thức sau chế độ ngủ đông, vui lòng bấm Dự đoán lại sau 3-5 giây.</span>`, 'error', 7000);
                    statusPill.className = 'px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-rose-950/60 text-rose-400 border border-rose-800';
                    statusPill.innerText = 'Lỗi';
                }
            } finally {
                if (runBtn) {
                    runBtn.disabled = false;
                    runBtn.classList.remove('opacity-75', 'cursor-not-allowed');
                }
                btnText.innerText = 'Dự đoán thể loại âm nhạc';
                spinner.classList.add('hidden');
                btnIcon.classList.remove('hidden');
                isPredicting = false;
            }
        }

        // Render Prediction Result
        function renderPredictionResult(data) {
            const genre = data.predicted_genre || '--';
            document.getElementById('result-genre-title').innerText = genre;
            document.getElementById('genre-avatar-box').innerText = genreEmojis[genre] || '🎵';
            
            const confPct = ((data.confidence || 0) * 100).toFixed(1) + '%';
            document.getElementById('result-conf-pill').innerText = 'Độ tin cậy: ' + confPct;
            document.getElementById('result-conf-pill').className = 'text-xs font-bold text-blue-400 font-mono bg-blue-950/80 px-2 py-0.5 rounded-md border border-blue-700/60';
            document.getElementById('result-sublabel').innerText = data.genre_sublabel || `Nhạc ${genre} • Nhãn hàng đầu`;
            document.getElementById('result-latency').innerText = `${data.latency_ms || 14.2} ms (${data.device || 'CPU'})`;
            document.getElementById('domain-dot').className = 'w-2 h-2 rounded-full bg-emerald-400';
            document.getElementById('domain-text').className = 'text-emerald-300';
            document.getElementById('domain-text').innerText = 'Đã thích nghi';

            // Show probability list & hide placeholder
            document.getElementById('probability-placeholder').classList.add('hidden');
            const listEl = document.getElementById('probability-list');
            listEl.classList.remove('hidden');
            listEl.innerHTML = '';

            const topList = data.top_predictions || [];
            topList.forEach((item, idx) => {
                const pct = (item.confidence * 100).toFixed(1);
                const barColor = idx === 0 ? 'bg-gradient-to-r from-blue-500 to-cyan-400 glow-blue-bar' : 'bg-blue-600/' + Math.max(20, 60 - idx * 20);
                listEl.innerHTML += `
                    <div>
                        <div class="flex justify-between text-xs font-medium mb-1">
                            <span class="${idx === 0 ? 'text-white font-bold' : 'text-slate-300'}">${item.genre}</span>
                            <span class="${idx === 0 ? 'text-blue-400 font-bold' : 'text-slate-400'} font-mono">${pct}%</span>
                        </div>
                        <div class="w-full bg-[#17233f] rounded-full h-2 overflow-hidden">
                            <div class="${barColor} h-2 rounded-full transition-all duration-700" style="width: ${pct}%"></div>
                        </div>
                    </div>
                `;
            });
        }

        // Prediction History (Table)
        function addHistoryRecord(data) {
            const now = new Date();
            const timeStr = now.toLocaleTimeString('vi-VN');
            const historyItem = {
                time: timeStr,
                filename: data.filename || (currentFileObj ? currentFileObj.name : 'audio_sample.wav'),
                genre: data.predicted_genre,
                confidence: ((data.confidence || 0) * 100).toFixed(1) + '%',
                top2: (data.top_predictions && data.top_predictions[1]) ? `${data.top_predictions[1].genre} (${(data.top_predictions[1].confidence*100).toFixed(1)}%)` : '--',
                latency: (data.latency_ms || 15) + ' ms'
            };

            let history = JSON.parse(localStorage.getItem('cotmix_music_history') || '[]');
            history.unshift(historyItem);
            if (history.length > 10) history = history.slice(0, 10);
            localStorage.setItem('cotmix_music_history', JSON.stringify(history));

            renderHistoryTable(history);
        }

        function loadHistoryFromStorage() {
            const history = JSON.parse(localStorage.getItem('cotmix_music_history') || '[]');
            renderHistoryTable(history);
        }

        function renderHistoryTable(history) {
            const tbody = document.getElementById('history-table-body');
            if (!history || history.length === 0) {
                tbody.innerHTML = `
                    <tr id="empty-history-row">
                        <td colspan="6" class="text-center py-6 text-slate-500">
                            Chưa có dự đoán nào. Hãy tải lên tệp âm thanh hoặc bấm "Dự đoán thể loại âm nhạc".
                        </td>
                    </tr>
                `;
                return;
            }

            tbody.innerHTML = '';
            history.forEach(item => {
                tbody.innerHTML += `
                    <tr class="hover:bg-[#121c33]/50 transition">
                        <td class="py-2.5 px-3 font-mono text-slate-400 text-[11px]">${item.time}</td>
                        <td class="py-2.5 px-3 font-medium text-slate-200">${item.filename}</td>
                        <td class="py-2.5 px-3">
                            <span class="px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-blue-950 text-blue-300 border border-blue-800">
                                ${item.genre}
                            </span>
                        </td>
                        <td class="py-2.5 px-3 font-mono font-bold text-emerald-400">${item.confidence}</td>
                        <td class="py-2.5 px-3 text-slate-400 text-[11px]">${item.top2}</td>
                        <td class="py-2.5 px-3 font-mono text-slate-400 text-[11px]">${item.latency}</td>
                    </tr>
                `;
            });
        }

        function clearHistory() {
            localStorage.removeItem('cotmix_music_history');
            renderHistoryTable([]);
        }

        // Domain Filter Simulator
        function applyFilter(type) {
            const filterBtns = ['clean', 'phone', 'cafe', 'reverb'];
            filterBtns.forEach(id => {
                const btn = document.getElementById('filter-btn-' + id);
                if (btn) {
                    if (id === type) {
                        btn.className = 'p-3.5 rounded-xl border border-blue-500 bg-blue-950/40 text-left transition';
                    } else {
                        btn.className = 'p-3.5 rounded-xl border border-[#1e2d4d] bg-[#121c33] text-left hover:border-slate-600 transition';
                    }
                }
            });
        }

        // Telemetry
        async function fetchTelemetry() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                document.getElementById('telemetry-ram-val').innerText = `${data.ram_usage_mb} MB / 512 MB`;
            } catch (e) {
                console.log('Telemetry fetch error:', e);
            }
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)
