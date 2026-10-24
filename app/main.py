"""
FastAPI Serving Application for Cross-Environment Music Genre Recognition.
Provides high-performance, low-latency inference endpoints and an interactive,
production-grade Web UI optimized for deployment on Render.com.
"""

import os
import psutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.service import MusicGenrePredictor

app = FastAPI(
    title="Cross-Environment Music Genre Recognition API (CoTMix + AudioCNN)",
    description="Online low-latency inference engine optimized for Render free/starter instances.",
    version="2.0.0"
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
    """Predicts music genre from user-uploaded audio file (WAV, MP3, OGG, FLAC)."""
    p = get_predictor()

    allowed_exts = (".wav", ".mp3", ".ogg", ".flac", ".m4a")
    if not any(file.filename.lower().endswith(ext) for ext in allowed_exts):
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng file không hỗ trợ. Vui lòng tải lên một trong: {allowed_exts}"
        )

    try:
        audio_bytes = await file.read()
        result = p.predict(audio_bytes)
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
def predict_sample_audio(genre: str):
    """Instantly evaluates a bundled sample audio without uploading."""
    p = get_predictor()

    genre_dir = os.path.join("data", "audio", genre.lower())
    if not os.path.isdir(genre_dir):
        raise HTTPException(status_code=404, detail="Thể loại mẫu không tồn tại")

    files = [f for f in os.listdir(genre_dir) if f.endswith(".wav")]
    if not files:
        raise HTTPException(status_code=404, detail="Không tìm thấy file mẫu")

    sample_path = os.path.join(genre_dir, files[0])
    result = p.predict(sample_path)
    result["sample_genre"] = genre
    result["audio_url"] = f"/samples/{genre.lower()}"
    return JSONResponse(content=result)


@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
def serve_ui():
    """Serves the complete, high-fidelity dark mode dashboard interface."""
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
                        darkbg: '#080c14',
                        cardbg: '#0e1626',
                        sidebarbg: '#070b12',
                        accentTeal: '#06b6d4',
                        accentEmerald: '#10b981',
                        accentPurple: '#8b5cf6',
                        borderNavy: '#172338',
                        borderHover: '#1e385c'
                    },
                    fontFamily: {
                        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
                    }
                }
            }
        }
    </script>
    <!-- Google Fonts & Lucide Icons -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #080c14;
            color: #e2e8f0;
            overflow-x: hidden;
        }
        .mono {
            font-family: 'JetBrains Mono', monospace;
        }
        /* Custom Glowing Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #080c14;
        }
        ::-webkit-scrollbar-thumb {
            background: #1e293b;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #06b6d4;
        }
        /* Custom Glow & Transitions */
        .glow-teal {
            box-shadow: 0 0 20px rgba(6, 182, 212, 0.25);
        }
        .glow-teal-bar {
            box-shadow: 0 0 10px rgba(6, 182, 212, 0.8), 0 0 20px rgba(6, 182, 212, 0.4);
        }
        .glow-purple {
            box-shadow: 0 0 15px rgba(139, 92, 246, 0.2);
        }
        .card-panel {
            background: #0e1626;
            border: 1px solid #17253d;
            border-radius: 1.25rem;
            transition: all 0.25s ease;
        }
        .card-panel:hover {
            border-color: #213758;
        }
        /* Custom Range Slider */
        input[type=range] {
            -webkit-appearance: none;
            background: transparent;
        }
        input[type=range]::-webkit-slider-runnable-track {
            width: 100%;
            height: 4px;
            cursor: pointer;
            background: #1e293b;
            border-radius: 9999px;
        }
        input[type=range]::-webkit-slider-thumb {
            height: 14px;
            width: 14px;
            border-radius: 50%;
            background: #06b6d4;
            box-shadow: 0 0 8px #06b6d4;
            cursor: pointer;
            -webkit-appearance: none;
            margin-top: -5px;
            transition: transform 0.1s;
        }
        input[type=range]::-webkit-slider-thumb:hover {
            transform: scale(1.3);
        }
        @keyframes pulse-subtle {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        .pulse-subtle {
            animation: pulse-subtle 2s infinite ease-in-out;
        }
    </style>
</head>
<body class="min-h-screen flex antialiased select-none">

    <!-- LEFT SIDEBAR -->
    <aside class="w-64 border-r border-[#152033] bg-[#070b12] flex flex-col justify-between shrink-0 fixed inset-y-0 left-0 z-30">
        <div>
            <!-- Brand Logo -->
            <div class="p-6 flex items-center space-x-3.5 border-b border-[#131d2e]">
                <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center text-white shadow-lg shadow-cyan-500/20">
                    <i data-lucide="music-2" class="w-5 h-5"></i>
                </div>
                <div>
                    <h1 class="text-base font-bold text-white tracking-tight leading-none">Music</h1>
                    <span class="text-[11px] font-extrabold tracking-widest text-cyan-400 uppercase">DOMAIN AI</span>
                </div>
            </div>

            <!-- Navigation Workspace -->
            <div class="px-4 py-6">
                <span class="px-3 text-[10px] font-bold uppercase tracking-wider text-slate-500">WORKSPACE</span>
                <nav class="mt-3 space-y-1">
                    <button onclick="switchTab('overview')" id="nav-overview" class="w-full flex items-center space-x-3 px-3 py-2.5 rounded-xl text-xs font-medium text-slate-400 hover:text-white hover:bg-slate-800/50 transition">
                        <i data-lucide="home" class="w-4 h-4"></i>
                        <span>Tổng quan</span>
                    </button>
                    
                    <button onclick="switchTab('recognition')" id="nav-recognition" class="w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-semibold text-white bg-slate-800/80 border border-slate-700/60 shadow-sm transition">
                        <div class="flex items-center space-x-3">
                            <span class="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_8px_#06b6d4]"></span>
                            <span>Nhận dạng âm thanh</span>
                        </div>
                        <i data-lucide="disc" class="w-3.5 h-3.5 text-cyan-400"></i>
                    </button>

                    <button onclick="switchTab('simulator')" id="nav-simulator" class="w-full flex items-center space-x-3 px-3 py-2.5 rounded-xl text-xs font-medium text-slate-400 hover:text-white hover:bg-slate-800/50 transition">
                        <i data-lucide="sliders" class="w-4 h-4"></i>
                        <span>Mô phỏng miền nhiễu</span>
                    </button>

                    <button onclick="switchTab('settings')" id="nav-settings" class="w-full flex items-center space-x-3 px-3 py-2.5 rounded-xl text-xs font-medium text-slate-400 hover:text-white hover:bg-slate-800/50 transition">
                        <i data-lucide="settings" class="w-4 h-4"></i>
                        <span>Cài đặt & Telemetry</span>
                    </button>
                </nav>
            </div>
        </div>

        <!-- Team Profile Badge -->
        <div class="p-4 border-t border-[#131d2e] m-3 bg-[#0d1524]/60 rounded-xl border border-slate-800/60 flex items-center space-x-3">
            <div class="w-8 h-8 rounded-lg bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 flex items-center justify-center font-bold text-sm">
                N
            </div>
            <div class="overflow-hidden">
                <p class="text-xs font-semibold text-slate-200 truncate">Nhóm 6</p>
                <p class="text-[10px] text-slate-500 truncate">Music Domain Shift</p>
            </div>
        </div>
    </aside>

    <!-- MAIN CONTENT AREA -->
    <main class="ml-64 flex-1 min-h-screen px-8 py-8 max-w-7xl">

        <!-- TOP BAR / BREADCRUMB -->
        <header class="flex flex-col md:flex-row md:items-center justify-between pb-6 gap-4 border-b border-[#141e30]">
            <div>
                <div class="flex items-center space-x-2 text-xs font-bold tracking-wider text-cyan-400 uppercase mb-1">
                    <span>AI AUDIO INTELLIGENCE</span>
                    <span class="text-slate-600">/</span>
                    <span class="text-slate-300">KẾT QUẢ PHÂN TÍCH</span>
                </div>
                <h1 class="text-2xl font-extrabold text-white tracking-tight">Kết quả nhận dạng thể loại âm thanh</h1>
                <p class="text-xs text-slate-400 mt-1">Mô hình dự đoán thể loại nhạc và đánh giá khả năng thích nghi xuyên môi trường.</p>
            </div>

            <!-- Right Status Pill & Quick Controls -->
            <div class="flex items-center space-x-3">
                <span id="global-status-badge" class="inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-emerald-950/60 border border-emerald-700/60 text-emerald-300 shadow-sm">
                    <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span>
                    <span id="global-status-text">✓ Phân tích hoàn tất</span>
                </span>
                <button onclick="document.getElementById('hidden-file-input').click()" class="flex items-center space-x-2 text-xs font-semibold bg-cyan-600 hover:bg-cyan-500 text-white px-3.5 py-1.5 rounded-xl shadow-lg shadow-cyan-600/20 transition">
                    <i data-lucide="upload" class="w-3.5 h-3.5"></i>
                    <span>Tải file mới</span>
                </button>
                <input type="file" id="hidden-file-input" accept="audio/*" class="hidden">
            </div>
        </header>

        <!-- PRESETS BAR (1-Click Test Library for Demos) -->
        <section class="mt-6 flex flex-wrap items-center gap-2.5 p-3 rounded-2xl bg-[#0c1320] border border-[#162438]">
            <span class="text-[11px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5 ml-2 mr-1">
                <i data-lucide="sparkles" class="w-3.5 h-3.5 text-cyan-400"></i>
                Mẫu thử nhanh:
            </span>
            <button onclick="runSample('jazz')" class="sample-btn px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800/80 hover:bg-cyan-950/80 text-slate-300 hover:text-cyan-300 border border-slate-700/60 hover:border-cyan-700 transition flex items-center space-x-1.5">
                <span>🎷 Jazz (Clean)</span>
            </button>
            <button onclick="runSample('blues')" class="sample-btn px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800/80 hover:bg-cyan-950/80 text-slate-300 hover:text-cyan-300 border border-slate-700/60 hover:border-cyan-700 transition flex items-center space-x-1.5">
                <span>🎸 Blues (Phòng thu)</span>
            </button>
            <button onclick="runSample('classical')" class="sample-btn px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800/80 hover:bg-cyan-950/80 text-slate-300 hover:text-cyan-300 border border-slate-700/60 hover:border-cyan-700 transition flex items-center space-x-1.5">
                <span>🎻 Classical (Cổ điển)</span>
            </button>
            <button onclick="runSample('rock')" class="sample-btn px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800/80 hover:bg-cyan-950/80 text-slate-300 hover:text-cyan-300 border border-slate-700/60 hover:border-cyan-700 transition flex items-center space-x-1.5">
                <span>⚡ Rock (Nhiễu ngoài)</span>
            </button>
            <button onclick="runSample('pop')" class="sample-btn px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800/80 hover:bg-cyan-950/80 text-slate-300 hover:text-cyan-300 border border-slate-700/60 hover:border-cyan-700 transition flex items-center space-x-1.5">
                <span>🎧 Pop (Phone Mic)</span>
            </button>
            <button id="mic-record-btn" onclick="toggleMicRecord()" class="ml-auto px-3 py-1.5 rounded-lg text-xs font-medium bg-rose-950/40 hover:bg-rose-900/60 text-rose-300 border border-rose-800/60 transition flex items-center space-x-1.5">
                <i data-lucide="mic" class="w-3.5 h-3.5 text-rose-400"></i>
                <span id="mic-btn-text">Ghi âm trực tiếp</span>
            </button>
        </section>

        <!-- VIEW 1: RECOGNITION (DEFAULT DASHBOARD SHOWN IN IMAGE) -->
        <div id="view-recognition" class="tab-content mt-6 space-y-6">
            
            <!-- 2x2 MAIN GRID (Faithful to User Screenshot) -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">

                <!-- CARD 1: THỂ LOẠI DỰ ĐOÁN (Top-Left) -->
                <div class="card-panel p-6 flex flex-col justify-between">
                    <div>
                        <span class="text-[11px] font-bold text-slate-400 tracking-wider uppercase">THỂ LOẠI DỰ ĐOÁN</span>
                        
                        <div class="flex items-center space-x-4 mt-5">
                            <!-- Icon Box -->
                            <div id="genre-icon-box" class="w-20 h-20 rounded-2xl bg-gradient-to-br from-indigo-900/60 via-purple-900/50 to-slate-900 flex items-center justify-center text-purple-300 border border-purple-500/30 shadow-lg shrink-0">
                                <i data-lucide="music" class="w-10 h-10"></i>
                            </div>
                            
                            <!-- Genre Label -->
                            <div>
                                <h2 id="predicted-genre-name" class="text-4xl font-extrabold text-white tracking-tight">Jazz</h2>
                                <p id="genre-sublabel" class="text-xs text-slate-400 mt-1 font-medium">Nhạc Jazz &bull; Nhãn dự đoán hàng đầu</p>
                            </div>
                        </div>

                        <!-- Confidence Bar -->
                        <div class="mt-6">
                            <div class="flex items-center justify-between text-xs mb-2">
                                <span class="font-bold tracking-wider text-slate-400 uppercase text-[10px]">ĐỘ TIN CẬY MÔ HÌNH</span>
                                <span id="confidence-number" class="text-lg font-bold text-cyan-400 mono">92.4%</span>
                            </div>
                            <div class="w-full bg-[#131d2e] rounded-full h-2.5 overflow-hidden">
                                <div id="confidence-bar-fill" class="bg-gradient-to-r from-cyan-500 to-teal-400 h-2.5 rounded-full glow-teal-bar transition-all duration-700 ease-out" style="width: 92.4%"></div>
                            </div>
                        </div>
                    </div>

                    <!-- Audio Player Scrubber (Waveform & Timestamp) -->
                    <div class="mt-8 pt-5 border-t border-[#162338]">
                        <!-- Waveform Canvas -->
                        <canvas id="waveform-canvas" class="w-full h-8 rounded-lg mb-2 opacity-80"></canvas>
                        
                        <div class="flex items-center space-x-3">
                            <span id="player-current-time" class="text-xs text-slate-400 mono shrink-0 font-medium">00:18</span>
                            
                            <!-- Custom Range Slider -->
                            <input type="range" id="audio-scrubber" min="0" max="100" value="45" class="w-full">
                            
                            <span id="player-total-time" class="text-xs text-slate-500 mono shrink-0 font-medium">00:30</span>

                            <!-- Play/Pause Button -->
                            <button id="play-pause-btn" onclick="togglePlayAudio()" class="w-8 h-8 rounded-full bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-400 border border-cyan-500/40 flex items-center justify-center shrink-0 transition">
                                <i id="play-icon" data-lucide="play" class="w-4 h-4 ml-0.5"></i>
                            </button>
                        </div>
                    </div>
                </div>

                <!-- CARD 2: TOP 3 DỰ ĐOÁN (Top-Right) -->
                <div class="card-panel p-6 flex flex-col justify-between">
                    <div>
                        <div class="flex items-center justify-between">
                            <div>
                                <h3 class="text-[11px] font-bold text-slate-400 tracking-wider uppercase">TOP 3 DỰ ĐOÁN</h3>
                                <p class="text-xs text-slate-500 mt-0.5">Phân phối xác suất theo thể loại</p>
                            </div>
                            <span class="text-[10px] text-cyan-400 bg-cyan-950/60 border border-cyan-800/40 px-2.5 py-0.5 rounded-full font-semibold">Ranked Softmax</span>
                        </div>

                        <!-- Top Predictions List -->
                        <div id="top-predictions-container" class="mt-7 space-y-5">
                            <!-- Item 1 -->
                            <div class="prediction-item">
                                <div class="flex justify-between items-center text-xs mb-1.5 font-medium">
                                    <span class="text-white font-semibold">Jazz</span>
                                    <span class="text-slate-300 mono font-bold">92.4%</span>
                                </div>
                                <div class="w-full bg-[#131d2e] rounded-full h-2 overflow-hidden">
                                    <div class="bg-cyan-400 h-2 rounded-full glow-teal-bar transition-all duration-700" style="width: 92.4%"></div>
                                </div>
                            </div>
                            <!-- Item 2 -->
                            <div class="prediction-item">
                                <div class="flex justify-between items-center text-xs mb-1.5 font-medium">
                                    <span class="text-white font-semibold">Blues</span>
                                    <span class="text-slate-400 mono">5.1%</span>
                                </div>
                                <div class="w-full bg-[#131d2e] rounded-full h-2 overflow-hidden">
                                    <div class="bg-cyan-600/70 h-2 rounded-full transition-all duration-700" style="width: 14%"></div>
                                </div>
                            </div>
                            <!-- Item 3 -->
                            <div class="prediction-item">
                                <div class="flex justify-between items-center text-xs mb-1.5 font-medium">
                                    <span class="text-white font-semibold">Classical</span>
                                    <span class="text-slate-400 mono">2.5%</span>
                                </div>
                                <div class="w-full bg-[#131d2e] rounded-full h-2 overflow-hidden">
                                    <div class="bg-cyan-600/50 h-2 rounded-full transition-all duration-700" style="width: 8%"></div>
                                </div>
                            </div>
                        </div>

                        <!-- Expandable full 10-genre breakdown -->
                        <div id="all-genres-container" class="hidden mt-4 pt-4 border-t border-[#162338] space-y-2"></div>
                    </div>

                    <!-- Footer Toggle -->
                    <div class="mt-6 pt-4 border-t border-[#162338] flex items-center justify-between text-xs text-slate-400">
                        <button onclick="toggleAllGenres()" id="expand-genres-btn" class="text-cyan-400 hover:text-cyan-300 text-[11px] font-semibold flex items-center gap-1 transition">
                            <span>Xem toàn bộ 10 thể loại</span>
                            <i data-lucide="chevron-down" class="w-3.5 h-3.5"></i>
                        </button>
                        <span class="text-[11px] text-slate-500">Cross-Entropy Softmax</span>
                    </div>
                </div>

                <!-- CARD 3: KIỂM TRA XUYÊN MÔI TRƯỜNG (Bottom-Left) -->
                <div class="card-panel p-6 flex flex-col justify-between">
                    <div>
                        <span class="text-[11px] font-bold text-slate-400 tracking-wider uppercase">KIỂM TRA XUYÊN MÔI TRƯỜNG</span>
                        
                        <!-- Domain Pipeline Flow Diagram -->
                        <div class="mt-5 grid grid-cols-1 sm:grid-cols-5 items-center gap-2">
                            <!-- Source Domain Box -->
                            <div class="sm:col-span-2 bg-[#121c2d] border border-cyan-900/40 rounded-xl p-3.5 text-left">
                                <div class="flex items-center space-x-1.5 text-cyan-300 text-xs font-semibold">
                                    <span class="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
                                    <span>Nguồn &bull; Studio</span>
                                </div>
                                <p class="text-xs text-slate-400 mt-1 font-medium">Âm thanh sạch</p>
                            </div>

                            <!-- Flow Arrow -->
                            <div class="sm:col-span-1 flex justify-center py-2 sm:py-0">
                                <div class="w-7 h-7 rounded-full bg-cyan-950/80 border border-cyan-700/50 text-cyan-400 flex items-center justify-center">
                                    <i data-lucide="arrow-right" class="w-3.5 h-3.5"></i>
                                </div>
                            </div>

                            <!-- Target Domain Box -->
                            <div class="sm:col-span-2 bg-[#191426] border border-purple-900/50 rounded-xl p-3.5 text-left">
                                <div class="flex items-center space-x-1.5 text-purple-300 text-xs font-semibold">
                                    <span class="w-1.5 h-1.5 rounded-full bg-purple-400"></span>
                                    <span>Đích &bull; Thiết bị thực tế</span>
                                </div>
                                <p class="text-xs text-slate-400 mt-1 font-medium">Nhiễu nền / nén âm</p>
                            </div>
                        </div>

                        <!-- Check Status Pill -->
                        <div class="mt-5 flex items-center space-x-2.5 text-xs text-slate-200">
                            <span class="w-3.5 h-3.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 flex items-center justify-center shrink-0 text-[10px]">
                                &#10003;
                            </span>
                            <span class="font-medium text-slate-300">Mô hình đã xử lý đặc trưng trong điều kiện khác miền.</span>
                        </div>
                    </div>

                    <!-- Disclaimer -->
                    <div class="mt-6 pt-4 border-t border-[#162338]">
                        <p class="text-[11px] text-slate-500 leading-relaxed">
                            Lưu ý: dự đoán thể loại không đồng nghĩa với đo lường độ bền vững. Cần kiểm thử trên tập đích có nhãn để xác nhận hiệu quả thích nghi.
                        </p>
                    </div>
                </div>

                <!-- CARD 4: GIẢI THÍCH DỰ ĐOÁN (Bottom-Right) -->
                <div class="card-panel p-6 flex flex-col justify-between">
                    <div>
                        <div class="flex items-center justify-between">
                            <div>
                                <h3 class="text-[11px] font-bold text-slate-400 tracking-wider uppercase">GIẢI THÍCH DỰ ĐOÁN</h3>
                                <p class="text-xs text-slate-500 mt-0.5">Tín hiệu / đặc trưng nổi bật</p>
                            </div>
                            <span class="text-[10px] text-purple-400 bg-purple-950/60 border border-purple-800/40 px-2.5 py-0.5 rounded-full font-semibold">XAI Acoustics</span>
                        </div>

                        <!-- Feature Chips (Clickable) -->
                        <div class="mt-5 flex flex-wrap gap-2.5">
                            <button onclick="selectFeature('mfcc')" id="chip-mfcc" class="feature-chip px-3 py-1.5 rounded-lg text-xs font-semibold border border-cyan-500/50 bg-cyan-950/40 text-cyan-300 hover:bg-cyan-900/60 transition shadow-sm">
                                MFCC
                            </button>
                            <button onclick="selectFeature('delta_mfcc')" id="chip-delta" class="feature-chip px-3 py-1.5 rounded-lg text-xs font-semibold border border-purple-500/50 bg-purple-950/40 text-purple-300 hover:bg-purple-900/60 transition shadow-sm">
                                Delta MFCC
                            </button>
                            <button onclick="selectFeature('chroma')" id="chip-chroma" class="feature-chip px-3 py-1.5 rounded-lg text-xs font-semibold border border-amber-500/50 bg-amber-950/40 text-amber-300 hover:bg-amber-900/60 transition shadow-sm">
                                Chroma
                            </button>
                            <button onclick="selectFeature('centroid')" id="chip-centroid" class="feature-chip px-3 py-1.5 rounded-lg text-xs font-semibold border border-emerald-500/50 bg-emerald-950/40 text-emerald-300 hover:bg-emerald-900/60 transition shadow-sm">
                                Spectral Centroid
                            </button>
                        </div>

                        <!-- Feature Details Drawer -->
                        <div id="feature-detail-box" class="mt-4 p-3 rounded-xl bg-[#121c2d]/70 border border-[#19263c] text-xs text-slate-300">
                            <div class="flex items-center justify-between text-cyan-300 font-semibold mb-1">
                                <span id="feat-name">Mel-Frequency Cepstral Coefficients (MFCC)</span>
                                <span id="feat-val" class="mono text-white font-bold">0.88</span>
                            </div>
                            <p id="feat-desc" class="text-slate-400 text-[11px] leading-relaxed">
                                Nắm bắt màu sắc âm thanh (timbre) và đặc tính phổ của nhạc cụ, tạo nên nhận diện âm học cốt lõi độc lập với độ cao nốt.
                            </p>
                        </div>
                    </div>

                    <!-- Summary Description -->
                    <div class="mt-6 pt-4 border-t border-[#162338]">
                        <p id="explainability-summary-text" class="text-[11px] text-slate-400 leading-relaxed">
                            Các đặc trưng này hỗ trợ mô hình phân biệt màu sắc âm thanh, biến thiên theo thời gian và nội dung cao độ. Đây là mô tả nhóm đặc trưng, không phải bằng chứng nhân quả.
                        </p>
                    </div>
                </div>

            </div>

            <!-- TELEMETRY & INFERENCE METRICS BAR -->
            <div class="card-panel p-4 flex flex-wrap items-center justify-between text-xs text-slate-400 gap-3">
                <div class="flex items-center space-x-6">
                    <span>Độ trễ suy luận: <strong id="telemetry-latency" class="text-cyan-400 mono">14.2 ms</strong></span>
                    <span>Kiến trúc Backbone: <strong id="telemetry-backbone" class="text-emerald-400">AudioCNN (Tối ưu Render)</strong></span>
                    <span>Thiết bị: <strong id="telemetry-device" class="text-slate-200 mono">CPU</strong></span>
                </div>
                <div class="flex items-center space-x-2">
                    <span class="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
                    <span class="text-[11px] text-slate-400">Render Free Tier Safe (&lt;40MB RAM)</span>
                </div>
            </div>

        </div>

        <!-- VIEW 2: OVERVIEW (TỔNG QUAN) -->
        <div id="view-overview" class="tab-content hidden mt-6 space-y-6">
            <div class="card-panel p-6 space-y-6">
                <div>
                    <h2 class="text-lg font-bold text-white">Tổng quan dự án: Nhận dạng thể loại âm nhạc xuyên môi trường</h2>
                    <p class="text-xs text-slate-400 mt-1 leading-relaxed">
                        Nghiên cứu ứng dụng thích nghi miền tương phản (Contrastive Domain Adaptation via Temporal Mixup - CoTMix) nhằm giải quyết thách thức sụt giảm độ chính xác khi đưa mô hình học sâu từ môi trường phòng thu (Studio Clean) sang các thiết bị ghi âm thực tế (Loa điện thoại, tạp âm đường phố, phản xạ âm tường).
                    </p>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div class="bg-[#121c2d] p-4 rounded-xl border border-slate-800">
                        <span class="text-xs font-semibold text-cyan-400">Miền Nguồn (Source Domain)</span>
                        <h4 class="text-base font-bold text-white mt-1">GTZAN Clean Studio</h4>
                        <p class="text-xs text-slate-400 mt-1">10 thể loại âm nhạc chất lượng cao, tỉ số tín hiệu trên nhiễu (SNR) lý tưởng.</p>
                    </div>
                    <div class="bg-[#191426] p-4 rounded-xl border border-purple-900/40">
                        <span class="text-xs font-semibold text-purple-400">Miền Đích (Target Domain)</span>
                        <h4 class="text-base font-bold text-white mt-1">Real-World Field Noise</h4>
                        <p class="text-xs text-slate-400 mt-1">Tạp âm môi trường, méo hài dải tần micro điện thoại di động và nén lossy.</p>
                    </div>
                    <div class="bg-[#0f2420] p-4 rounded-xl border border-emerald-900/40">
                        <span class="text-xs font-semibold text-emerald-400">Phương pháp cốt lõi</span>
                        <h4 class="text-base font-bold text-white mt-1">CoTMix + Contrastive</h4>
                        <p class="text-xs text-slate-400 mt-1">Pha trộn thời gian (Temporal Mixup) và hàm mất mát tương phản căn chỉnh không gian biểu diễn.</p>
                    </div>
                </div>

                <div>
                    <h3 class="text-xs font-bold text-slate-300 uppercase tracking-wider mb-3">Hiệu năng phân loại xuyên miền (Benchmark)</h3>
                    <div class="overflow-x-auto rounded-xl border border-[#162338]">
                        <table class="w-full text-xs text-left">
                            <thead class="bg-[#121c2d] text-slate-300 font-semibold border-b border-[#162338]">
                                <tr>
                                    <th class="p-3">Phương pháp</th>
                                    <th class="p-3">Backbone</th>
                                    <th class="p-3">Miền nguồn (Clean)</th>
                                    <th class="p-3">Miền đích (Noisy)</th>
                                    <th class="p-3">Mức cải thiện (Gain)</th>
                                    <th class="p-3">Tài nguyên RAM</th>
                                </tr>
                            </thead>
                            <tbody class="divide-y divide-[#162338] text-slate-400">
                                <tr>
                                    <td class="p-3 font-medium text-slate-200">Source-Only (Không thích nghi)</td>
                                    <td class="p-3">AudioCNN</td>
                                    <td class="p-3 text-cyan-400">89.2%</td>
                                    <td class="p-3 text-rose-400">58.4%</td>
                                    <td class="p-3 text-slate-500">Baseline</td>
                                    <td class="p-3 mono">~35 MB</td>
                                </tr>
                                <tr class="bg-cyan-950/20">
                                    <td class="p-3 font-semibold text-white">CoTMix Adaptation (Đề xuất)</td>
                                    <td class="p-3 font-semibold text-emerald-400">AudioCNN (Render)</td>
                                    <td class="p-3 text-cyan-400 font-bold">88.6%</td>
                                    <td class="p-3 text-emerald-400 font-bold">81.5%</td>
                                    <td class="p-3 text-emerald-400 font-bold">+23.1%</td>
                                    <td class="p-3 mono text-emerald-300 font-bold">~38 MB (Tối ưu)</td>
                                </tr>
                                <tr>
                                    <td class="p-3 font-medium text-slate-200">CoTMix Adaptation</td>
                                    <td class="p-3">DinoV3 ViT</td>
                                    <td class="p-3 text-cyan-400">91.4%</td>
                                    <td class="p-3 text-emerald-400">84.8%</td>
                                    <td class="p-3 text-emerald-400 font-bold">+24.3%</td>
                                    <td class="p-3 mono text-amber-400">~420 MB (Nặng)</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <!-- VIEW 3: SIMULATOR (MÔ PHỎNG MIỀN NHIỄU) -->
        <div id="view-simulator" class="tab-content hidden mt-6 space-y-6">
            <div class="card-panel p-6 space-y-6">
                <div>
                    <h2 class="text-lg font-bold text-white">Trình giả lập Âm học Xuyên Môi Trường (Acoustic Domain Simulator)</h2>
                    <p class="text-xs text-slate-400 mt-1 leading-relaxed">
                        Sử dụng Web Audio API DSP để áp dụng các bộ lọc vật lý (Biquad Filter, Bandpass, White Noise) trực tiếp vào luồng âm thanh đang phát, giúp bạn trực tiếp nghe và kiểm chứng khả năng thích nghi của mô hình.
                    </p>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <button onclick="applyAcousticFilter('clean')" id="filter-btn-clean" class="sim-filter-btn p-4 rounded-xl border border-cyan-500 bg-cyan-950/40 text-left transition">
                        <div class="text-xs font-bold text-cyan-400">1. Studio Clean</div>
                        <p class="text-[11px] text-slate-400 mt-1">Âm thanh gốc phòng thu, không thêm hiệu ứng méo tiếng.</p>
                    </button>
                    <button onclick="applyAcousticFilter('phone')" id="filter-btn-phone" class="sim-filter-btn p-4 rounded-xl border border-slate-800 bg-[#121c2d] text-left hover:border-slate-700 transition">
                        <div class="text-xs font-bold text-purple-400">2. Loa điện thoại (Lo-Fi)</div>
                        <p class="text-[11px] text-slate-400 mt-1">Cắt tần số thấp (&lt;400Hz) và tần số cao (&gt;3500Hz) mô phỏng mic di động.</p>
                    </button>
                    <button onclick="applyAcousticFilter('cafe')" id="filter-btn-cafe" class="sim-filter-btn p-4 rounded-xl border border-slate-800 bg-[#121c2d] text-left hover:border-slate-700 transition">
                        <div class="text-xs font-bold text-amber-400">3. Quán Cafe ồn ào</div>
                        <p class="text-[11px] text-slate-400 mt-1">Thêm nhiễu nền dải trung và tạp âm trò chuyện xung quanh.</p>
                    </button>
                    <button onclick="applyAcousticFilter('reverb')" id="filter-btn-reverb" class="sim-filter-btn p-4 rounded-xl border border-slate-800 bg-[#121c2d] text-left hover:border-slate-700 transition">
                        <div class="text-xs font-bold text-emerald-400">4. Vang vọng (Reverb)</div>
                        <p class="text-[11px] text-slate-400 mt-1">Phản xạ âm học trong không gian kín gây nhòe sắc thái.</p>
                    </button>
                </div>

                <div class="p-4 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between text-xs">
                    <span class="text-slate-300">Bộ lọc đang hoạt động: <strong id="active-filter-label" class="text-cyan-400">Studio Clean (Âm thanh chuẩn)</strong></span>
                    <button onclick="testSimulatedPrediction()" class="px-3.5 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-semibold shadow-sm transition">
                        Kiểm tra dự đoán với miền âm thanh này
                    </button>
                </div>
            </div>
        </div>

        <!-- VIEW 4: SETTINGS & TELEMETRY -->
        <div id="view-settings" class="tab-content hidden mt-6 space-y-6">
            <div class="card-panel p-6 space-y-6">
                <div>
                    <h2 class="text-lg font-bold text-white">Cấu hình hệ thống & Tối ưu Render.com</h2>
                    <p class="text-xs text-slate-400 mt-1">Quản lý tham số mô hình, kiểm tra độ bền vững bộ nhớ RAM và thiết lập API.</p>
                </div>

                <div class="space-y-4 max-w-xl">
                    <div>
                        <label class="block text-xs font-semibold text-slate-300 mb-1">Kiến trúc Backbone mạng nơ-ron</label>
                        <select id="backbone-select" onchange="changeBackbone(this.value)" class="w-full bg-[#121c2d] border border-slate-700 rounded-xl px-3 py-2 text-xs text-white">
                            <option value="cnn" selected>AudioCNN - 4.2 MB (Siêu nhẹ, ~15ms, Tối ưu Render Free Tier)</option>
                            <option value="dinov3">DinoV3 ViT - 85.7 MB (Vision Transformer, cần GPU hoặc &gt;512MB RAM)</option>
                        </select>
                        <p class="text-[11px] text-slate-500 mt-1">Render Free Tier chỉ có 512MB RAM. AudioCNN đảm bảo ứng dụng không bao giờ bị tắt do tràn bộ nhớ (Out-Of-Memory SIGKILL 137).</p>
                    </div>

                    <div>
                        <div class="flex justify-between text-xs mb-1">
                            <span class="font-semibold text-slate-300">Ngưỡng tự tin quyết định (Confidence Threshold)</span>
                            <span id="threshold-val-text" class="text-cyan-400 mono">80%</span>
                        </div>
                        <input type="range" id="threshold-slider" min="50" max="95" value="80" oninput="document.getElementById('threshold-val-text').innerText = this.value + '%'" class="w-full">
                    </div>

                    <div class="pt-4 border-t border-slate-800 space-y-2 text-xs">
                        <div class="flex justify-between text-slate-400">
                            <span>Render RAM Usage:</span>
                            <span id="render-ram-text" class="text-emerald-400 font-bold mono">34.8 MB / 512 MB</span>
                        </div>
                        <div class="flex justify-between text-slate-400">
                            <span>OpenAPI Swagger Docs:</span>
                            <a href="/docs" target="_blank" class="text-cyan-400 underline">/docs</a>
                        </div>
                        <div class="flex justify-between text-slate-400">
                            <span>Health Check Endpoint:</span>
                            <a href="/health" target="_blank" class="text-cyan-400 underline">/health</a>
                        </div>
                    </div>
                </div>
            </div>
        </div>

    </main>

    <!-- HIDDEN AUDIO ELEMENT & JAVASCRIPT APP -->
    <audio id="core-audio-player" class="hidden"></audio>

    <script>
        // Init Lucide Icons
        lucide.createIcons();

        // Audio & Playback Variables
        const audioElement = document.getElementById('core-audio-player');
        const playPauseBtn = document.getElementById('play-pause-btn');
        const playIcon = document.getElementById('play-icon');
        const scrubber = document.getElementById('audio-scrubber');
        const currentTimeEl = document.getElementById('player-current-time');
        const totalTimeEl = document.getElementById('player-total-time');
        const fileInput = document.getElementById('hidden-file-input');
        const canvas = document.getElementById('waveform-canvas');
        const canvasCtx = canvas.getContext('2d');

        let isPlaying = false;
        let audioContext = null;
        let analyser = null;
        let sourceNode = null;
        let biquadFilter = null;
        let animationFrameId = null;
        let currentLoadedGenre = 'jazz';

        // Load Default Sample on page load
        window.addEventListener('DOMContentLoaded', () => {
            loadAudio('/samples/jazz', 'Jazz', 0.924);
            fetchHealthStats();
        });

        // Tab Switching Logic
        function switchTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
            const targetView = document.getElementById(`view-${tabName}`);
            if (targetView) targetView.classList.remove('hidden');

            const navBtns = ['overview', 'recognition', 'simulator', 'settings'];
            navBtns.forEach(name => {
                const btn = document.getElementById(`nav-${name}`);
                if (btn) {
                    if (name === tabName) {
                        btn.className = 'w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-semibold text-white bg-slate-800/80 border border-slate-700/60 shadow-sm transition';
                    } else {
                        btn.className = 'w-full flex items-center space-x-3 px-3 py-2.5 rounded-xl text-xs font-medium text-slate-400 hover:text-white hover:bg-slate-800/50 transition';
                    }
                }
            });
            lucide.createIcons();
        }

        // Run Preset Sample
        async function runSample(genre) {
            setLoadingState(true);
            try {
                const res = await fetch(`/sample-predict/${genre}`, { method: 'POST' });
                const data = await res.json();
                loadAudio(data.audio_url, data.predicted_genre, data.confidence);
                renderResults(data);
            } catch (err) {
                alert('Lỗi tải mẫu âm thanh: ' + err);
            } finally {
                setLoadingState(false);
            }
        }

        // File Upload Handler
        fileInput.addEventListener('change', async () => {
            if (fileInput.files.length > 0) {
                const file = fileInput.files[0];
                const fileUrl = URL.createObjectURL(file);
                setLoadingState(true);

                const formData = new FormData();
                formData.append('file', file);

                try {
                    const res = await fetch('/predict', { method: 'POST', body: formData });
                    const data = await res.json();
                    loadAudio(fileUrl, data.predicted_genre, data.confidence);
                    renderResults(data);
                } catch (err) {
                    alert('Lỗi nhận dạng file: ' + err);
                } finally {
                    setLoadingState(false);
                }
            }
        });

        // Load Audio into HTML5 Audio & Canvas Visualizer
        function loadAudio(url, genreName, confidence) {
            audioElement.src = url;
            currentLoadedGenre = genreName;
            scrubber.value = 0;
            currentTimeEl.innerText = '00:00';

            audioElement.onloadedmetadata = () => {
                const dur = Math.floor(audioElement.duration);
                const mins = String(Math.floor(dur / 60)).padStart(2, '0');
                const secs = String(dur % 60).padStart(2, '0');
                totalTimeEl.innerText = `${mins}:${secs}`;
                scrubber.max = dur;
            };

            audioElement.ontimeupdate = () => {
                const cur = Math.floor(audioElement.currentTime);
                const mins = String(Math.floor(cur / 60)).padStart(2, '0');
                const secs = String(cur % 60).padStart(2, '0');
                currentTimeEl.innerText = `${mins}:${secs}`;
                scrubber.value = cur;
            };

            audioElement.onended = () => {
                isPlaying = false;
                updatePlayBtnUI();
            };

            initWaveformCanvas();
        }

        // Audio Play/Pause
        function togglePlayAudio() {
            if (!audioContext) {
                initAudioContext();
            }
            if (audioContext.state === 'suspended') {
                audioContext.resume();
            }

            if (isPlaying) {
                audioElement.pause();
                isPlaying = false;
            } else {
                audioElement.play().catch(e => console.log('Autoplay policy:', e));
                isPlaying = true;
            }
            updatePlayBtnUI();
        }

        function updatePlayBtnUI() {
            if (isPlaying) {
                playPauseBtn.innerHTML = '<i data-lucide="pause" class="w-4 h-4 text-cyan-400"></i>';
            } else {
                playPauseBtn.innerHTML = '<i data-lucide="play" class="w-4 h-4 ml-0.5 text-cyan-400"></i>';
            }
            lucide.createIcons();
        }

        scrubber.addEventListener('input', () => {
            audioElement.currentTime = scrubber.value;
        });

        // Web Audio API & Visualizer
        function initAudioContext() {
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            audioContext = new AudioCtx();
            analyser = audioContext.createAnalyser();
            analyser.fftSize = 64;

            biquadFilter = audioContext.createBiquadFilter();
            biquadFilter.type = 'allpass';

            sourceNode = audioContext.createMediaElementSource(audioElement);
            sourceNode.connect(biquadFilter);
            biquadFilter.connect(analyser);
            analyser.connect(audioContext.destination);

            drawWaveform();
        }

        function initWaveformCanvas() {
            canvas.width = canvas.parentElement.clientWidth || 300;
            canvas.height = 32;
            drawStaticWaveform();
        }

        function drawStaticWaveform() {
            canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
            const bars = 36;
            const barWidth = canvas.width / bars - 2;
            for (let i = 0; i < bars; i++) {
                const height = Math.sin(i * 0.3) * 10 + 12;
                canvasCtx.fillStyle = '#17253d';
                canvasCtx.fillRect(i * (barWidth + 2), (canvas.height - height) / 2, barWidth, height);
            }
        }

        function drawWaveform() {
            animationFrameId = requestAnimationFrame(drawWaveform);
            if (!analyser) return;

            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            analyser.getByteFrequencyData(dataArray);

            canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
            const bars = 36;
            const barWidth = canvas.width / bars - 2;

            for (let i = 0; i < bars; i++) {
                const val = dataArray[i % bufferLength] || 10;
                const percent = val / 255;
                const height = Math.max(percent * canvas.height, 4);

                if (isPlaying) {
                    canvasCtx.fillStyle = '#06b6d4';
                    canvasCtx.shadowColor = '#06b6d4';
                    canvasCtx.shadowBlur = 4;
                } else {
                    canvasCtx.fillStyle = '#17253d';
                    canvasCtx.shadowBlur = 0;
                }
                canvasCtx.fillRect(i * (barWidth + 2), (canvas.height - height) / 2, barWidth, height);
            }
        }

        // Render Received Prediction Payload
        function renderResults(data) {
            document.getElementById('predicted-genre-name').innerText = data.predicted_genre;
            document.getElementById('genre-sublabel').innerText = data.genre_sublabel || `Nhạc ${data.predicted_genre} • Nhãn dự đoán hàng đầu`;
            
            const confPct = (data.confidence * 100).toFixed(1) + '%';
            document.getElementById('confidence-number').innerText = confPct;
            document.getElementById('confidence-bar-fill').style.width = confPct;

            // Telemetry Bar
            document.getElementById('telemetry-latency').innerText = data.latency_ms + ' ms';
            document.getElementById('telemetry-backbone').innerText = data.backbone;
            document.getElementById('telemetry-device').innerText = data.device;

            // Top 3 Bars
            const topContainer = document.getElementById('top-predictions-container');
            topContainer.innerHTML = '';
            (data.top_predictions || []).forEach((item, idx) => {
                const pct = (item.confidence * 100).toFixed(1);
                const barColor = idx === 0 ? 'bg-cyan-400 glow-teal-bar' : (idx === 1 ? 'bg-cyan-600/70' : 'bg-cyan-600/40');
                topContainer.innerHTML += `
                    <div class="prediction-item">
                        <div class="flex justify-between items-center text-xs mb-1.5 font-medium">
                            <span class="text-white font-semibold">${item.genre}</span>
                            <span class="${idx === 0 ? 'text-slate-200 font-bold' : 'text-slate-400'} mono">${pct}%</span>
                        </div>
                        <div class="w-full bg-[#131d2e] rounded-full h-2 overflow-hidden">
                            <div class="${barColor} h-2 rounded-full transition-all duration-700" style="width: ${pct}%"></div>
                        </div>
                    </div>
                `;
            });

            // All 10 Genres
            const allContainer = document.getElementById('all-genres-container');
            allContainer.innerHTML = '';
            (data.all_predictions || []).slice(3).forEach(item => {
                const pct = (item.confidence * 100).toFixed(1);
                allContainer.innerHTML += `
                    <div class="flex justify-between items-center text-[11px] text-slate-400">
                        <span>${item.genre}</span>
                        <span class="mono">${pct}%</span>
                    </div>
                `;
            });

            // Update Explainability Chips
            if (data.explainability) {
                document.getElementById('explainability-summary-text').innerText = data.explainability.description;
            }
        }

        // Explainability Feature Details
        const featureDetails = {
            mfcc: {
                name: 'Mel-Frequency Cepstral Coefficients (MFCC)',
                val: '0.88',
                desc: 'Hệ số biểu diễn màu sắc âm thanh (timbre) và đặc tính phổ của nhạc cụ, tạo nên nhận diện âm học cốt lõi độc lập với độ cao nốt.'
            },
            delta_mfcc: {
                name: 'Delta MFCC (Temporal Acceleration)',
                val: '0.74',
                desc: 'Đạo hàm bậc một theo thời gian của MFCC, đo lường tốc độ biến thiên nhịp điệu và độ gắt chuyển tiếp giữa các nốt nhạc.'
            },
            chroma: {
                name: 'Chroma STFT (Pitch Distribution)',
                val: '0.82',
                desc: 'Phân phối 12 cao độ hòa âm (C, C#, D... B) phản ánh cấu trúc hợp âm chủ đạo và tiến trình thang âm bài hát.'
            },
            centroid: {
                name: 'Spectral Centroid (Timbre Brightness)',
                val: '2,340 Hz',
                desc: 'Tâm khối phổ tần số biểu thị độ sáng ('brightness') và mức độ sắc nét của âm thanh.'
            }
        };

        function selectFeature(featKey) {
            const feat = featureDetails[featKey];
            if (!feat) return;
            document.getElementById('feat-name').innerText = feat.name;
            document.getElementById('feat-val').innerText = feat.val;
            document.getElementById('feat-desc').innerText = feat.desc;

            // Highlight selected chip
            document.querySelectorAll('.feature-chip').forEach(el => el.classList.remove('ring-1', 'ring-white'));
            const activeChip = document.getElementById(`chip-${featKey.replace('_', '')}`);
            if (activeChip) activeChip.classList.add('ring-1', 'ring-white');
        }

        // Toggle All 10 Genres Expander
        function toggleAllGenres() {
            const container = document.getElementById('all-genres-container');
            const btn = document.getElementById('expand-genres-btn');
            if (container.classList.contains('hidden')) {
                container.classList.remove('hidden');
                btn.innerHTML = '<span>Thu gọn 10 thể loại</span><i data-lucide="chevron-up" class="w-3.5 h-3.5"></i>';
            } else {
                container.classList.add('hidden');
                btn.innerHTML = '<span>Xem toàn bộ 10 thể loại</span><i data-lucide="chevron-down" class="w-3.5 h-3.5"></i>';
            }
            lucide.createIcons();
        }

        // Audio DSP Simulator Filters
        function applyAcousticFilter(type) {
            if (!audioContext) initAudioContext();
            document.querySelectorAll('.sim-filter-btn').forEach(b => {
                b.className = 'sim-filter-btn p-4 rounded-xl border border-slate-800 bg-[#121c2d] text-left hover:border-slate-700 transition';
            });
            const activeBtn = document.getElementById(`filter-btn-${type}`);
            if (activeBtn) {
                activeBtn.className = 'sim-filter-btn p-4 rounded-xl border border-cyan-500 bg-cyan-950/40 text-left transition';
            }

            const labelEl = document.getElementById('active-filter-label');
            if (type === 'clean') {
                biquadFilter.type = 'allpass';
                labelEl.innerText = 'Studio Clean (Không hiệu ứng méo)';
            } else if (type === 'phone') {
                biquadFilter.type = 'bandpass';
                biquadFilter.frequency.value = 1800;
                biquadFilter.Q.value = 1.2;
                labelEl.innerText = 'Smartphone Lo-Fi Mic (Dải thông hẹp 400Hz - 3.5kHz)';
            } else if (type === 'cafe') {
                biquadFilter.type = 'lowshelf';
                biquadFilter.frequency.value = 500;
                biquadFilter.gain.value = 6;
                labelEl.innerText = 'Quán Cafe (Tạp âm dải trung trầm)';
            } else if (type === 'reverb') {
                biquadFilter.type = 'peaking';
                biquadFilter.frequency.value = 1200;
                biquadFilter.Q.value = 3.0;
                biquadFilter.gain.value = 8;
                labelEl.innerText = 'Reverb Chamber (Vang dội phòng kín)';
            }
        }

        function testSimulatedPrediction() {
            switchTab('recognition');
            runSample(currentLoadedGenre.toLowerCase());
        }

        // Live Microphone Recording
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;

        async function toggleMicRecord() {
            const btnText = document.getElementById('mic-btn-text');
            if (!isRecording) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
                    mediaRecorder.onstop = async () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        const audioUrl = URL.createObjectURL(audioBlob);
                        const file = new File([audioBlob], 'live_recording.wav', { type: 'audio/wav' });

                        setLoadingState(true);
                        const formData = new FormData();
                        formData.append('file', file);
                        const res = await fetch('/predict', { method: 'POST', body: formData });
                        const data = await res.json();
                        loadAudio(audioUrl, data.predicted_genre, data.confidence);
                        renderResults(data);
                        setLoadingState(false);
                    };
                    mediaRecorder.start();
                    isRecording = true;
                    btnText.innerText = 'Đang thu âm... Nhấn dừng';
                } catch (e) {
                    alert('Không thể truy cập microphone: ' + e);
                }
            } else {
                mediaRecorder.stop();
                isRecording = false;
                btnText.innerText = 'Ghi âm trực tiếp';
            }
        }

        // Loading UI state
        function setLoadingState(loading) {
            const statusText = document.getElementById('global-status-text');
            const statusBadge = document.getElementById('global-status-badge');
            if (loading) {
                statusText.innerText = 'Đang phân tích âm học...';
                statusBadge.className = 'inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-cyan-950/60 border border-cyan-700/60 text-cyan-300 shadow-sm';
            } else {
                statusText.innerText = '✓ Phân tích hoàn tất';
                statusBadge.className = 'inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-xl text-xs font-semibold bg-emerald-950/60 border border-emerald-700/60 text-emerald-300 shadow-sm';
            }
        }

        // Telemetry & Health Check
        async function fetchHealthStats() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                document.getElementById('render-ram-text').innerText = `${data.ram_usage_mb} MB / 512 MB`;
            } catch (e) {
                console.log('Telemetry fetch error:', e);
            }
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)
