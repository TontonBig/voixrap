import streamlit as st
import numpy as np
import io, wave
from scipy import signal

EPS = 1e-12

st.set_page_config(
    page_title="VoixRap — Ta voix qui claque",
    layout="centered",
    page_icon="🎤"
)

# ═══════════════════════════════════════════════════
#  CSS — UI SOMBRE STYLE RAP
# ═══════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Mono', monospace !important;
    background-color: #0a0a0a !important;
    color: #f0f0f0 !important;
}

.stApp { background-color: #0a0a0a; }

h1 { font-family: 'Bebas Neue', sans-serif !important; font-size: 72px !important;
     letter-spacing: 6px !important; color: #ff3c3c !important; line-height: 1 !important; }

h3 { font-family: 'Space Mono', monospace !important; font-size: 11px !important;
     letter-spacing: 3px !important; color: #ff3c3c !important; text-transform: uppercase; }

.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #ff3c3c, #cc1a1a) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 26px !important;
    letter-spacing: 4px !important;
    padding: 18px !important;
    cursor: pointer !important;
    box-shadow: 0 8px 32px rgba(255,60,60,0.3) !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    box-shadow: 0 12px 40px rgba(255,60,60,0.5) !important;
    transform: translateY(-2px) !important;
}

.stDownloadButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #00ff88, #00cc66) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 26px !important;
    letter-spacing: 4px !important;
    padding: 18px !important;
    box-shadow: 0 8px 32px rgba(0,255,136,0.25) !important;
}

.stRadio > div { gap: 8px !important; }
.stRadio label { 
    background: #161616 !important;
    border: 1px solid #222 !important;
    border-radius: 10px !important;
    padding: 12px 16px !important;
    font-size: 12px !important;
}

.stFileUploader {
    background: #161616 !important;
    border: 2px dashed #333 !important;
    border-radius: 16px !important;
    padding: 20px !important;
}

.stSuccess {
    background: rgba(0,255,136,0.08) !important;
    border: 1px solid #00ff88 !important;
    border-radius: 12px !important;
}

.stInfo {
    background: rgba(255,140,0,0.08) !important;
    border: 1px solid #ff8c00 !important;
    border-radius: 12px !important;
}

.stMetric {
    background: #161616 !important;
    border: 1px solid #222 !important;
    border-radius: 10px !important;
    padding: 12px !important;
}

footer { display: none !important; }
#MainMenu { display: none !important; }
header { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════

def lin(db):  return 10 ** (db / 20.0)
def dbv(v):   return 20.0 * np.log10(max(abs(float(v)), EPS))

def to_mono(s, ch):
    return s.astype(np.float32) if ch == 1 else s.reshape(-1, ch).mean(axis=1).astype(np.float32)

def pcm_to_float(s, sw):
    x = s.astype(np.float32)
    return x if np.max(np.abs(x)) <= 1.0 else x / float(2 ** (8 * sw - 1))

def frames(x, sr, ms=20, hop=10):
    f = max(1, int(sr * ms / 1000))
    h = max(1, int(sr * hop / 1000))
    n = 1 + max(0, (len(x) - f) // h)
    idx = np.clip(np.arange(f)[None, :] + np.arange(n)[:, None] * h, 0, len(x) - 1)
    return x[idx]

def rms_db_frames(x, sr, ms=20, hop=10):
    return 20 * np.log10(np.sqrt(np.mean(frames(x, sr, ms, hop) ** 2, axis=1) + EPS))

def rms_act(x, sr):
    d = rms_db_frames(x, sr)
    a = d[d > -42]
    return float(np.percentile(a, 50)) if a.size > 5 else float(np.mean(d))

def peak_normalize(x, target_db=-0.3):
    pk = float(np.max(np.abs(x)) + EPS)
    tg = lin(target_db)
    if pk > tg:
        return (x * tg / pk).astype(np.float32)
    return x

def smooth_gain_db(gf_db, sr_frames, atk_ms, rel_ms):
    """Lisse un tableau de gains EN DB."""
    atk = 1 - np.exp(-1 / max(1, sr_frames * atk_ms / 1000))
    rel = 1 - np.exp(-1 / max(1, sr_frames * rel_ms / 1000))
    gs = np.zeros_like(gf_db)
    g = gf_db[0] if len(gf_db) else 0.0
    for i, v in enumerate(gf_db):
        g += (atk if v < g else rel) * (v - g)
        gs[i] = g
    return gs

def interpolate_gain(gain_frames, x_len, hop):
    frame_idx = np.arange(len(gain_frames)) * hop
    sample_idx = np.arange(x_len)
    return np.interp(sample_idx, frame_idx, gain_frames).astype(np.float32)

# ═══════════════════════════════════════════════════
#  BIQUAD EQ
# ═══════════════════════════════════════════════════

def biquad_peak(sr, freq, Q, gain_db):
    freq = min(freq, sr / 2 * 0.95)
    A  = 10 ** (gain_db / 40.)
    w0 = 2 * np.pi * freq / sr
    al = np.sin(w0) / (2 * Q)
    b0 = 1 + al * A;  b1 = -2 * np.cos(w0);  b2 = 1 - al * A
    a0 = 1 + al / A;  a1 = -2 * np.cos(w0);  a2 = 1 - al / A
    return np.array([b0/a0, b1/a0, b2/a0]), np.array([1., a1/a0, a2/a0])

def apply_eq(x, sr, freq, Q, gain_db):
    b, a = biquad_peak(sr, freq, Q, gain_db)
    return signal.lfilter(b, a, x).astype(np.float32)

def apply_hpf(x, sr, fc):
    fc = min(fc, sr / 2 * 0.95)
    b, a = signal.butter(4, fc / (sr / 2), btype='highpass')
    return signal.lfilter(b, a, x).astype(np.float32)

# ═══════════════════════════════════════════════════
#  GATE — CORRIGÉ
# ═══════════════════════════════════════════════════

def apply_gate(x, sr):
    hop = max(1, int(sr * 10 / 1000))
    db_f = rms_db_frames(x, sr, 20, 10)
    noise_floor = float(np.percentile(db_f, 5))
    threshold_db = noise_floor + 18.0  # ouvre 18 dB au-dessus du bruit

    # OUVERT (0 dB) quand signal > seuil, FERMÉ (-80 dB) quand < seuil
    gf_db = np.where(db_f > threshold_db, 0.0, -80.0)

    # Lisse : attaque rapide (3ms), release longue (300ms)
    frames_per_sec = 1000 / 10  # hop = 10ms
    gs_db = smooth_gain_db(gf_db, frames_per_sec, atk_ms=3, rel_ms=300)

    gi = interpolate_gain(gs_db, len(x), hop)
    return (x * lin(gi)).astype(np.float32)

# ═══════════════════════════════════════════════════
#  DE-ESSER
# ═══════════════════════════════════════════════════

def apply_deesser(x, sr):
    nyq = sr / 2
    if 6000 >= nyq:
        return x
    b, a = signal.butter(2, [6000/nyq, min(10000/nyq, 0.99)], btype='bandpass')
    sib = signal.lfilter(b, a, x)
    hop = max(1, int(sr * 2 / 1000))
    rms_s = np.sqrt(np.mean(frames(sib, sr, 5, 2) ** 2, axis=1) + EPS)
    thr = float(np.percentile(rms_s, 78))
    gf = np.where(rms_s > thr, lin(-5.0), 1.0)
    # Lisse en linéaire
    gs = np.zeros_like(gf)
    g = float(gf[0])
    atk = 1 - np.exp(-1 / max(1, sr * 0.001 / hop))
    rel = 1 - np.exp(-1 / max(1, sr * 0.004 / hop))
    for i, v in enumerate(gf):
        g += (atk if v < g else rel) * (v - g)
        gs[i] = g
    gi = interpolate_gain(gs, len(x), hop)
    return (x * gi).astype(np.float32)

# ═══════════════════════════════════════════════════
#  COMPRESSEUR — CORRIGÉ
# ═══════════════════════════════════════════════════

def apply_compressor(x, sr, thr_db, ratio, atk_ms, rel_ms, makeup_db):
    hop = max(1, int(sr * 2 / 1000))
    db_f = rms_db_frames(x, sr, 5, 2)
    over = db_f - thr_db
    act  = db_f > -42.0
    # Gain reduction en dB (négatif)
    gr_db = np.where(act & (over > 0), -over * (1 - 1 / ratio), 0.0)
    frames_per_sec = 1000 / 2
    gs_db = smooth_gain_db(gr_db, frames_per_sec, atk_ms, rel_ms)
    gi = interpolate_gain(gs_db + makeup_db, len(x), hop)
    out = (x * lin(gi)).astype(np.float32)
    return peak_normalize(out, -0.5)

# ═══════════════════════════════════════════════════
#  SATURATION
# ═══════════════════════════════════════════════════

def apply_saturation(x, style):
    if style == "Propre ❄️":
        return x
    drive_db = 5.0 if style == "Saturé ⚡" else 3.0
    mix      = 0.18 if style == "Saturé ⚡" else 0.10
    ra = rms_act(x, 44100)
    if ra > -12: drive_db = max(1.0, drive_db - 2.0)
    hot = x * lin(drive_db)
    sat = (np.tanh(hot * 0.9) + 0.08 * np.tanh(hot * 0.4)).astype(np.float32)
    ri = np.sqrt(np.mean(x ** 2) + EPS)
    rs = np.sqrt(np.mean(sat ** 2) + EPS)
    if rs > EPS: sat *= ri / rs
    out = ((1 - mix) * x + mix * sat).astype(np.float32)
    return peak_normalize(out, -0.5)

# ═══════════════════════════════════════════════════
#  LIMITER STÉRÉO
# ═══════════════════════════════════════════════════

def apply_limiter_stereo(left, right, sr, ceil_db=-0.1):
    ceil = lin(ceil_db)
    lh = max(1, int(sr * 0.008))
    ll = np.concatenate([left[lh:],  np.zeros(lh, dtype=np.float32)])
    lr = np.concatenate([right[lh:], np.zeros(lh, dtype=np.float32)])
    rel = 1 - np.exp(-1 / (sr * 0.060))
    g = 1.0
    outl = np.zeros_like(left)
    outr = np.zeros_like(right)
    for i in range(len(left)):
        pk  = max(abs(float(ll[i])), abs(float(lr[i])))
        ned = ceil / (pk + EPS) if pk > ceil else 1.0
        if ned < g: g = ned
        else: g += rel * (1 - g)
        outl[i] = left[i] * g
        outr[i] = right[i] * g
    # Safety hard clip
    pk_max = max(float(np.max(np.abs(outl))), float(np.max(np.abs(outr)))) + EPS
    if pk_max > ceil * 1.001:
        outl = (outl * ceil / pk_max).astype(np.float32)
        outr = (outr * ceil / pk_max).astype(np.float32)
    return outl.astype(np.float32), outr.astype(np.float32)

# ═══════════════════════════════════════════════════
#  LUFS NORMALIZE
# ═══════════════════════════════════════════════════

def lufs_simple(x, sr):
    hop = max(1, int(sr * 0.05))
    d = rms_db_frames(x, sr, 100, 50)
    mask = np.zeros(len(x), dtype=bool)
    for i, v in enumerate(d):
        s = i * hop; e = min(s + hop, len(x))
        if v > -42: mask[s:e] = True
    x_act = x[mask] if mask.sum() > sr else x
    b1, a1 = signal.bilinear([1., -2., 1.], [1., -1.99004745, 0.99007225], fs=sr)
    s1 = signal.lfilter(b1, a1, x_act)
    b2, a2 = signal.butter(2, 100. / (sr / 2), btype='highpass')
    s2 = signal.lfilter(b2, a2, s1)
    return -0.691 + 10 * np.log10(float(np.mean(s2 ** 2)) + EPS)

def normalize_lufs(left, right, sr, target):
    cur = -999.0
    for _ in range(6):
        mono = (left + right) * 0.5
        cur = lufs_simple(mono, sr)
        diff = target - cur
        if abs(diff) < 0.3: break
        g = lin(min(diff, 6.0))
        left  = (left  * g).astype(np.float32)
        right = (right * g).astype(np.float32)
        pk = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))))
        if pk > 0.98:
            left, right = apply_limiter_stereo(left, right, sr, -0.1)
    return left, right

# ═══════════════════════════════════════════════════
#  WIDENER
# ═══════════════════════════════════════════════════

def apply_widener(x, sr, style):
    width = {"Propre ❄️": 0.12, "Chaud 🔥": 0.20, "Saturé ⚡": 0.28}[style]
    delay = int(sr * 0.012)
    side  = np.zeros(len(x), dtype=np.float32)
    if delay < len(x):
        side[delay:] = x[:-delay] * width
        b, a = signal.butter(2, 180. / (sr / 2), btype='highpass')
        side = signal.lfilter(b, a, side).astype(np.float32)
    left  = peak_normalize((x + side).astype(np.float32), -0.3)
    right = peak_normalize((x - side).astype(np.float32), -0.3)
    return left, right

# ═══════════════════════════════════════════════════
#  BUILD WAV 24bit
# ═══════════════════════════════════════════════════

def build_wav(left, right, sr):
    def pcm24(x):
        x = np.clip(x, -1, 1)
        iv = (x * 8388607).astype(np.int32)
        out = np.zeros((len(iv), 3), dtype=np.uint8)
        out[:, 0] = (iv & 0xFF).astype(np.uint8)
        out[:, 1] = ((iv >> 8) & 0xFF).astype(np.uint8)
        out[:, 2] = ((iv >> 16) & 0xFF).astype(np.uint8)
        return out.tobytes()
    il = np.empty(len(left) + len(right), dtype=np.float32)
    il[0::2] = left; il[1::2] = right
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(3)
        wf.setframerate(sr)
        wf.writeframes(pcm24(il))
    return buf.getvalue()

# ═══════════════════════════════════════════════════
#  PROFILS MICRO
# ═══════════════════════════════════════════════════

PROFILS = {
    "📱 Téléphone / BandLab / AirPods": {
        "hpf": 150,
        "cuts":   [(200, 1.5, -5.0), (380, 1.8, -4.0), (2000, 1.2, -3.0)],
        "boosts": [(3500, 1.0, 3.5), (10000, 0.8, 3.0)],
        "comp":   {"thr": -14, "ratio": 5.0, "atk": 8, "rel": 120, "mg": 8},
    },
    "🔵 Micro USB (Blue Yeti, HyperX...)": {
        "hpf": 100,
        "cuts":   [(150, 1.3, -4.0), (380, 1.8, -3.5), (3000, 1.2, -2.5)],
        "boosts": [(3500, 1.0, 3.0), (12000, 0.7, 2.5)],
        "comp":   {"thr": -14, "ratio": 4.5, "atk": 7, "rel": 100, "mg": 7},
    },
    "🎤 Micro condensateur (Rode, AKG, AT...)": {
        "hpf": 80,
        "cuts":   [(100, 1.5, -3.0), (380, 1.8, -3.0), (6000, 1.2, -2.5)],
        "boosts": [(3500, 1.0, 3.0), (15000, 0.7, 2.0)],
        "comp":   {"thr": -16, "ratio": 5.0, "atk": 5, "rel": 80, "mg": 6},
    },
    "🔴 Micro dynamique (SM7B, SM58...)": {
        "hpf": 60,
        "cuts":   [(100, 1.5, -3.0), (380, 1.8, -2.5)],
        "boosts": [(3500, 1.0, 3.5), (5000, 0.8, 2.0), (12000, 0.7, 3.0)],
        "comp":   {"thr": -12, "ratio": 4.0, "atk": 7, "rel": 100, "mg": 8},
    },
}

STYLES = ["Propre ❄️", "Chaud 🔥", "Saturé ⚡"]

# ═══════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════

st.markdown("<h1>🎤 VOIXRAP</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#555;font-size:11px;letter-spacing:3px;margin-top:-10px'>UPLOAD · TRAITE · TÉLÉCHARGE — GRATUIT</p>", unsafe_allow_html=True)
st.markdown("<hr style='border-color:#222;margin:24px 0'>", unsafe_allow_html=True)

# Micro
st.markdown("<h3>🎙 Avec quoi t'as enregistré ?</h3>", unsafe_allow_html=True)
micro = st.radio("", list(PROFILS.keys()), label_visibility="collapsed")

st.markdown("<hr style='border-color:#1a1a1a;margin:20px 0'>", unsafe_allow_html=True)

# Style
st.markdown("<h3>🔥 Quel style tu veux ?</h3>", unsafe_allow_html=True)
style = st.radio("", STYLES, index=1, horizontal=True, label_visibility="collapsed")

st.markdown("<hr style='border-color:#1a1a1a;margin:20px 0'>", unsafe_allow_html=True)

# Upload
st.markdown("<h3>📂 Ta prise voix</h3>", unsafe_allow_html=True)
uploaded = st.file_uploader(
    "WAV, MP3, M4A, FLAC, OGG",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    label_visibility="visible"
)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

if uploaded is None:
    st.info("👆 Charge ta prise pour commencer.")
    st.stop()

# TRAITEMENT
if st.button("⚡ TRAITER MA VOIX"):

    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(uploaded)
        sr    = audio.frame_rate
        sw    = audio.sample_width
        raw   = to_mono(np.array(audio.get_array_of_samples()), audio.channels)
        x     = pcm_to_float(raw, sw)
        x     = (x - float(np.mean(x))).astype(np.float32)
    except Exception as e:
        st.error(f"❌ Impossible de lire le fichier : {e}")
        st.stop()

    p = PROFILS[micro]
    c = p["comp"]

    progress = st.progress(0, text="Analyse...")

    # 1. Gate
    progress.progress(10, text="🚪 Nettoyage du silence...")
    x = apply_gate(x, sr)

    # 2. HPF
    progress.progress(20, text="🎚️ Filtre basses fréquences...")
    x = apply_hpf(x, sr, p["hpf"])

    # 3. EQ coupures
    progress.progress(30, text="🎚️ Égalisation...")
    for freq, Q, gain in p["cuts"]:
        x = apply_eq(x, sr, freq, Q, gain)

    # 4. EQ boosts
    for freq, Q, gain in p["boosts"]:
        x = apply_eq(x, sr, freq, Q, gain)
    x = peak_normalize(x, -0.5)

    # 5. De-esser
    progress.progress(45, text="👂 Suppression sibilances...")
    x = apply_deesser(x, sr)

    # 6. Transient — accentue l'attaque
    progress.progress(55, text="⚡ Punch & attaque...")
    # Simple transient boost via short-term vs long-term envelope
    fast = 1 - np.exp(-1 / (sr * 0.002)); slow = 1 - np.exp(-1 / (sr * 0.025))
    ef = es = 0.0
    envf = np.zeros(len(x)); envs = np.zeros(len(x))
    for i in range(len(x)):
        s2 = float(x[i]) ** 2
        ef += fast * (s2 - ef); es += slow * (s2 - es)
        envf[i] = ef; envs[i] = es
    boost = 4.5
    ratio_env = (envf + EPS) / (envs + EPS)
    gain_t = np.where(ratio_env > 2.5, lin(boost), lin(-1.5))
    sm = 1 - np.exp(-1 / (sr * 0.001))
    gs2 = np.zeros_like(gain_t); g2 = 1.0
    for i, v in enumerate(gain_t): g2 += sm * (v - g2); gs2[i] = g2
    x = peak_normalize((x * gs2).astype(np.float32), -0.5)

    # 7. Compression
    progress.progress(65, text="🗜️ Compression...")
    ra = rms_act(x, sr)
    thr = max(-32., min(-6., ra - 1.5))
    x = apply_compressor(x, sr, thr, c["ratio"], c["atk"], c["rel"], c["mg"])

    # 8. Saturation
    progress.progress(75, text="🔥 Caractère & chaleur...")
    x = apply_saturation(x, style)

    # 9. Safety peak
    x = peak_normalize(x, -0.5)

    # 10. Widener
    progress.progress(82, text="🎧 Espace stéréo...")
    left, right = apply_widener(x, sr, style)

    # 11. Limiter
    progress.progress(88, text="🛑 Limiteur final...")
    left, right = apply_limiter_stereo(left, right, sr, -0.1)

    # 12. LUFS
    progress.progress(93, text="📢 Normalisation volume...")
    target_lufs = -10.0
    left, right = normalize_lufs(left, right, sr, target_lufs)

    # 13. Export
    progress.progress(98, text="💾 Export WAV 24bit...")
    wav_bytes = build_wav(left, right, sr)
    progress.progress(100, text="✅ Terminé !")

    # Stats
    mono_out = (left + right) * 0.5
    rms_out  = rms_act(mono_out, sr)
    pk_out   = dbv(np.max(np.abs(mono_out)))
    clips    = int(np.sum(np.abs(mono_out) >= 0.998))
    lufs_out = lufs_simple(mono_out, sr)

    st.markdown("<hr style='border-color:#222;margin:24px 0'>", unsafe_allow_html=True)
    st.markdown("<h3>✅ Voix traitée — prête pour le beat</h3>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Niveau RMS", f"{rms_out:.0f} dB")
    c2.metric("Peak",       f"{pk_out:.1f} dB")
    c3.metric("LUFS",       f"{lufs_out:.0f}")
    c4.metric("Clips",      f"{clips} {'✅' if clips == 0 else '⚠️'}")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    nom_sortie = uploaded.name.rsplit(".", 1)[0] + "_voixrap_v5.wav"
    st.download_button(
        label="⬇ TÉLÉCHARGER MA VOIX",
        data=wav_bytes,
        file_name=nom_sortie,
        mime="audio/wav",
        use_container_width=True,
    )

    st.markdown("<hr style='border-color:#1a1a1a;margin:24px 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#111;border:1px solid #1a1a1a;border-radius:12px;padding:20px;font-size:11px;color:#555;line-height:2'>
    💡 <b style='color:#888'>Pour une meilleure prise la prochaine fois :</b><br>
    📍 15 à 20 cm du micro — ni trop près ni trop loin<br>
    🔇 Ferme les fenêtres, coupe la clim<br>
    👕 Enregistre dans un placard plein de vêtements<br>
    📱 Sur BandLab : enregistre en WAV brut sans effets
    </div>
    """, unsafe_allow_html=True)
