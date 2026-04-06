import streamlit as st
import numpy as np
import io, wave, tempfile, os
from scipy import signal
from scipy.fft import rfft, rfftfreq

EPS = 1e-12

st.set_page_config(
    page_title="VoixRap — Ta voix qui claque",
    layout="centered",
    page_icon="🎤"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Mono:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Mono', monospace !important; background-color: #0a0a0a !important; color: #f0f0f0 !important; }
.stApp { background-color: #0a0a0a; }
h1 { font-family: 'Bebas Neue', sans-serif !important; font-size: 72px !important; letter-spacing: 6px !important; color: #ff3c3c !important; line-height: 1 !important; }
h3 { font-family: 'Space Mono', monospace !important; font-size: 11px !important; letter-spacing: 3px !important; color: #ff3c3c !important; text-transform: uppercase; }
.stButton > button { width: 100% !important; background: linear-gradient(135deg, #ff3c3c, #cc1a1a) !important; color: white !important; border: none !important; border-radius: 12px !important; font-family: 'Bebas Neue', sans-serif !important; font-size: 26px !important; letter-spacing: 4px !important; padding: 18px !important; box-shadow: 0 8px 32px rgba(255,60,60,0.3) !important; }
.stDownloadButton > button { width: 100% !important; background: linear-gradient(135deg, #00ff88, #00cc66) !important; color: #000 !important; border: none !important; border-radius: 12px !important; font-family: 'Bebas Neue', sans-serif !important; font-size: 26px !important; letter-spacing: 4px !important; padding: 18px !important; box-shadow: 0 8px 32px rgba(0,255,136,0.25) !important; }
.stSlider { padding: 10px 0 !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }
header { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════
#  HELPERS DSP
# ═══════════════════════════════════════════════════

def lin(db):   return 10 ** (db / 20.0)
def dbv(v):    return 20.0 * np.log10(max(abs(float(v)), EPS))

def frames_fn(x, sr, ms=20, hop_ms=10):
    f = max(1, int(sr * ms / 1000))
    h = max(1, int(sr * hop_ms / 1000))
    n = 1 + max(0, (len(x) - f) // h)
    idx = np.clip(np.arange(f)[None,:] + np.arange(n)[:,None] * h, 0, len(x)-1)
    return x[idx]

def rms_db_frames(x, sr, ms=20, hop_ms=10):
    return 20 * np.log10(np.sqrt(np.mean(frames_fn(x, sr, ms, hop_ms)**2, axis=1) + EPS))

def rms_act(x, sr):
    d = rms_db_frames(x, sr)
    a = d[d > -42]
    return float(np.percentile(a, 50)) if a.size > 5 else float(np.mean(d))

def peak_normalize(x, target_db=-0.3):
    pk = float(np.max(np.abs(x)) + EPS)
    tg = lin(target_db)
    if pk > tg: return (x * tg/pk).astype(np.float32)
    return x

def smooth_gain(gf, atk, rel):
    gs = np.zeros_like(gf); g = gf[0]
    for i, v in enumerate(gf):
        g += (atk if v < g else rel) * (v - g); gs[i] = g
    return gs

# ── Filtres ──────────────────────────────────────────

def biquad_peak(sr, freq, Q, gain_db):
    freq = min(freq, sr/2*0.95)
    A  = 10**(gain_db/40.); w0 = 2*np.pi*freq/sr
    al = np.sin(w0)/(2*Q)
    b0=1+al*A; b1=-2*np.cos(w0); b2=1-al*A
    a0=1+al/A; a1=-2*np.cos(w0); a2=1-al/A
    return np.array([b0/a0,b1/a0,b2/a0]), np.array([1.,a1/a0,a2/a0])

def apply_eq(x, sr, freq, Q, gain_db):
    b, a = biquad_peak(sr, freq, Q, gain_db)
    return signal.lfilter(b, a, x).astype(np.float32)

def apply_hpf(x, sr, fc, order=4):
    fc = min(fc, sr/2*0.95)
    b, a = signal.butter(order, fc/(sr/2), btype='highpass')
    return signal.lfilter(b, a, x).astype(np.float32)

def apply_lpf(x, sr, fc, order=4):
    fc = min(fc, sr/2*0.95)
    b, a = signal.butter(order, fc/(sr/2), btype='lowpass')
    return signal.lfilter(b, a, x).astype(np.float32)

def apply_low_shelf(x, sr, fc, gain_db):
    fc = min(fc, sr/2*0.95)
    A = 10**(gain_db/40.); w0 = 2*np.pi*fc/sr
    cosW = np.cos(w0); sinW = np.sin(w0)
    beta = np.sqrt(A)/1.0
    b0=A*((A+1)-(A-1)*cosW+beta*sinW); b1=2*A*((A-1)-(A+1)*cosW)
    b2=A*((A+1)-(A-1)*cosW-beta*sinW)
    a0=(A+1)+(A-1)*cosW+beta*sinW; a1=-2*((A-1)+(A+1)*cosW)
    a2=(A+1)+(A-1)*cosW-beta*sinW
    b=np.array([b0/a0,b1/a0,b2/a0]); a=np.array([1.,a1/a0,a2/a0])
    return signal.lfilter(b, a, x).astype(np.float32)

def apply_high_shelf(x, sr, fc, gain_db):
    fc = min(fc, sr/2*0.95)
    A = 10**(gain_db/40.); w0 = 2*np.pi*fc/sr
    cosW = np.cos(w0); sinW = np.sin(w0)
    beta = np.sqrt(A)/1.0
    b0=A*((A+1)+(A-1)*cosW+beta*sinW); b1=-2*A*((A-1)+(A+1)*cosW)
    b2=A*((A+1)+(A-1)*cosW-beta*sinW)
    a0=(A+1)-(A-1)*cosW+beta*sinW; a1=2*((A-1)-(A+1)*cosW)
    a2=(A+1)-(A-1)*cosW-beta*sinW
    b=np.array([b0/a0,b1/a0,b2/a0]); a=np.array([1.,a1/a0,a2/a0])
    return signal.lfilter(b, a, x).astype(np.float32)

# ── Compresseur RMS ───────────────────────────────────

def apply_comp(x, sr, thr_db, ratio, atk_ms, rel_ms, makeup_db, knee_db=3.0):
    hop  = max(1, int(sr*2/1000))
    db_f = rms_db_frames(x, sr, 5, 2)
    fps  = 1000/2
    # Soft knee
    over = db_f - thr_db
    half_knee = knee_db / 2
    gr = np.where(
        over <= -half_knee, 0.0,
        np.where(over >= half_knee,
                 -over*(1-1/ratio),
                 -(over+half_knee)**2 / (2*knee_db*(1-1/ratio) + EPS)
        )
    )
    gr = np.where(db_f > -42, gr, 0.0)
    atk = 1-np.exp(-1/max(1, fps*atk_ms/1000))
    rel = 1-np.exp(-1/max(1, fps*rel_ms/1000))
    gs  = smooth_gain(gr, atk, rel)
    gi  = np.interp(np.arange(len(x)), np.arange(len(gs))*hop, gs+makeup_db)
    return (x * lin(gi)).astype(np.float32)

# ═══════════════════════════════════════════════════
#  MULTIBANDE — Style C6 Waves
#  Chaque bande = filtre bande passante + compresseur indépendant
# ═══════════════════════════════════════════════════

def apply_multiband(x, sr, a):
    """
    4 bandes adaptatives selon l analyse de la prise.
    Reproduit la logique du C6 Waves.
    """
    # Crossovers
    f1 = 120   # Sub / Graves
    f2 = 500   # Graves / Lo-mid
    f3 = 3000  # Lo-mid / Présence
    f4 = 8000  # Présence / Air

    # Séparation en 4 bandes
    b1 = apply_lpf(x, sr, f1, 4)                                    # Sub (< 120 Hz)
    b2 = apply_hpf(apply_lpf(x, sr, f2, 4), sr, f1, 4)             # Graves (120-500 Hz)
    b3 = apply_hpf(apply_lpf(x, sr, f3, 4), sr, f2, 4)             # Lo-mid (500-3k Hz)
    b4 = apply_hpf(apply_lpf(x, sr, f4, 4), sr, f3, 4)             # Présence (3k-8k Hz)
    b5 = apply_hpf(x, sr, f4, 4)                                    # Air (> 8k Hz)

    # ── Bande 1 — Sub (< 120 Hz) ─────────────────────
    # Coupe forte — le sub est rarement utile sur une voix rap
    thr1 = max(-50., rms_act(b1, sr) - 3)
    c1   = apply_comp(b1, sr, thr1, 8.0, 10, 40, 0, knee_db=2)
    # Atténuation supplémentaire si trop de sub
    sub_gain = -6.0 if a['sub'] > 15 else -3.0
    c1 = (c1 * lin(sub_gain)).astype(np.float32)

    # ── Bande 2 — Graves (120-500 Hz) ────────────────
    # Contrôlée par le slider graves_db
    # Compression douce pour uniformiser
    thr2 = max(-40., rms_act(b2, sr) - 4)
    c2   = apply_comp(b2, sr, thr2, 4.0, 15, 80, 0, knee_db=3)

    # ── Bande 3 — Lo-mid / Boue (500-3k Hz) ──────────
    # Zone la plus problématique — compression agressive si boue détectée
    mud_ratio = 6.0 if a['lo_mid'] > 35 else 4.0
    mud_thr   = max(-40., rms_act(b3, sr) - 2)
    c3 = apply_comp(b3, sr, mud_thr, mud_ratio, 8, 50, 0, knee_db=2)
    # Coupe supplémentaire si très boueux
    if a['lo_mid'] > 35:
        c3 = (c3 * lin(-3.0)).astype(np.float32)

    # ── Bande 4 — Présence (3k-8k Hz) ────────────────
    # DeBoxy + DeHarsher style — compresse les pics agressifs
    pres_ratio = 5.0 if a['presence'] > 30 else 3.0
    pres_thr   = max(-40., rms_act(b4, sr) - 3)
    c4 = apply_comp(b4, sr, pres_thr, pres_ratio, 5, 40, 0, knee_db=2)

    # ── Bande 5 — Air (> 8k Hz) ──────────────────────
    # Compression légère pour contrôler les sibilances hautes
    thr5 = max(-50., rms_act(b5, sr) - 5)
    c5   = apply_comp(b5, sr, thr5, 3.0, 3, 30, 0, knee_db=1)

    # Reconstruction
    out = (c1 + c2 + c3 + c4 + c5).astype(np.float32)
    return out

# ═══════════════════════════════════════════════════
#  DE-ESSER — Style Triple D (DeBoxy + DeHarsher + DeEsser)
# ═══════════════════════════════════════════════════

def apply_triple_deess(x, sr, a):
    """
    3 étages de de-essing ciblés — reproduit la logique du Triple D de Manny Marroquin.
    DeBoxy → DeHarsher → DeEsser
    """
    proc = x.copy()
    nyq  = sr / 2

    # ── DeBoxy — 200-600 Hz ───────────────────────────
    # Élimine le son en carton / boîte
    if 200 < nyq:
        fc1 = min(400/nyq, 0.95); fc2 = min(600/nyq, 0.99)
        b, a_f = signal.butter(2, [fc1, fc2], btype='bandpass')
        boxy  = signal.lfilter(b, a_f, proc)
        hop   = max(1, int(sr*2/1000))
        rms_b = np.sqrt(np.mean(frames_fn(boxy, sr, 5, 2)**2, axis=1) + EPS)
        thr_b = float(np.percentile(rms_b, 70))
        gf_b  = np.where(rms_b > thr_b, lin(-4.0), 1.0)
        atk_b = 1-np.exp(-1/max(1, sr*0.002/hop))
        rel_b = 1-np.exp(-1/max(1, sr*0.008/hop))
        gs_b  = smooth_gain(gf_b, atk_b, rel_b)
        gi_b  = np.interp(np.arange(len(proc)), np.arange(len(gs_b))*hop, gs_b)
        proc  = (proc * gi_b).astype(np.float32)

    # ── DeHarsher — 2k-5k Hz ─────────────────────────
    # Élimine l agressivité et la dureté
    if 2000 < nyq:
        fc3 = min(2000/nyq, 0.95); fc4 = min(5000/nyq, 0.99)
        b2, a_f2 = signal.butter(2, [fc3, fc4], btype='bandpass')
        harsh  = signal.lfilter(b2, a_f2, proc)
        rms_h  = np.sqrt(np.mean(frames_fn(harsh, sr, 5, 2)**2, axis=1) + EPS)
        thr_h  = float(np.percentile(rms_h, 72))
        amount_h = -5.0 if a['presence'] > 25 else -3.0
        gf_h   = np.where(rms_h > thr_h, lin(amount_h), 1.0)
        atk_h  = 1-np.exp(-1/max(1, sr*0.001/hop))
        rel_h  = 1-np.exp(-1/max(1, sr*0.006/hop))
        gs_h   = smooth_gain(gf_h, atk_h, rel_h)
        gi_h   = np.interp(np.arange(len(proc)), np.arange(len(gs_h))*hop, gs_h)
        proc   = (proc * gi_h).astype(np.float32)

    # ── DeEsser — fréquence exacte des sibilances ─────
    # Cible la fréquence détectée dans l analyse
    sib_f = a.get('sib_freq', 7000)
    if sib_f < nyq and a['sib_zone'] > 12:
        fc5 = min((sib_f-1500)/nyq, 0.95)
        fc6 = min((sib_f+1500)/nyq, 0.99)
        if fc5 > 0 and fc6 > fc5:
            b3, a_f3 = signal.butter(2, [fc5, fc6], btype='bandpass')
            sib    = signal.lfilter(b3, a_f3, proc)
            rms_s  = np.sqrt(np.mean(frames_fn(sib, sr, 5, 2)**2, axis=1) + EPS)
            thr_s  = float(np.percentile(rms_s, 75))
            amount_s = max(-8.0, -(a['sib_zone']-10)*0.35)
            gf_s   = np.where(rms_s > thr_s, lin(amount_s), 1.0)
            atk_s  = 1-np.exp(-1/max(1, sr*0.0008/hop))
            rel_s  = 1-np.exp(-1/max(1, sr*0.004/hop))
            gs_s   = smooth_gain(gf_s, atk_s, rel_s)
            gi_s   = np.interp(np.arange(len(proc)), np.arange(len(gs_s))*hop, gs_s)
            proc   = (proc * gi_s).astype(np.float32)

    return proc

# ═══════════════════════════════════════════════════
#  SATURATION HARMONIQUE
# ═══════════════════════════════════════════════════

def apply_saturation(x, drive_db=3.0, mix=0.12):
    hot = x * lin(drive_db)
    sat = (np.tanh(hot*0.9) + 0.08*np.tanh(hot*0.4)).astype(np.float32)
    ri  = np.sqrt(np.mean(x**2)+EPS); rs = np.sqrt(np.mean(sat**2)+EPS)
    if rs > EPS: sat *= ri/rs
    return peak_normalize(((1-mix)*x + mix*sat).astype(np.float32), -0.5)

# ═══════════════════════════════════════════════════
#  LIMITER STÉRÉO
# ═══════════════════════════════════════════════════

def apply_limiter_stereo(left, right, sr, ceil_db=-0.1):
    ceil = lin(ceil_db)
    lh   = max(1, int(sr*0.008))
    ll   = np.concatenate([left[lh:],  np.zeros(lh, dtype=np.float32)])
    lr   = np.concatenate([right[lh:], np.zeros(lh, dtype=np.float32)])
    rel  = 1-np.exp(-1/(sr*0.060))
    g    = 1.0; outl=np.zeros_like(left); outr=np.zeros_like(right)
    for i in range(len(left)):
        pk  = max(abs(float(ll[i])), abs(float(lr[i])))
        ned = ceil/(pk+EPS) if pk>ceil else 1.0
        if ned < g: g=ned
        else: g += rel*(1-g)
        outl[i]=left[i]*g; outr[i]=right[i]*g
    pk_max = max(float(np.max(np.abs(outl))), float(np.max(np.abs(outr))))+EPS
    if pk_max > ceil*1.001:
        outl=(outl*ceil/pk_max).astype(np.float32)
        outr=(outr*ceil/pk_max).astype(np.float32)
    return outl.astype(np.float32), outr.astype(np.float32)

# ═══════════════════════════════════════════════════
#  WIDENER — Side Widener style
# ═══════════════════════════════════════════════════

def apply_widener(mono, sr, width=0.20):
    delay = int(sr*0.012)
    side  = np.zeros(len(mono), dtype=np.float32)
    if delay < len(mono):
        side[delay:] = mono[:-delay]*width
        b, a = signal.butter(2, 200./(sr/2), btype='highpass')
        side = signal.lfilter(b, a, side).astype(np.float32)
    left  = peak_normalize((mono+side).astype(np.float32), -0.3)
    right = peak_normalize((mono-side).astype(np.float32), -0.3)
    return left, right

# ═══════════════════════════════════════════════════
#  BUILD WAV 24bit
# ═══════════════════════════════════════════════════

def build_wav(left, right, sr):
    def pcm24(x):
        x=np.clip(x,-1,1); iv=(x*8388607).astype(np.int32)
        out=np.zeros((len(iv),3),dtype=np.uint8)
        out[:,0]=(iv&0xFF).astype(np.uint8)
        out[:,1]=((iv>>8)&0xFF).astype(np.uint8)
        out[:,2]=((iv>>16)&0xFF).astype(np.uint8)
        return out.tobytes()
    il=np.empty(len(left)+len(right),dtype=np.float32)
    il[0::2]=left; il[1::2]=right
    buf=io.BytesIO()
    with wave.open(buf,'wb') as wf:
        wf.setnchannels(2); wf.setsampwidth(3)
        wf.setframerate(sr); wf.writeframes(pcm24(il))
    return buf.getvalue()

# ═══════════════════════════════════════════════════
#  ANALYSE INTELLIGENTE
# ═══════════════════════════════════════════════════

def analyse_prise(x, sr):
    resultats = {}
    seg = x[:min(len(x), sr*8)]
    win = signal.windows.hann(len(seg), sym=False)
    P   = (np.abs(rfft(seg*win))**2)+EPS
    F   = rfftfreq(len(seg), 1./sr)
    tot = P.sum()

    def energie(fa, fb):
        m=(F>=fa)&(F<fb)
        return float(P[m].sum()/tot*100) if np.any(m) else 0.

    resultats['sub']      = energie(20,100)
    resultats['graves']   = energie(100,200)
    resultats['lo_mid']   = energie(200,500)
    resultats['mid']      = energie(500,2500)
    resultats['presence'] = energie(2500,6000)
    resultats['sib_zone'] = energie(6000,12000)
    resultats['air']      = energie(12000,20000)

    # Fréquence exacte sibilances
    if sr > 12000:
        m_sib = (F>=5000)&(F<12000)
        resultats['sib_freq'] = float(F[m_sib][np.argmax(P[m_sib])]) if np.any(m_sib) else 7000.
    else:
        resultats['sib_freq'] = 7000.

    rms   = rms_act(x, sr)
    peak  = dbv(np.max(np.abs(x)))
    clips = int(np.sum(np.abs(x)>=0.998))
    resultats['rms']=rms; resultats['peak']=peak; resultats['clips']=clips

    db_f  = rms_db_frames(x, sr, 400, 100)
    a_    = db_f[db_f>-42]
    dr    = float(np.percentile(a_,95)-np.percentile(a_,5)) if a_.size>10 else 20.
    crest = peak-rms
    resultats['dr']=dr; resultats['crest']=crest
    resultats['deja_compresse'] = dr<12. and crest<10.

    noise_floor = float(np.percentile(db_f,5))
    resultats['noise_floor']=noise_floor
    resultats['snr']=rms-noise_floor

    plos = energie(20,120)/(energie(500,3000)+EPS)
    resultats['plos_risk']=plos>0.3

    env=np.abs(x); b_e,a_e=signal.butter(1,10/(sr/2),btype='lowpass')
    env_s=signal.lfilter(b_e,a_e,env)
    peaks_idx=signal.find_peaks(env_s,height=np.max(env_s)*0.3,distance=sr//4)[0]
    rev=False
    if len(peaks_idx)>1:
        decays=[]
        for pk in peaks_idx[:5]:
            end=min(pk+int(sr*0.3),len(env_s)-1)
            if end>pk: decays.append((env_s[pk]-env_s[end])/(env_s[pk]+EPS))
        if decays: rev=float(np.mean(decays))<0.5
    resultats['reverb']=rev
    return resultats

def generer_diagnostic(a):
    diag=[]
    if a['clips']>100:
        diag.append(("🔴","Ta voix sature fort",f"Elle dépasse le maximum sur {a['clips']} points — réenregistre si possible"))
    elif a['clips']>0:
        diag.append(("🟡","Légère saturation détectée","Quelques points saturés — on peut tenter"))
    if a['rms']<-32:
        diag.append(("🔴","Ta voix est trop silencieuse","Monte le gain — le bruit va s'amplifier"))
    elif a['rms']<-24:
        diag.append(("🟡","Niveau un peu faible","Traitable mais monte ton gain la prochaine fois"))
    else:
        diag.append(("🟢","Niveau correct",f"Ta voix est bien enregistrée ({a['rms']:.0f} dBFS)"))
    if a['sub']>15:
        diag.append(("🔴","Trop de basses inutiles","Beaucoup d'énergie sous 100 Hz — ça pompe dans la compression"))
    if a['graves']>30:
        diag.append(("🔴","Ta voix est trop grave et boueuse","Trop d'énergie dans les graves — elle va se perdre dans le beat"))
    elif a['lo_mid']>35:
        diag.append(("🟡","Beaucoup de boue dans les mids","Zone 200-500 Hz très chargée — on va nettoyer ça"))
    elif a['lo_mid']>25:
        diag.append(("🟡","Un peu de boue dans les mids","Zone 200-500 Hz chargée — on va nettoyer ça"))
    if a['presence']<8:
        diag.append(("🟡","Ta voix manque de présence","Pas assez d'énergie entre 2.5k et 6k — elle va se noyer dans le beat"))
    elif a['presence']>35:
        diag.append(("🟡","Ta voix est un peu agressive","Beaucoup d'énergie entre 2.5k et 6k — on va adoucir ça"))
    if a['sib_zone']>20:
        diag.append(("🟡",f"Tes 'S' et 'CH' font mal aux oreilles",f"Sibilances détectées à {a['sib_freq']:.0f} Hz"))
    if a['air']<2:
        diag.append(("🟡","Ta voix manque d'air et de brillance","Pas assez d'énergie dans les hautes fréquences"))
    if a['snr']<15:
        diag.append(("🔴","Beaucoup de bruit de fond","Ferme les fenêtres, coupe la clim"))
    elif a['snr']<25:
        diag.append(("🟡","Un peu de bruit de fond","C'est gérable"))
    else:
        diag.append(("🟢","Prise silencieuse","Peu de bruit de fond — bien joué"))
    if a['deja_compresse']:
        diag.append(("🟡","Ta voix semble déjà compressée","BandLab ou app mobile ? On adapte la compression"))
    if a['plos_risk']:
        diag.append(("🟡","Des 'P' et 'B' qui claquent","Utilise un filtre anti-pop"))
    if a['reverb']:
        diag.append(("🟡","Un peu de reverb de pièce","Ta pièce résonne"))
    return diag

# ═══════════════════════════════════════════════════
#  CHAÎNE DE TRAITEMENT PRINCIPALE
# ═══════════════════════════════════════════════════

def traiter_prise(x, sr, a, wet=1.0, graves_db=-6, comp_pct=0.5, presence_db=2):
    if wet == 0.0:
        left=peak_normalize(np.array(x,dtype=np.float32),-0.3); return left,left.copy()

    proc = x.copy()

    # ── 1. HPF — coupe tout sous 80 Hz ───────────────
    proc = apply_hpf(proc, sr, 80, order=4)

    # ── 2. EQ correctif — style Pro-Q4 ───────────────
    # Low shelf graves — slider contrôle l intensité
    proc = apply_low_shelf(proc, sr, 180, float(graves_db) * wet)
    # Coupe lo-mid 380 Hz — boue principale
    mud = -5.0 if a['lo_mid']>35 else -3.0
    proc = apply_eq(proc, sr, 380, 1.8, mud * wet)
    # Coupe nasale 1kHz si besoin
    if a['lo_mid']>30 or a['mid']>30:
        proc = apply_eq(proc, sr, 1000, 1.2, -2.5 * wet)

    # ── 3. Compresseur 1 — Maintien (C1 style) ───────
    ratio1 = 2.0 + comp_pct*2.0
    ra1    = rms_act(proc, sr)
    thr1   = max(-32., min(-10., ra1-3.0))
    mg1    = max(2., min(8., abs(thr1-ra1)*(1-1/ratio1)))
    proc   = apply_comp(proc, sr, thr1, ratio1, 20, 200, mg1*wet, knee_db=4)

    # ── 4. Multibande — C6 style ─────────────────────
    proc_mb = apply_multiband(proc, sr, a)
    proc = (wet*proc_mb + (1-wet)*proc).astype(np.float32)

    # ── 5. Triple De-ess — Triple D style ────────────
    proc = apply_triple_deess(proc, sr, a)

    # ── 6. EQ boosts — API 550B style ────────────────
    # Présence — slider contrôle
    proc = apply_eq(proc, sr, 3500, 1.0, float(presence_db)*wet)
    # Air — high shelf
    air_boost = 3.5 if a['air']<3 else 2.0
    proc = apply_high_shelf(proc, sr, 10000, air_boost*wet)

    # ── 7. Compresseur 2 — RVox style ────────────────
    ratio2 = 4.0 + comp_pct*4.0
    atk2   = 8 - comp_pct*6
    ra2    = rms_act(proc, sr)
    thr2   = max(-32., min(-6., ra2-1.5))
    mg2    = max(2., min(14., (-10-ra2)+(ra2-thr2)*(1-1/ratio2)))
    proc   = apply_comp(proc, sr, thr2, ratio2, atk2, 80, mg2*wet, knee_db=2)

    # ── 8. Saturation harmonique ─────────────────────
    drive   = 4.0 if a['crest']<12 else 2.5
    mix_sat = 0.14 if a['crest']<12 else 0.08
    proc = apply_saturation(proc, drive_db=drive*wet, mix=mix_sat*wet)

    # ── 9. Safety peak ───────────────────────────────
    proc = peak_normalize(proc, -0.5)

    # ── 10. SideWidener ──────────────────────────────
    width = 0.22 * wet
    left, right = apply_widener(proc, sr, width)

    # ── 11. Limiter stéréo ───────────────────────────
    left, right = apply_limiter_stereo(left, right, sr, -0.1)

    # ── Wet/Dry mix ──────────────────────────────────
    if wet < 1.0:
        dry   = peak_normalize(x.copy(), -0.3)
        left  = (wet*left  + (1-wet)*dry).astype(np.float32)
        right = (wet*right + (1-wet)*dry).astype(np.float32)
        left, right = apply_limiter_stereo(left, right, sr, -0.1)

    return left, right

# ═══════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════

st.markdown("<h1>🎤 VOIXRAP</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#555;font-size:11px;letter-spacing:3px;margin-top:-10px'>ANALYSE · DIAGNOSTIQUE · TRAITE — GRATUIT</p>", unsafe_allow_html=True)
st.markdown("<hr style='border-color:#222;margin:24px 0'>", unsafe_allow_html=True)

st.markdown("<h3>📂 Ta prise voix</h3>", unsafe_allow_html=True)
uploaded = st.file_uploader("WAV, MP3, M4A, FLAC, OGG", type=["wav","mp3","flac","ogg","m4a"], label_visibility="visible")

if uploaded is None:
    st.info("👆 Charge ta prise pour commencer.")
    st.stop()

with st.spinner("Chargement..."):
    try:
        import librosa
        suffix = "." + uploaded.name.rsplit(".",1)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read()); tmp_path=tmp.name
        x, sr = librosa.load(tmp_path, sr=44100, mono=True)
        x = (x - float(np.mean(x))).astype(np.float32)
        os.unlink(tmp_path)
    except Exception as e:
        st.error(f"❌ Impossible de lire le fichier : {e}"); st.stop()

with st.spinner("Analyse de ta prise en cours..."):
    a    = analyse_prise(x, sr)
    diag = generer_diagnostic(a)

st.markdown("<hr style='border-color:#222;margin:24px 0'>", unsafe_allow_html=True)
st.markdown("<h3>🔍 Diagnostic de ta prise</h3>", unsafe_allow_html=True)

for emoji, titre, detail in diag:
    couleur = {"🔴":"#ff3c3c","🟡":"#ff8c00","🟢":"#00ff88"}[emoji]
    st.markdown(f"""
    <div style='background:#111;border-left:3px solid {couleur};border-radius:8px;padding:12px 16px;margin:8px 0'>
        <div style='color:{couleur};font-size:13px;font-weight:bold'>{emoji} {titre}</div>
        <div style='color:#666;font-size:11px;margin-top:4px'>{detail}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr style='border-color:#222;margin:24px 0'>", unsafe_allow_html=True)
st.markdown("<h3>🎚️ Réglages</h3>", unsafe_allow_html=True)

col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    st.markdown("<p style='color:#ff3c3c;font-size:10px;letter-spacing:2px;margin-bottom:4px'>⚡ TRAITEMENT</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#444;font-size:9px'>0% brut → 100% full</p>", unsafe_allow_html=True)
    wet_pct = st.slider("wet", 0, 100, 100, 5, format="%d%%", label_visibility="collapsed")
with col_s2:
    st.markdown("<p style='color:#ff8c00;font-size:10px;letter-spacing:2px;margin-bottom:4px'>🔉 GRAVES</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#444;font-size:9px'>Low shelf 180 Hz</p>", unsafe_allow_html=True)
    graves_db = st.slider("graves", -12, 0, -6, 1, format="%d dB", label_visibility="collapsed")
with col_s3:
    st.markdown("<p style='color:#ffd700;font-size:10px;letter-spacing:2px;margin-bottom:4px'>🗜️ COMPRESSION</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#444;font-size:9px'>Léger → Agressif</p>", unsafe_allow_html=True)
    comp_pct = st.slider("comp", 0, 100, 50, 5, format="%d%%", label_visibility="collapsed")
with col_s4:
    st.markdown("<p style='color:#00ff88;font-size:10px;letter-spacing:2px;margin-bottom:4px'>✨ PRÉSENCE</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#444;font-size:9px'>3.5 kHz — voix en avant</p>", unsafe_allow_html=True)
    presence_db = st.slider("presence", -3, 6, 2, 1, format="%d dB", label_visibility="collapsed")

wet = wet_pct/100.0

if st.button("⚡ TRAITER MA VOIX"):
    with st.spinner("Traitement en cours..."):
        prog = st.progress(0, text="Analyse...")
        prog.progress(10, text="🎚️ EQ correctif...")
        prog.progress(25, text="🗜️ Compression maintien...")
        prog.progress(40, text="🎛️ Multibande C6...")
        prog.progress(60, text="👂 Triple de-essing...")
        prog.progress(75, text="✨ EQ boosts...")
        prog.progress(85, text="🗜️ Compression RVox...")
        prog.progress(92, text="🔥 Saturation...")
        prog.progress(97, text="🛑 Limiteur final...")
        left, right = traiter_prise(x, sr, a, wet, graves_db=graves_db, comp_pct=comp_pct/100, presence_db=presence_db)
        prog.progress(100, text="✅ Terminé !")

    mono_out  = (left+right)*0.5
    rms_out   = rms_act(mono_out, sr)
    pk_out    = dbv(np.max(np.abs(mono_out)))
    clips_out = int(np.sum(np.abs(mono_out)>=0.998))

    st.markdown("<hr style='border-color:#222;margin:24px 0'>", unsafe_allow_html=True)
    st.markdown("<h3>✅ Voix traitée — prête pour le beat</h3>", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Niveau RMS", f"{rms_out:.0f} dB")
    c2.metric("Peak",       f"{pk_out:.1f} dB")
    c3.metric("Clips",      f"{clips_out} {'✅' if clips_out==0 else '⚠️'}")
    c4.metric("Traitement", f"{wet_pct}%")

    wav_bytes = build_wav(left, right, sr)
    nom = uploaded.name.rsplit(".",1)[0] + f"_voixrap_v7.wav"

    st.markdown("<h3>🎧 Écoute avant / après</h3>", unsafe_allow_html=True)
    col_av, col_ap = st.columns(2)
    with col_av:
        st.markdown("<p style='color:#555;font-size:11px;letter-spacing:2px'>AVANT — BRUT</p>", unsafe_allow_html=True)
        dry_norm = peak_normalize(x.copy(), -0.3)
        wav_dry  = build_wav(dry_norm, dry_norm, sr)
        st.audio(wav_dry, format="audio/wav")
    with col_ap:
        st.markdown("<p style='color:#00ff88;font-size:11px;letter-spacing:2px'>APRÈS — TRAITÉ</p>", unsafe_allow_html=True)
        st.audio(wav_bytes, format="audio/wav")

    st.download_button(
        label="⬇ TÉLÉCHARGER MA VOIX",
        data=wav_bytes, file_name=nom, mime="audio/wav",
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
    </div>""", unsafe_allow_html=True)
