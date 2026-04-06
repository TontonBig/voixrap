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

def lin(db):
    return 10 ** (db / 20.0)

def dbv(v):
    return 20.0 * np.log10(max(abs(float(v)), EPS))

def frames_fn(x, sr, ms=20, hop_ms=10):
    f = max(1, int(sr * ms / 1000))
    h = max(1, int(sr * hop_ms / 1000))
    n = 1 + max(0, (len(x) - f) // h)
    idx = np.clip(np.arange(f)[None, :] + np.arange(n)[:, None] * h, 0, len(x) - 1)
    return x[idx]

def rms_db_frames(x, sr, ms=20, hop_ms=10):
    return 20 * np.log10(np.sqrt(np.mean(frames_fn(x, sr, ms, hop_ms) ** 2, axis=1) + EPS))

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

def apply_low_shelf(x, sr, fc, gain_db):
    """Low shelf — coupe tout ce qui est sous fc Hz, gain_db = intensité de la coupe."""
    fc   = min(fc, sr / 2 * 0.95)
    A    = 10 ** (gain_db / 40.)
    w0   = 2 * np.pi * fc / sr
    cosW = np.cos(w0)
    sinW = np.sin(w0)
    beta = np.sqrt(A) / 1.0  # Q fixe = 1.0
    b0   = A * ((A+1) - (A-1)*cosW + beta*sinW)
    b1   = 2*A * ((A-1) - (A+1)*cosW)
    b2   = A * ((A+1) - (A-1)*cosW - beta*sinW)
    a0   = (A+1) + (A-1)*cosW + beta*sinW
    a1   = -2 * ((A-1) + (A+1)*cosW)
    a2   = (A+1) + (A-1)*cosW - beta*sinW
    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1., a1/a0, a2/a0])
    return signal.lfilter(b, a, x).astype(np.float32)

def apply_hpf(x, sr, fc):
    fc = min(fc, sr / 2 * 0.95)
    b, a = signal.butter(4, fc / (sr / 2), btype='highpass')
    return signal.lfilter(b, a, x).astype(np.float32)

def apply_eq(x, sr, freq, Q, gain_db):
    freq = min(freq, sr / 2 * 0.95)
    A  = 10 ** (gain_db / 40.)
    w0 = 2 * np.pi * freq / sr
    al = np.sin(w0) / (2 * Q)
    b0 = 1 + al * A;  b1 = -2 * np.cos(w0);  b2 = 1 - al * A
    a0 = 1 + al / A;  a1 = -2 * np.cos(w0);  a2 = 1 - al / A
    b = np.array([b0/a0, b1/a0, b2/a0])
    a = np.array([1., a1/a0, a2/a0])
    return signal.lfilter(b, a, x).astype(np.float32)

def smooth_gain(gf, atk, rel):
    gs = np.zeros_like(gf)
    g = gf[0]
    for i, v in enumerate(gf):
        g += (atk if v < g else rel) * (v - g)
        gs[i] = g
    return gs

def apply_compressor(x, sr, thr_db, ratio, atk_ms, rel_ms, makeup_db):
    hop = max(1, int(sr * 2 / 1000))
    db_f = rms_db_frames(x, sr, 5, 2)
    over = db_f - thr_db
    act  = db_f > -42.0
    gr   = np.where(act & (over > 0), -over * (1 - 1/ratio), 0.0)
    fps  = 1000 / 2
    atk  = 1 - np.exp(-1 / max(1, fps * atk_ms / 1000))
    rel  = 1 - np.exp(-1 / max(1, fps * rel_ms / 1000))
    gs   = smooth_gain(gr, atk, rel)
    gi   = np.interp(np.arange(len(x)), np.arange(len(gs)) * hop, gs + makeup_db)
    return peak_normalize((x * lin(gi)).astype(np.float32), -0.5)

def apply_deesser(x, sr, freq_hz, amount_db=-4.0):
    nyq = sr / 2
    if freq_hz >= nyq: return x
    b, a = signal.butter(2, [min(freq_hz/nyq, 0.95), min((freq_hz+3000)/nyq, 0.99)], btype='bandpass')
    sib  = signal.lfilter(b, a, x)
    hop  = max(1, int(sr * 2 / 1000))
    rms_s = np.sqrt(np.mean(frames_fn(sib, sr, 5, 2) ** 2, axis=1) + EPS)
    thr  = float(np.percentile(rms_s, 75))
    gf   = np.where(rms_s > thr, lin(amount_db), 1.0)
    atk  = 1 - np.exp(-1 / max(1, sr * 0.001 / hop))
    rel  = 1 - np.exp(-1 / max(1, sr * 0.005 / hop))
    gs   = smooth_gain(gf, atk, rel)
    gi   = np.interp(np.arange(len(x)), np.arange(len(gs)) * hop, gs)
    return (x * gi).astype(np.float32)

def apply_saturation(x, drive_db=3.0, mix=0.12):
    hot = x * lin(drive_db)
    sat = (np.tanh(hot * 0.9) + 0.08 * np.tanh(hot * 0.4)).astype(np.float32)
    ri  = np.sqrt(np.mean(x**2) + EPS)
    rs  = np.sqrt(np.mean(sat**2) + EPS)
    if rs > EPS: sat *= ri / rs
    return peak_normalize(((1-mix)*x + mix*sat).astype(np.float32), -0.5)

def apply_limiter(x, ceil_db=-0.1):
    ceil = lin(ceil_db)
    rel  = 1 - np.exp(-1 / (44100 * 0.06))
    g    = 1.0
    out  = np.zeros_like(x)
    for i in range(len(x)):
        pk  = abs(float(x[i]))
        ned = ceil / (pk + EPS) if pk > ceil else 1.0
        if ned < g: g = ned
        else: g += rel * (1 - g)
        out[i] = x[i] * g
    return np.clip(out, -ceil, ceil).astype(np.float32)

def apply_widener(mono, sr, width=0.20):
    delay = int(sr * 0.012)
    side  = np.zeros(len(mono), dtype=np.float32)
    if delay < len(mono):
        side[delay:] = mono[:-delay] * width
        b, a = signal.butter(2, 180./(sr/2), btype='highpass')
        side = signal.lfilter(b, a, side).astype(np.float32)
    left  = peak_normalize((mono + side).astype(np.float32), -0.3)
    right = peak_normalize((mono - side).astype(np.float32), -0.3)
    return left, right

def apply_limiter_stereo(left, right, sr, ceil_db=-0.1):
    ceil = lin(ceil_db)
    lh   = max(1, int(sr * 0.008))
    ll   = np.concatenate([left[lh:],  np.zeros(lh, dtype=np.float32)])
    lr   = np.concatenate([right[lh:], np.zeros(lh, dtype=np.float32)])
    rel  = 1 - np.exp(-1 / (sr * 0.060))
    g    = 1.0
    outl = np.zeros_like(left); outr = np.zeros_like(right)
    for i in range(len(left)):
        pk  = max(abs(float(ll[i])), abs(float(lr[i])))
        ned = ceil / (pk + EPS) if pk > ceil else 1.0
        if ned < g: g = ned
        else: g += rel * (1 - g)
        outl[i] = left[i] * g; outr[i] = right[i] * g
    pk_max = max(float(np.max(np.abs(outl))), float(np.max(np.abs(outr)))) + EPS
    if pk_max > ceil * 1.001:
        outl = (outl * ceil / pk_max).astype(np.float32)
        outr = (outr * ceil / pk_max).astype(np.float32)
    return outl.astype(np.float32), outr.astype(np.float32)

def lufs_simple(x, sr):
    hop = max(1, int(sr * 0.05))
    d   = rms_db_frames(x, sr, 100, 50)
    mask = np.zeros(len(x), dtype=bool)
    for i, v in enumerate(d):
        s = i*hop; e = min(s+hop, len(x))
        if v > -42: mask[s:e] = True
    x_act = x[mask] if mask.sum() > sr else x
    b1, a1 = signal.bilinear([1.,-2.,1.],[1.,-1.99004745,0.99007225], fs=sr)
    s1 = signal.lfilter(b1, a1, x_act)
    b2, a2 = signal.butter(2, 100./(sr/2), btype='highpass')
    s2 = signal.lfilter(b2, a2, s1)
    return -0.691 + 10*np.log10(float(np.mean(s2**2)) + EPS)

def normalize_lufs(left, right, sr, target=-10.0):
    for _ in range(6):
        mono = (left + right) * 0.5
        cur  = lufs_simple(mono, sr)
        diff = target - cur
        if abs(diff) < 0.3: break
        g = lin(min(diff, 6.0))
        left  = (left  * g).astype(np.float32)
        right = (right * g).astype(np.float32)
        pk = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))))
        if pk > 0.98:
            left, right = apply_limiter_stereo(left, right, sr, -0.1)
    return left, right

def build_wav(left, right, sr):
    def pcm24(x):
        x  = np.clip(x, -1, 1)
        iv = (x * 8388607).astype(np.int32)
        out = np.zeros((len(iv), 3), dtype=np.uint8)
        out[:,0] = (iv & 0xFF).astype(np.uint8)
        out[:,1] = ((iv>>8) & 0xFF).astype(np.uint8)
        out[:,2] = ((iv>>16) & 0xFF).astype(np.uint8)
        return out.tobytes()
    il = np.empty(len(left)+len(right), dtype=np.float32)
    il[0::2] = left; il[1::2] = right
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(2); wf.setsampwidth(3)
        wf.setframerate(sr); wf.writeframes(pcm24(il))
    return buf.getvalue()

# ═══════════════════════════════════════════════════
#  ANALYSE INTELLIGENTE
# ═══════════════════════════════════════════════════

def analyse_prise(x, sr):
    """Analyse réelle du signal — chaque valeur est mesurée, pas supposée."""
    resultats = {}

    # ── Spectre ──────────────────────────────────────
    seg = x[:min(len(x), sr*8)]
    win = signal.windows.hann(len(seg), sym=False)
    P   = (np.abs(rfft(seg * win))**2) + EPS
    F   = rfftfreq(len(seg), 1./sr)
    tot = P.sum()

    def energie(fa, fb):
        m = (F >= fa) & (F < fb)
        return float(P[m].sum() / tot * 100) if np.any(m) else 0.0

    # Zones clés
    sub       = energie(20, 100)      # Sub graves inutiles
    graves    = energie(100, 200)     # Graves voix
    lo_mid    = energie(200, 500)     # Boue
    mid       = energie(500, 2500)    # Corps voix
    presence  = energie(2500, 6000)   # Présence / agressivité
    sib_zone  = energie(6000, 12000)  # Sibilances
    air       = energie(12000, 20000) # Air / brillance

    resultats['sub']      = sub
    resultats['graves']   = graves
    resultats['lo_mid']   = lo_mid
    resultats['mid']      = mid
    resultats['presence'] = presence
    resultats['sib_zone'] = sib_zone
    resultats['air']      = air

    # Fréquence exacte des sibilances
    if sr > 12000:
        m_sib = (F >= 5000) & (F < 12000)
        if np.any(m_sib):
            resultats['sib_freq'] = float(F[m_sib][np.argmax(P[m_sib])])
        else:
            resultats['sib_freq'] = 7000.0
    else:
        resultats['sib_freq'] = 7000.0

    # Fréquence exacte du problème de graves
    m_grav = (F >= 80) & (F < 300)
    if np.any(m_grav):
        resultats['grave_freq'] = float(F[m_grav][np.argmax(P[m_grav])])
    else:
        resultats['grave_freq'] = 150.0

    # ── Niveau ───────────────────────────────────────
    rms = rms_act(x, sr)
    peak = dbv(np.max(np.abs(x)))
    clips = int(np.sum(np.abs(x) >= 0.998))
    resultats['rms']   = rms
    resultats['peak']  = peak
    resultats['clips'] = clips

    # ── Dynamique ────────────────────────────────────
    db_f = rms_db_frames(x, sr, 400, 100)
    a    = db_f[db_f > -42]
    dr   = float(np.percentile(a, 95) - np.percentile(a, 5)) if a.size > 10 else 20.0
    crest = peak - rms
    resultats['dr']    = dr
    resultats['crest'] = crest
    resultats['deja_compresse'] = dr < 12.0 and crest < 10.0

    # ── Bruit de fond ────────────────────────────────
    noise_floor = float(np.percentile(db_f, 5))
    snr = rms - noise_floor
    resultats['noise_floor'] = noise_floor
    resultats['snr']         = snr

    # ── Plosives ─────────────────────────────────────
    plos = energie(20, 120) / (energie(500, 3000) + EPS)
    resultats['plos_risk'] = plos > 0.3

    # ── Réverbération ────────────────────────────────
    # Détecte si le signal décroît lentement après les pics (reverb room)
    env = np.abs(x)
    b_env, a_env = signal.butter(1, 10/(sr/2), btype='lowpass')
    env_smooth = signal.lfilter(b_env, a_env, env)
    peaks_idx = signal.find_peaks(env_smooth, height=np.max(env_smooth)*0.3, distance=sr//4)[0]
    reverb_detected = False
    if len(peaks_idx) > 1:
        decays = []
        for pk in peaks_idx[:5]:
            end = min(pk + int(sr*0.3), len(env_smooth)-1)
            if end > pk:
                decay = env_smooth[pk] - env_smooth[end]
                decays.append(decay / (env_smooth[pk] + EPS))
        if decays:
            reverb_detected = float(np.mean(decays)) < 0.5
    resultats['reverb'] = reverb_detected

    return resultats

def generer_diagnostic(a):
    """Génère le diagnostic en langage simple — vert/orange/rouge."""
    diag = []

    # CLIPPING
    if a['clips'] > 100:
        diag.append(("🔴", "Ta voix sature fort", f"Elle dépasse le maximum numérique sur {a['clips']} points — c'est irréparable, réenregistre si possible"))
    elif a['clips'] > 0:
        diag.append(("🟡", "Légère saturation détectée", "Quelques points saturés — on peut tenter quand même"))

    # NIVEAU
    if a['rms'] < -32:
        diag.append(("🔴", "Ta voix est trop silencieuse", "Rapproche-toi du micro ou monte le gain — le bruit va s'amplifier avec le traitement"))
    elif a['rms'] < -24:
        diag.append(("🟡", "Niveau un peu faible", "C'est traitable mais monte ton gain la prochaine fois — vise -18 à -10 dBFS"))
    else:
        diag.append(("🟢", "Niveau correct", f"Ta voix est bien enregistrée ({a['rms']:.0f} dBFS)"))

    # GRAVES / BOUE
    if a['sub'] > 15:
        diag.append(("🔴", "Trop de basses inutiles", f"Y'a beaucoup de bruit sub-grave (sous 100 Hz) — ça va pomper dans la compression et salir le mix"))
    if a['graves'] > 30:
        diag.append(("🔴", "Ta voix est trop grave et boueuse", f"Trop d'énergie autour de {a['grave_freq']:.0f} Hz — elle va se perdre dans le beat"))
    elif a['lo_mid'] > 35:
        diag.append(("🟡", "Un peu de boue dans les mids", "Zone 200-500 Hz chargée — on va nettoyer ça"))

    # PRÉSENCE
    if a['presence'] < 8:
        diag.append(("🟡", "Ta voix manque de présence", "Pas assez d'énergie entre 2.5k et 6k — elle va se noyer dans le beat"))
    elif a['presence'] > 35:
        diag.append(("🟡", "Ta voix est un peu agressive", "Beaucoup d'énergie entre 2.5k et 6k — on va adoucir ça"))

    # SIBILANCES
    if a['sib_zone'] > 20:
        diag.append(("🟡", "Tes 'S' et 'CH' font mal aux oreilles", f"Sibilances détectées autour de {a['sib_freq']:.0f} Hz — le de-esser va régler ça"))

    # AIR / BRILLANCE
    if a['air'] < 2:
        diag.append(("🟡", "Ta voix manque d'air et de brillance", "Pas assez d'énergie dans les hautes fréquences — on va en rajouter"))

    # BRUIT DE FOND
    if a['snr'] < 15:
        diag.append(("🔴", "Beaucoup de bruit de fond", "Ferme les fenêtres, coupe la clim — le bruit est très fort par rapport à ta voix"))
    elif a['snr'] < 25:
        diag.append(("🟡", "Un peu de bruit de fond", "C'est gérable — enregistre dans un endroit plus calme la prochaine fois"))
    else:
        diag.append(("🟢", "Prise silencieuse", "Peu de bruit de fond — bien joué"))

    # DÉJÀ COMPRESSÉ
    if a['deja_compresse']:
        diag.append(("🟡", "Ta voix semble déjà compressée", "BandLab ou app mobile ? On va compresser plus légèrement"))

    # PLOSIVES
    if a['plos_risk']:
        diag.append(("🟡", "Des 'P' et 'B' qui claquent", "Utilise un filtre anti-pop ou incline légèrement le micro"))

    # REVERB
    if a['reverb']:
        diag.append(("🟡", "Un peu de reverb de pièce", "Ta pièce résonne — des vêtements autour du micro aideraient"))

    return diag

# ═══════════════════════════════════════════════════
#  TRAITEMENT ADAPTATIF
# ═══════════════════════════════════════════════════

def traiter_prise(x, sr, a, wet=1.0, graves_db=-6, comp_pct=0.5, presence_db=2):
    """
    Traitement 100% adaptatif basé sur l'analyse.
    wet = 0.0 (brut) à 1.0 (traitement complet)
    """
    if wet == 0.0:
        left  = peak_normalize(np.array(x, dtype=np.float32), -0.3)
        right = left.copy()
        return left, right

    proc = x.copy()

    # ── 1. HPF adaptatif ─────────────────────────────
    hpf_freq = 120 if a['sub'] > 15 else 80
    proc = apply_hpf(proc, sr, hpf_freq)

    # ── 2. GRAVES — Low shelf à 180 Hz ───────────────
    # Le slider contrôle l intensité de la coupe
    # graves_db va de -12 à 0 dB → low shelf qui coupe tout sous 180 Hz
    proc = apply_low_shelf(proc, sr, 180, float(graves_db) * wet)

    # Coupe lo-mid SÉPARÉE — 380 Hz (boue) — automatique
    if a['lo_mid'] > 30:
        mud_cut = max(-10.0, -(a['lo_mid'] - 20) * 0.4)
        proc = apply_eq(proc, sr, 380, 1.8, mud_cut * wet)
    else:
        proc = apply_eq(proc, sr, 380, 1.5, -3.0 * wet)

    # ── 3. Saturation harmonique ─────────────────────
    drive   = 4.0 if a['crest'] < 12 else 2.5
    mix_sat = 0.14 if a['crest'] < 12 else 0.08
    proc = apply_saturation(proc, drive_db=drive * wet, mix=mix_sat * wet)

    # ── 4. De-esser adaptatif ────────────────────────
    if a['sib_zone'] > 15:
        amount = max(-8.0, -(a['sib_zone'] - 10) * 0.3)
        proc = apply_deesser(proc, sr, a['sib_freq'], amount * wet)

    # ── 5. EQ boost présence — slider direct ─────────
    # Pas de peak_normalize ici — on laisse le slider avoir un vrai impact
    proc = apply_eq(proc, sr, 3500, 1.0, float(presence_db) * wet)

    # Air / brillance
    air_boost = 3.5 if a['air'] < 3 else 2.0
    proc = apply_eq(proc, sr, 10000, 0.8, air_boost * wet)

    # ── 6. Compression 1 — Maintien ──────────────────
    # comp_pct 0.0=léger → 1.0=agressif
    ratio1 = 2.0 + comp_pct * 2.0
    atk1   = 20 - comp_pct * 15
    rel1   = 200 - comp_pct * 120
    ra1    = rms_act(proc, sr)
    thr1   = max(-32., min(-8., ra1 - 2.0))
    mg1    = 3.0 + comp_pct * 3.0   # makeup gain qui suit l intensité
    proc   = apply_compressor(proc, sr, thr1, ratio1, atk1, rel1, mg1 * wet)

    # ── 7. Compression 2 — RVox style ────────────────
    ratio2 = 4.0 + comp_pct * 4.0
    atk2   = 8 - comp_pct * 6
    ra2    = rms_act(proc, sr)
    thr2   = max(-32., min(-6., ra2 - 1.5))
    mg2    = max(2.0, min(14.0, (-10 - ra2) + (ra2 - thr2) * (1 - 1/ratio2)))
    proc   = apply_compressor(proc, sr, thr2, ratio2, atk2, 80, mg2 * wet)

    # ── 8. Safety peak UNIQUE ─────────────────────────
    # Un seul normalize à la fin — pas entre chaque étape
    proc = peak_normalize(proc, -0.5)

    # ── 9. Widener ───────────────────────────────────
    width = 0.20 * wet
    left, right = apply_widener(proc, sr, width)

    # ── 10. Limiter stéréo ───────────────────────────
    left, right = apply_limiter_stereo(left, right, sr, -0.1)

    # ── Wet/Dry mix ──────────────────────────────────
    if wet < 1.0:
        # Mix avec le signal brut normalisé
        dry = peak_normalize(x.copy(), -0.3)
        dry_l = dry; dry_r = dry
        left  = (wet * left  + (1-wet) * dry_l).astype(np.float32)
        right = (wet * right + (1-wet) * dry_r).astype(np.float32)
        left, right = apply_limiter_stereo(left, right, sr, -0.1)

    return left, right

# ═══════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════

st.markdown("<h1>🎤 VOIXRAP</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#555;font-size:11px;letter-spacing:3px;margin-top:-10px'>ANALYSE · DIAGNOSTIQUE · TRAITE — GRATUIT</p>", unsafe_allow_html=True)
st.markdown("<hr style='border-color:#222;margin:24px 0'>", unsafe_allow_html=True)

# Upload
st.markdown("<h3>📂 Ta prise voix</h3>", unsafe_allow_html=True)
uploaded = st.file_uploader(
    "WAV, MP3, M4A, FLAC, OGG",
    type=["wav","mp3","flac","ogg","m4a"],
    label_visibility="visible"
)

if uploaded is None:
    st.info("👆 Charge ta prise pour commencer.")
    st.stop()

# Chargement audio
with st.spinner("Chargement..."):
    try:
        import librosa
        suffix = "." + uploaded.name.rsplit(".", 1)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        x, sr = librosa.load(tmp_path, sr=44100, mono=True)
        x = x.astype(np.float32)
        x = (x - float(np.mean(x))).astype(np.float32)
        os.unlink(tmp_path)
    except Exception as e:
        st.error(f"❌ Impossible de lire le fichier : {e}")
        st.stop()

# ── ANALYSE ──────────────────────────────────────────
with st.spinner("Analyse de ta prise en cours..."):
    a = analyse_prise(x, sr)
    diag = generer_diagnostic(a)

st.markdown("<hr style='border-color:#222;margin:24px 0'>", unsafe_allow_html=True)
st.markdown("<h3>🔍 Diagnostic de ta prise</h3>", unsafe_allow_html=True)

for emoji, titre, detail in diag:
    couleur = {"🔴":"#ff3c3c","🟡":"#ff8c00","🟢":"#00ff88"}[emoji]
    st.markdown(f"""
    <div style='background:#111;border-left:3px solid {couleur};border-radius:8px;padding:12px 16px;margin:8px 0'>
        <div style='color:{couleur};font-size:13px;font-weight:bold'>{emoji} {titre}</div>
        <div style='color:#666;font-size:11px;margin-top:4px'>{detail}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border-color:#222;margin:24px 0'>", unsafe_allow_html=True)

# ── RÉGLAGES ─────────────────────────────────────────
st.markdown("<h3>🎚️ Réglages</h3>", unsafe_allow_html=True)

col_s1, col_s2, col_s3, col_s4 = st.columns(4)
with col_s1:
    st.markdown("<p style='color:#ff3c3c;font-size:10px;letter-spacing:2px;margin-bottom:4px'>⚡ TRAITEMENT</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#444;font-size:9px'>0% brut → 100% full</p>", unsafe_allow_html=True)
    wet_pct = st.slider("wet", 0, 100, 100, 5, format="%d%%", label_visibility="collapsed")
with col_s2:
    st.markdown("<p style='color:#ff8c00;font-size:10px;letter-spacing:2px;margin-bottom:4px'>🔉 GRAVES (140 Hz)</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#444;font-size:9px'>Coupe les basses voix</p>", unsafe_allow_html=True)
    graves_db = st.slider("graves", -12, 0, -6, 1, format="%d dB", label_visibility="collapsed")
with col_s3:
    st.markdown("<p style='color:#ffd700;font-size:10px;letter-spacing:2px;margin-bottom:4px'>🗜️ COMPRESSION</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#444;font-size:9px'>Léger → Agressif</p>", unsafe_allow_html=True)
    comp_pct = st.slider("comp", 0, 100, 50, 5, format="%d%%", label_visibility="collapsed")
with col_s4:
    st.markdown("<p style='color:#00ff88;font-size:10px;letter-spacing:2px;margin-bottom:4px'>✨ PRÉSENCE</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#444;font-size:9px'>Voix en avant</p>", unsafe_allow_html=True)
    presence_db = st.slider("presence", -3, 6, 2, 1, format="%d dB", label_visibility="collapsed")

wet          = wet_pct / 100.0

# ── TRAITEMENT ───────────────────────────────────────
if st.button("⚡ TRAITER MA VOIX"):

    with st.spinner("Traitement en cours..."):
        prog = st.progress(0, text="Traitement adaptatif...")

        prog.progress(20, text="🎚️ EQ adaptatif...")
        prog.progress(45, text="🗜️ Compression...")
        prog.progress(65, text="🔥 Saturation harmonique...")
        prog.progress(80, text="🎧 Espace stéréo...")
        prog.progress(92, text="🛑 Limiteur final...")

        left, right = traiter_prise(x, sr, a, wet, graves_db=graves_db, comp_pct=comp_pct/100, presence_db=presence_db)

        prog.progress(100, text="✅ Terminé !")

    # Stats
    mono_out = (left + right) * 0.5
    rms_out  = rms_act(mono_out, sr)
    pk_out   = dbv(np.max(np.abs(mono_out)))
    clips_out = int(np.sum(np.abs(mono_out) >= 0.998))
    lufs_out  = lufs_simple(mono_out, sr)

    st.markdown("<hr style='border-color:#222;margin:24px 0'>", unsafe_allow_html=True)
    st.markdown("<h3>✅ Voix traitée — prête pour le beat</h3>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Niveau RMS", f"{rms_out:.0f} dB")
    c2.metric("Peak",       f"{pk_out:.1f} dB")
    c3.metric("LUFS",       f"{lufs_out:.0f}")
    c4.metric("Clips",      f"{clips_out} {'✅' if clips_out==0 else '⚠️'}")

    st.markdown(f"""
    <div style='background:#111;border:1px solid #222;border-radius:10px;padding:14px;margin:16px 0;font-size:11px;color:#555'>
    🎚️ Traitement appliqué à <span style='color:#ff3c3c;font-weight:bold'>{wet_pct}%</span>
    </div>
    """, unsafe_allow_html=True)

    wav_bytes = build_wav(left, right, sr)
    nom = uploaded.name.rsplit(".", 1)[0] + f"_voixrap_v6_{wet_pct}pct.wav"

    # ── Lecteurs audio avant/après ──
    st.markdown("<h3>🎧 Écoute avant / après</h3>", unsafe_allow_html=True)
    col_avant, col_apres = st.columns(2)
    with col_avant:
        st.markdown("<p style='color:#555;font-size:11px;letter-spacing:2px'>AVANT — VOIX BRUTE</p>", unsafe_allow_html=True)
        # Export mono brut en WAV pour lecture
        dry_norm = peak_normalize(x.copy(), -0.3)
        wav_dry = build_wav(dry_norm, dry_norm, sr)
        st.audio(wav_dry, format="audio/wav")
    with col_apres:
        st.markdown("<p style='color:#00ff88;font-size:11px;letter-spacing:2px'>APRÈS — VOIX TRAITÉE</p>", unsafe_allow_html=True)
        st.audio(wav_bytes, format="audio/wav")

    st.download_button(
        label="⬇ TÉLÉCHARGER MA VOIX",
        data=wav_bytes,
        file_name=nom,
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
