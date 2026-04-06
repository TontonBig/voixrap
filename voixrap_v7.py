import streamlit as st
import numpy as np
import io, wave, tempfile, os
from scipy import signal
from scipy.fft import rfft, rfftfreq

EPS = 1e-12

st.set_page_config(page_title="VoixRap — Ta voix qui claque", layout="centered", page_icon="🎤")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Mono:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Mono', monospace !important; background-color: #0a0a0a !important; color: #f0f0f0 !important; }
.stApp { background-color: #0a0a0a; }
h1 { font-family: 'Bebas Neue', sans-serif !important; font-size: 72px !important; letter-spacing: 6px !important; color: #ff3c3c !important; line-height: 1 !important; }
h3 { font-family: 'Space Mono', monospace !important; font-size: 11px !important; letter-spacing: 3px !important; color: #ff3c3c !important; text-transform: uppercase; }
.stButton > button { width: 100% !important; background: linear-gradient(135deg, #ff3c3c, #cc1a1a) !important; color: white !important; border: none !important; border-radius: 12px !important; font-family: 'Bebas Neue', sans-serif !important; font-size: 26px !important; letter-spacing: 4px !important; padding: 18px !important; box-shadow: 0 8px 32px rgba(255,60,60,0.3) !important; }
.stDownloadButton > button { width: 100% !important; background: linear-gradient(135deg, #00ff88, #00cc66) !important; color: #000 !important; border: none !important; border-radius: 12px !important; font-family: 'Bebas Neue', sans-serif !important; font-size: 26px !important; letter-spacing: 4px !important; padding: 18px !important; box-shadow: 0 8px 32px rgba(0,255,136,0.25) !important; }
footer { display: none !important; } #MainMenu { display: none !important; } header { display: none !important; }
</style>
""", unsafe_allow_html=True)

def lin(db):  return 10 ** (db / 20.0)
def dbv(v):   return 20.0 * np.log10(max(abs(float(v)), EPS))

def frames_fn(x, sr, ms=20, hop_ms=10):
    f=max(1,int(sr*ms/1000)); h=max(1,int(sr*hop_ms/1000))
    n=1+max(0,(len(x)-f)//h)
    idx=np.clip(np.arange(f)[None,:]+np.arange(n)[:,None]*h,0,len(x)-1)
    return x[idx]

def rms_db_frames(x, sr, ms=20, hop_ms=10):
    return 20*np.log10(np.sqrt(np.mean(frames_fn(x,sr,ms,hop_ms)**2,axis=1)+EPS))

def rms_act(x, sr):
    d=rms_db_frames(x,sr); a=d[d>-42]
    return float(np.percentile(a,50)) if a.size>5 else float(np.mean(d))

def peak_normalize(x, target_db=-0.3):
    pk=float(np.max(np.abs(x))+EPS); tg=lin(target_db)
    return (x*tg/pk).astype(np.float32) if pk>tg else x

def smooth_gain(gf, atk, rel):
    gs=np.zeros_like(gf); g=gf[0]
    for i,v in enumerate(gf):
        g+=(atk if v<g else rel)*(v-g); gs[i]=g
    return gs

def apply_hpf(x, sr, fc, order=4):
    fc=min(max(fc,1.),sr/2*0.95)
    b,a=signal.butter(order,fc/(sr/2),btype='highpass')
    return signal.lfilter(b,a,x).astype(np.float32)

def apply_lpf(x, sr, fc, order=4):
    fc=min(max(fc,1.),sr/2*0.95)
    b,a=signal.butter(order,fc/(sr/2),btype='lowpass')
    return signal.lfilter(b,a,x).astype(np.float32)

def apply_eq(x, sr, freq, Q, gain_db):
    freq=min(max(freq,1.),sr/2*0.95)
    A=10**(gain_db/40.); w0=2*np.pi*freq/sr; al=np.sin(w0)/(2*Q)
    b0=1+al*A; b1=-2*np.cos(w0); b2=1-al*A
    a0=1+al/A; a1=-2*np.cos(w0); a2=1-al/A
    return signal.lfilter(np.array([b0/a0,b1/a0,b2/a0]),np.array([1.,a1/a0,a2/a0]),x).astype(np.float32)

def apply_low_shelf(x, sr, fc, gain_db):
    fc=min(max(fc,1.),sr/2*0.95)
    A=10**(gain_db/40.); w0=2*np.pi*fc/sr; cosW=np.cos(w0); sinW=np.sin(w0); beta=np.sqrt(A)
    b0=A*((A+1)-(A-1)*cosW+beta*sinW); b1=2*A*((A-1)-(A+1)*cosW); b2=A*((A+1)-(A-1)*cosW-beta*sinW)
    a0=(A+1)+(A-1)*cosW+beta*sinW; a1=-2*((A-1)+(A+1)*cosW); a2=(A+1)+(A-1)*cosW-beta*sinW
    return signal.lfilter(np.array([b0/a0,b1/a0,b2/a0]),np.array([1.,a1/a0,a2/a0]),x).astype(np.float32)

def apply_high_shelf(x, sr, fc, gain_db):
    fc=min(max(fc,1.),sr/2*0.95)
    A=10**(gain_db/40.); w0=2*np.pi*fc/sr; cosW=np.cos(w0); sinW=np.sin(w0); beta=np.sqrt(A)
    b0=A*((A+1)+(A-1)*cosW+beta*sinW); b1=-2*A*((A-1)+(A+1)*cosW); b2=A*((A+1)+(A-1)*cosW-beta*sinW)
    a0=(A+1)-(A-1)*cosW+beta*sinW; a1=2*((A-1)-(A+1)*cosW); a2=(A+1)-(A-1)*cosW-beta*sinW
    return signal.lfilter(np.array([b0/a0,b1/a0,b2/a0]),np.array([1.,a1/a0,a2/a0]),x).astype(np.float32)

def apply_comp(x, sr, thr_db, ratio, atk_ms, rel_ms, makeup_db, knee_db=3.0):
    hop=max(1,int(sr*2/1000)); db_f=rms_db_frames(x,sr,5,2); fps=1000/2
    over=db_f-thr_db; hk=knee_db/2
    gr=np.where(over<=-hk,0.,np.where(over>=hk,-over*(1-1/ratio),-(over+hk)**2/(2*knee_db+EPS)*(1-1/ratio)))
    gr=np.where(db_f>-42,gr,0.)
    atk=1-np.exp(-1/max(1,fps*atk_ms/1000)); rel=1-np.exp(-1/max(1,fps*rel_ms/1000))
    gs=smooth_gain(gr,atk,rel)
    gi=np.interp(np.arange(len(x)),np.arange(len(gs))*hop,gs+makeup_db)
    return (x*lin(gi)).astype(np.float32)

def band_deess(x, sr, fc_low, fc_high, percentile=75, amount_db=-4.0):
    nyq=sr/2; fc1=min(max(fc_low/nyq,0.001),0.95); fc2=min(max(fc_high/nyq,fc1+0.001),0.99)
    b,a=signal.butter(2,[fc1,fc2],btype='bandpass'); band=signal.lfilter(b,a,x)
    hop=max(1,int(sr*2/1000))
    rms_b=np.sqrt(np.mean(frames_fn(band,sr,5,2)**2,axis=1)+EPS)
    thr_b=float(np.percentile(rms_b,percentile))
    gf=np.where(rms_b>thr_b,lin(amount_db),1.0)
    atk=1-np.exp(-1/max(1,sr*0.001/hop)); rel=1-np.exp(-1/max(1,sr*0.005/hop))
    gs=smooth_gain(gf,atk,rel)
    gi=np.interp(np.arange(len(x)),np.arange(len(gs))*hop,gs)
    return (x*gi).astype(np.float32)

def apply_multiband(x, sr, a):
    proc=x.copy()
    # Bande sub < 120Hz
    sub_band=apply_lpf(x,sr,120,4)
    sub_rms=rms_act(sub_band,sr)
    sub_comp=apply_comp(sub_band,sr,sub_rms-2,8.,10,40,-4.,2)
    proc=(proc-sub_band+sub_comp).astype(np.float32)
    # Bande lo-mid 120-500Hz
    lo_band=apply_hpf(apply_lpf(proc,sr,500,4),sr,120,4)
    lo_rms=rms_act(lo_band,sr)
    mud_ratio=5. if a['lo_mid']>35 else 3.5
    lo_comp=apply_comp(lo_band,sr,lo_rms-3,mud_ratio,12,80,0.,3)
    if a['lo_mid']>35: lo_comp=(lo_comp*lin(-2.)).astype(np.float32)
    proc=(proc-lo_band+lo_comp).astype(np.float32)
    # Bande présence 2k-6kHz
    pres_band=apply_hpf(apply_lpf(proc,sr,6000,4),sr,2000,4)
    pres_rms=rms_act(pres_band,sr)
    p_ratio=5. if a['presence']>30 else 3.
    pres_comp=apply_comp(pres_band,sr,pres_rms-3,p_ratio,5,40,0.,2)
    proc=(proc-pres_band+pres_comp).astype(np.float32)
    return proc

def apply_triple_deess(x, sr, a):
    proc=band_deess(x,sr,300,600,70,-4.)
    amount_h=-5. if a['presence']>25 else -3.
    proc=band_deess(proc,sr,2000,5000,72,amount_h)
    if a['sib_zone']>12:
        sib_f=a.get('sib_freq',7000)
        fc_lo=max(sib_f-2000,4000); fc_hi=min(sib_f+2000,sr/2*0.95)
        amount_s=max(-8.,-(a['sib_zone']-10)*0.35)
        proc=band_deess(proc,sr,fc_lo,fc_hi,75,amount_s)
    return proc

def apply_saturation(x, drive_db=3., mix=0.12):
    hot=x*lin(drive_db)
    sat=(np.tanh(hot*0.9)+0.08*np.tanh(hot*0.4)).astype(np.float32)
    ri=np.sqrt(np.mean(x**2)+EPS); rs=np.sqrt(np.mean(sat**2)+EPS)
    if rs>EPS: sat*=ri/rs
    return ((1-mix)*x+mix*sat).astype(np.float32)

def apply_limiter_stereo(left, right, sr, ceil_db=-0.1):
    ceil=lin(ceil_db); lh=max(1,int(sr*0.008))
    ll=np.concatenate([left[lh:],np.zeros(lh,dtype=np.float32)])
    lr=np.concatenate([right[lh:],np.zeros(lh,dtype=np.float32)])
    rel=1-np.exp(-1/(sr*0.060)); g=1.
    outl=np.zeros_like(left); outr=np.zeros_like(right)
    for i in range(len(left)):
        pk=max(abs(float(ll[i])),abs(float(lr[i])))
        ned=ceil/(pk+EPS) if pk>ceil else 1.
        if ned<g: g=ned
        else: g+=rel*(1-g)
        outl[i]=left[i]*g; outr[i]=right[i]*g
    pk_max=max(float(np.max(np.abs(outl))),float(np.max(np.abs(outr))))+EPS
    if pk_max>ceil*1.001:
        outl=(outl*ceil/pk_max).astype(np.float32)
        outr=(outr*ceil/pk_max).astype(np.float32)
    return outl.astype(np.float32),outr.astype(np.float32)

def apply_widener(mono, sr, width=0.20):
    delay=int(sr*0.012); side=np.zeros(len(mono),dtype=np.float32)
    if delay<len(mono):
        side[delay:]=mono[:-delay]*width
        b,a=signal.butter(2,200./(sr/2),btype='highpass')
        side=signal.lfilter(b,a,side).astype(np.float32)
    return (mono+side).astype(np.float32),(mono-side).astype(np.float32)

def build_wav(left, right, sr):
    def pcm24(x):
        x=np.clip(x,-1,1); iv=(x*8388607).astype(np.int32)
        out=np.zeros((len(iv),3),dtype=np.uint8)
        out[:,0]=(iv&0xFF).astype(np.uint8); out[:,1]=((iv>>8)&0xFF).astype(np.uint8); out[:,2]=((iv>>16)&0xFF).astype(np.uint8)
        return out.tobytes()
    il=np.empty(len(left)+len(right),dtype=np.float32); il[0::2]=left; il[1::2]=right
    buf=io.BytesIO()
    with wave.open(buf,'wb') as wf:
        wf.setnchannels(2); wf.setsampwidth(3); wf.setframerate(sr); wf.writeframes(pcm24(il))
    return buf.getvalue()

def analyse_prise(x, sr):
    r={}; seg=x[:min(len(x),sr*8)]; win=signal.windows.hann(len(seg),sym=False)
    P=(np.abs(rfft(seg*win))**2)+EPS; F=rfftfreq(len(seg),1./sr); tot=P.sum()
    def energie(fa,fb):
        m=(F>=fa)&(F<fb); return float(P[m].sum()/tot*100) if np.any(m) else 0.
    r['sub']=energie(20,100); r['graves']=energie(100,200); r['lo_mid']=energie(200,500)
    r['mid']=energie(500,2500); r['presence']=energie(2500,6000)
    r['sib_zone']=energie(6000,12000); r['air']=energie(12000,20000)
    if sr>12000:
        m_sib=(F>=5000)&(F<12000)
        r['sib_freq']=float(F[m_sib][np.argmax(P[m_sib])]) if np.any(m_sib) else 7000.
    else: r['sib_freq']=7000.
    r['rms']=rms_act(x,sr); r['peak']=dbv(np.max(np.abs(x))); r['clips']=int(np.sum(np.abs(x)>=0.998))
    db_f=rms_db_frames(x,sr,400,100); a_=db_f[db_f>-42]
    dr=float(np.percentile(a_,95)-np.percentile(a_,5)) if a_.size>10 else 20.
    crest=r['peak']-r['rms']; r['dr']=dr; r['crest']=crest
    r['deja_compresse']=dr<12. and crest<10.
    r['noise_floor']=float(np.percentile(db_f,5)); r['snr']=r['rms']-r['noise_floor']
    r['plos_risk']=energie(20,120)/(energie(500,3000)+EPS)>0.3
    env=np.abs(x); b_e,a_e=signal.butter(1,10/(sr/2),btype='lowpass'); env_s=signal.lfilter(b_e,a_e,env)
    peaks_idx=signal.find_peaks(env_s,height=np.max(env_s)*0.3,distance=sr//4)[0]; rev=False
    if len(peaks_idx)>1:
        decays=[]
        for pk in peaks_idx[:5]:
            end=min(pk+int(sr*0.3),len(env_s)-1)
            if end>pk: decays.append((env_s[pk]-env_s[end])/(env_s[pk]+EPS))
        if decays: rev=float(np.mean(decays))<0.5
    r['reverb']=rev
    return r

def generer_diagnostic(a):
    diag=[]
    if a['clips']>100: diag.append(("🔴","Ta voix sature fort",f"{a['clips']} points saturés — réenregistre si possible"))
    elif a['clips']>0: diag.append(("🟡","Légère saturation","Quelques points saturés"))
    if a['rms']<-32: diag.append(("🔴","Voix trop silencieuse","Monte le gain"))
    elif a['rms']<-24: diag.append(("🟡","Niveau un peu faible","Monte ton gain la prochaine fois"))
    else: diag.append(("🟢","Niveau correct",f"Bien enregistrée ({a['rms']:.0f} dBFS)"))
    if a['sub']>15: diag.append(("🔴","Trop de sub inutile","Énergie sous 100 Hz — pompe dans la compression"))
    if a['graves']>30: diag.append(("🔴","Voix trop grave","Trop d'énergie dans les graves"))
    elif a['lo_mid']>35: diag.append(("🟡","Beaucoup de boue (200-500 Hz)","Zone très chargée"))
    elif a['lo_mid']>25: diag.append(("🟡","Un peu de boue (200-500 Hz)","Zone chargée"))
    if a['presence']<8: diag.append(("🟡","Manque de présence","Pas assez d'énergie entre 2.5k et 6k"))
    elif a['presence']>35: diag.append(("🟡","Voix agressive","Trop d'énergie entre 2.5k et 6k"))
    if a['sib_zone']>20: diag.append(("🟡","Sibilances détectées",f"'S' agressifs autour de {a['sib_freq']:.0f} Hz"))
    if a['air']<2: diag.append(("🟡","Manque d'air","Pas assez d'énergie dans les hautes fréquences"))
    if a['snr']<15: diag.append(("🔴","Beaucoup de bruit de fond","Ferme les fenêtres, coupe la clim"))
    elif a['snr']<25: diag.append(("🟡","Un peu de bruit de fond","C'est gérable"))
    else: diag.append(("🟢","Prise silencieuse","Peu de bruit de fond"))
    if a['deja_compresse']: diag.append(("🟡","Déjà compressée","BandLab ou app mobile — on adapte"))
    if a['plos_risk']: diag.append(("🟡","Plosives détectées","Utilise un filtre anti-pop"))
    if a['reverb']: diag.append(("🟡","Reverb de pièce","Ta pièce résonne"))
    return diag

def traiter_prise(x, sr, a, wet=1.0, graves_db=-9, comp_pct=0.5, presence_db=2):
    if wet==0.0:
        left=peak_normalize(np.array(x,dtype=np.float32),-0.3); return left,left.copy()
    proc=x.copy()

    # 1. HPF 80Hz
    proc=apply_hpf(proc,sr,80,order=4)

    # 2. EQ correctif — boue et nasal
    mud=-5. if a['lo_mid']>35 else -3.
    proc=apply_eq(proc,sr,380,1.8,mud*wet)
    if a['lo_mid']>30 or a['mid']>30:
        proc=apply_eq(proc,sr,1000,1.2,-2.5*wet)

    # 3. Saturation AVANT compression — pas d harmoniques réinjectés après
    drive=3.5 if a['crest']<12 else 2.
    mix_sat=0.12 if a['crest']<12 else 0.07
    proc=apply_saturation(proc,drive_db=drive*wet,mix=mix_sat*wet)

    # 4. Comp 1 — Maintien, makeup fixe +4dB
    ratio1=2.+comp_pct*2.
    ra1=rms_act(proc,sr); thr1=max(-32.,min(-10.,ra1-3.))
    proc=apply_comp(proc,sr,thr1,ratio1,20,200,4.*wet,knee_db=4)

    # 5. Multibande — reconstruction par soustraction (phase OK)
    proc_mb=apply_multiband(proc,sr,a)
    proc=(wet*proc_mb+(1-wet)*proc).astype(np.float32)

    # 6. Triple de-ess
    proc=apply_triple_deess(proc,sr,a)

    # 7. Low shelf graves APRES compression — slider a un vrai impact
    proc=apply_low_shelf(proc,sr,180,float(graves_db)*wet)
    if graves_db<=-9:
        extra=apply_hpf(proc,sr,120,order=2)
        proc=(0.7*extra+0.3*proc).astype(np.float32)

    # 8. EQ boosts — présence slider + air
    proc=apply_eq(proc,sr,3500,1.,float(presence_db)*wet)
    air_boost=3.5 if a['air']<3 else 2.
    proc=apply_high_shelf(proc,sr,10000,air_boost*wet)

    # 9. Comp 2 — RVox, makeup fixe selon comp_pct
    ratio2=4.+comp_pct*4.; atk2=max(2.,8-comp_pct*6)
    ra2=rms_act(proc,sr); thr2=max(-32.,min(-6.,ra2-1.5))
    makeup2=4.+comp_pct*4.
    proc=apply_comp(proc,sr,thr2,ratio2,atk2,80,makeup2*wet,knee_db=2)

    # 10. UN SEUL peak normalize
    proc=peak_normalize(proc,-2.)

    # 11. Widener
    left,right=apply_widener(proc,sr,0.22*wet)

    # 12. UN SEUL limiter
    left,right=apply_limiter_stereo(left,right,sr,-0.1)

    # Wet/dry
    if wet<1.:
        dry=peak_normalize(x.copy(),-0.3)
        left=(wet*left+(1-wet)*dry).astype(np.float32)
        right=(wet*right+(1-wet)*dry).astype(np.float32)
        left=np.clip(left,-0.98,0.98).astype(np.float32)
        right=np.clip(right,-0.98,0.98).astype(np.float32)

    return left,right

# UI
st.markdown("<h1>🎤 VOIXRAP</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#555;font-size:11px;letter-spacing:3px;margin-top:-10px'>ANALYSE · DIAGNOSTIQUE · TRAITE — GRATUIT</p>", unsafe_allow_html=True)
st.markdown("<hr style='border-color:#222;margin:24px 0'>", unsafe_allow_html=True)

st.markdown("<h3>📂 Ta prise voix</h3>", unsafe_allow_html=True)
uploaded=st.file_uploader("WAV, MP3, M4A, FLAC, OGG",type=["wav","mp3","flac","ogg","m4a"],label_visibility="visible")

if uploaded is None:
    st.info("👆 Charge ta prise pour commencer."); st.stop()

with st.spinner("Chargement..."):
    try:
        import librosa
        suffix="."+uploaded.name.rsplit(".",1)[-1]
        with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tmp:
            tmp.write(uploaded.read()); tmp_path=tmp.name
        x,sr=librosa.load(tmp_path,sr=44100,mono=True)
        x=(x-float(np.mean(x))).astype(np.float32); os.unlink(tmp_path)
    except Exception as e:
        st.error(f"❌ Impossible de lire le fichier : {e}"); st.stop()

with st.spinner("Analyse..."):
    a=analyse_prise(x,sr); diag=generer_diagnostic(a)

st.markdown("<hr style='border-color:#222;margin:24px 0'>", unsafe_allow_html=True)
st.markdown("<h3>🔍 Diagnostic de ta prise</h3>", unsafe_allow_html=True)

for emoji,titre,detail in diag:
    couleur={"🔴":"#ff3c3c","🟡":"#ff8c00","🟢":"#00ff88"}[emoji]
    st.markdown(f"<div style='background:#111;border-left:3px solid {couleur};border-radius:8px;padding:12px 16px;margin:8px 0'><div style='color:{couleur};font-size:13px;font-weight:bold'>{emoji} {titre}</div><div style='color:#666;font-size:11px;margin-top:4px'>{detail}</div></div>",unsafe_allow_html=True)

st.markdown("<hr style='border-color:#222;margin:24px 0'>", unsafe_allow_html=True)
st.markdown("<h3>🎚️ Réglages</h3>", unsafe_allow_html=True)

col_s1,col_s2,col_s3,col_s4=st.columns(4)
with col_s1:
    st.markdown("<p style='color:#ff3c3c;font-size:10px;letter-spacing:2px'>⚡ TRAITEMENT</p>",unsafe_allow_html=True)
    st.markdown("<p style='color:#444;font-size:9px'>0% brut → 100% full</p>",unsafe_allow_html=True)
    wet_pct=st.slider("wet",0,100,100,5,format="%d%%",label_visibility="collapsed")
with col_s2:
    st.markdown("<p style='color:#ff8c00;font-size:10px;letter-spacing:2px'>🔉 GRAVES</p>",unsafe_allow_html=True)
    st.markdown("<p style='color:#444;font-size:9px'>Low shelf 180 Hz</p>",unsafe_allow_html=True)
    graves_db=st.slider("graves",-12,0,-9,1,format="%d dB",label_visibility="collapsed")
with col_s3:
    st.markdown("<p style='color:#ffd700;font-size:10px;letter-spacing:2px'>🗜️ COMPRESSION</p>",unsafe_allow_html=True)
    st.markdown("<p style='color:#444;font-size:9px'>Léger → Agressif</p>",unsafe_allow_html=True)
    comp_pct=st.slider("comp",0,100,50,5,format="%d%%",label_visibility="collapsed")
with col_s4:
    st.markdown("<p style='color:#00ff88;font-size:10px;letter-spacing:2px'>✨ PRÉSENCE</p>",unsafe_allow_html=True)
    st.markdown("<p style='color:#444;font-size:9px'>3.5 kHz</p>",unsafe_allow_html=True)
    presence_db=st.slider("presence",-3,6,2,1,format="%d dB",label_visibility="collapsed")

wet=wet_pct/100.

if st.button("⚡ TRAITER MA VOIX"):
    with st.spinner("Traitement en cours..."):
        prog=st.progress(0)
        prog.progress(10,text="🎚️ EQ correctif...")
        prog.progress(20,text="🔥 Saturation harmonique...")
        prog.progress(35,text="🗜️ Compression maintien...")
        prog.progress(50,text="🎛️ Multibande...")
        prog.progress(65,text="👂 Triple de-essing...")
        prog.progress(75,text="🔉 Coupe graves...")
        prog.progress(85,text="✨ EQ boosts + compression...")
        prog.progress(95,text="🛑 Limiteur final...")
        left,right=traiter_prise(x,sr,a,wet,graves_db=graves_db,comp_pct=comp_pct/100,presence_db=presence_db)
        prog.progress(100,text="✅ Terminé !")

    mono_out=(left+right)*0.5
    rms_out=rms_act(mono_out,sr); pk_out=dbv(np.max(np.abs(mono_out))); clips_out=int(np.sum(np.abs(mono_out)>=0.998))

    st.markdown("<hr style='border-color:#222;margin:24px 0'>", unsafe_allow_html=True)
    st.markdown("<h3>✅ Voix traitée</h3>", unsafe_allow_html=True)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("RMS",f"{rms_out:.0f} dB"); c2.metric("Peak",f"{pk_out:.1f} dB")
    c3.metric("Clips",f"{clips_out} {'✅' if clips_out==0 else '⚠️'}"); c4.metric("Traitement",f"{wet_pct}%")

    wav_bytes=build_wav(left,right,sr)
    nom=uploaded.name.rsplit(".",1)[0]+"_voixrap_v7.wav"

    st.markdown("<h3>🎧 Écoute avant / après</h3>", unsafe_allow_html=True)
    col_av,col_ap=st.columns(2)
    with col_av:
        st.markdown("<p style='color:#555;font-size:11px;letter-spacing:2px'>AVANT</p>",unsafe_allow_html=True)
        dry_norm=peak_normalize(x.copy(),-0.3)
        st.audio(build_wav(dry_norm,dry_norm,sr),format="audio/wav")
    with col_ap:
        st.markdown("<p style='color:#00ff88;font-size:11px;letter-spacing:2px'>APRÈS</p>",unsafe_allow_html=True)
        st.audio(wav_bytes,format="audio/wav")

    st.download_button(label="⬇ TÉLÉCHARGER MA VOIX",data=wav_bytes,file_name=nom,mime="audio/wav",use_container_width=True)

    st.markdown("<hr style='border-color:#1a1a1a;margin:24px 0'>", unsafe_allow_html=True)
    st.markdown("<div style='background:#111;border:1px solid #1a1a1a;border-radius:12px;padding:20px;font-size:11px;color:#555;line-height:2'>💡 <b style='color:#888'>Pour une meilleure prise :</b><br>📍 15 à 20 cm du micro<br>🔇 Ferme les fenêtres, coupe la clim<br>👕 Enregistre dans un placard plein de vêtements<br>📱 Sur BandLab : WAV brut sans effets</div>",unsafe_allow_html=True)
