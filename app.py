import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import re

st.set_page_config(
    page_title="Prediksi Kelayakan Pinjaman",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════════════
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
D = st.session_state.dark_mode

P = {
    "bg"     : "#0d1117" if D else "#f0f4f8",
    "bg2"    : "#161b22" if D else "#ffffff",
    "bg3"    : "#21262d" if D else "#f6f8fa",
    "tx"     : "#e6edf3" if D else "#1c2526",
    "tx2"    : "#8b949e" if D else "#57606a",
    "bdr"    : "#30363d" if D else "#d0d7de",
    "gold"   : "#FFD700" if D else "#9a6700",
    "green"  : "#3fb950" if D else "#1a7f37",
    "red"    : "#f78166" if D else "#cf222e",
    "blue"   : "#58a6ff" if D else "#0969da",
    "orange" : "#e3a000" if D else "#9a5700",
    "s_bg"   : "#1a4d2e" if D else "#dafbe1",
    "e_bg"   : "#4d1a1a" if D else "#ffebe9",
    "w_bg"   : "#3d2e00" if D else "#fff3cd",
    "grad"   : "linear-gradient(135deg,#0f2027,#203a43,#2c5364)" if D
               else "linear-gradient(135deg,#e8f4fd,#c9e6fa,#a8d5f0)",
    "ht"     : "#ffffff"  if D else "#1c3d5a",
    "hs"     : "#a8d8f0"  if D else "#2d6a9f",
}

st.markdown(f"""<style>
.stApp,.main .block-container{{background:{P['bg']} !important;color:{P['tx']} !important}}
section[data-testid="stSidebar"]{{background:{P['bg2']} !important;border-right:1px solid {P['bdr']} !important}}
section[data-testid="stSidebar"] *{{color:{P['tx']} !important}}
h1,h2,h3,h4,p,label,span,div{{color:{P['tx']} !important}}
.stCaption{{color:{P['tx2']} !important}}
hr{{border-color:{P['bdr']} !important}}
input,textarea,[data-baseweb="input"] input{{
    background:{P['bg3']} !important;color:{P['tx']} !important;
    border:1px solid {P['bdr']} !important;border-radius:8px !important}}
input:focus{{border-color:{P['gold']} !important;box-shadow:0 0 0 2px {P['gold']}44 !important}}
[data-baseweb="select"]>div{{background:{P['bg3']} !important;border:1px solid {P['bdr']} !important;
    color:{P['tx']} !important;border-radius:8px !important}}
[data-baseweb="popover"] *{{background:{P['bg2']} !important;color:{P['tx']} !important}}
[data-baseweb="menu"] li:hover{{background:{P['bg3']} !important}}
.stTextInput label,.stNumberInput label,.stSelectbox label,
.stRadio>label,div[class*="radioLabel"] p{{
    color:{P['tx']} !important;font-weight:600 !important;font-size:.9em !important}}
[data-testid="stMetric"]{{background:{P['bg2']} !important;border:1px solid {P['bdr']} !important;
    border-radius:12px !important;padding:14px 18px !important}}
[data-testid="stMetricValue"]{{color:{P['gold']} !important;font-weight:900 !important}}
[data-testid="stMetricLabel"]{{color:{P['tx2']} !important}}
.stButton>button[kind="primary"]{{
    background:linear-gradient(135deg,{P['gold']},{P['gold']}cc) !important;
    color:#000 !important;font-weight:800 !important;border:none !important;
    border-radius:12px !important;padding:14px !important;font-size:1.1em !important;
    letter-spacing:.5px !important;box-shadow:0 4px 14px {P['gold']}44 !important}}
.stButton>button[kind="primary"]:hover{{transform:translateY(-2px) !important;
    box-shadow:0 6px 20px {P['gold']}66 !important}}
.stButton>button[kind="secondary"]{{background:{P['bg3']} !important;color:{P['tx']} !important;
    border:1px solid {P['bdr']} !important;border-radius:20px !important;
    padding:6px 16px !important;font-size:.88em !important}}
[data-testid="stDataFrame"] table{{background:{P['bg2']} !important;color:{P['tx']} !important}}
[data-testid="stDataFrame"] th{{background:{P['bg3']} !important;color:{P['gold']} !important;font-weight:700 !important}}
[data-testid="stDataFrame"] td{{color:{P['tx']} !important}}
[data-testid="stDataFrame"] tr:hover td{{background:{P['bg3']} !important}}
[data-testid="stProgressBar"]>div>div{{background:{P['green']} !important}}
[data-testid="stExpander"]{{background:{P['bg2']} !important;border:1px solid {P['bdr']} !important;border-radius:10px !important}}
.idr-hint{{font-size:.83em;color:{P['tx2']};padding:2px 10px 6px;letter-spacing:.3px}}
.idr-hint b{{color:{P['gold']}}}
.warn-box{{background:{P['w_bg']};border-left:3px solid {P['orange']};border-radius:0 8px 8px 0;
    padding:9px 14px;margin:4px 0;font-size:.88em}}
.tip-row{{display:flex;justify-content:space-between;padding:5px 0;
    border-bottom:1px solid {P['bdr']};font-size:.87em}}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def fmt(v):
    return f"{int(round(v)):,}".replace(",", ".")

def idr_field(label, default, help_text=None):
    raw = st.text_input(label, value=fmt(default),
                        help=help_text or f"Contoh: {fmt(default)}",
                        placeholder=fmt(default))
    clean = re.sub(r"[^\d]", "", raw)
    val   = int(clean) if clean else default
    val   = max(0, min(500_000_000, val))
    st.markdown(f'<div class="idr-hint">💰 Rp <b>{fmt(val)}</b></div>',
                unsafe_allow_html=True)
    return val

def star(n, total=5):
    return "★" * n + "☆" * (total - n)

# Income range from training data (IDR):
# ApplicantIncome in INR: min≈150, max≈81000 → IDR: min≈28K, max≈15.2M
# Typical median ≈ 3.8M IDR/month
INCOME_MIN_TRAIN = 200_000       # approx training minimum in IDR
INCOME_MED_TRAIN = 3_800_000     # approx training median in IDR
LOAN_MAX_TRAIN   = 130_000_000   # approx training maximum in IDR

@st.cache_resource
def load_model():
    model  = joblib.load("loan_model.pkl")
    scaler = joblib.load("loan_scaler.pkl")
    with open("model_info.json") as f:
        info = json.load(f)
    return model, scaler, info

model, scaler, info = load_model()

# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
bc = "#f0c040" if D else "#2d6a9f"
st.markdown(f"""
<div style="background:{P['grad']};padding:44px 40px 36px;border-radius:18px;
     text-align:center;margin-bottom:24px;box-shadow:0 8px 32px rgba(0,0,0,.3)">
  <div style="border:2px solid {bc};border-radius:14px;padding:26px 32px">
    <p style="color:{P['gold']};font-size:.85em;letter-spacing:4px;margin:0 0 6px">
      🏦 MACHINE LEARNING PROJECT</p>
    <h1 style="color:{P['ht']};font-size:2em;font-weight:900;margin:10px 0 6px">
      Prediksi Kelayakan Pinjaman</h1>
    <p style="color:{P['hs']};font-size:1em;margin:0 0 18px">
      Random Forest Classifier · Dataset Rupiah Indonesia (IDR)</p>
    <hr style="border-color:{bc}66;margin:12px 0">
    <div style="display:flex;justify-content:center;gap:8px;flex-wrap:wrap;margin-top:14px">
      {"".join(f'<span style="background:{P["bg3"]};color:{P["tx"]};padding:4px 14px;border-radius:20px;font-size:.8em;border:1px solid {P["bdr"]}">{t}</span>'
               for t in ["📊 EDA","🧹 Cleansing","🤖 6 Model","⚙️ GridSearchCV","🚀 Streamlit"])}
    </div>
  </div>
</div>""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
c1.metric("🤖 Model",    "Random Forest")
c2.metric("✅ Accuracy", f"{info.get('test_accuracy',0)*100:.1f}%")
c3.metric("📊 F1-Score", f"{info.get('test_f1',0):.4f}")
c4.metric("📈 ROC-AUC",  f"{info.get('test_auc',0):.4f}")
st.markdown("<hr style='margin:8px 0 20px'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    if st.button("☀️ Mode Terang" if D else "🌙 Mode Gelap", key="theme_btn"):
        st.session_state.dark_mode = not D
        st.rerun()

    st.markdown(f"""
    <div style="background:{P['bg3']};border:1px solid {P['bdr']};border-radius:10px;
         padding:14px 16px;margin:10px 0">
      <p style="color:{P['gold']};font-weight:700;font-size:.82em;letter-spacing:2px;margin:0 0 8px">
        📖 PANDUAN</p>
      <p style="font-size:.86em;color:{P['tx2']};margin:0;line-height:1.6">
        Isi semua kolom, klik tombol<br>
        <b style="color:{P['gold']}">🔮 Prediksi Kelayakan</b>
      </p>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"<p style='color:{P['gold']};font-weight:700;font-size:.82em;letter-spacing:2px;margin-top:16px'>🏆 FAKTOR TERPENTING</p>", unsafe_allow_html=True)
    for n, (feat, s) in enumerate([
        ("Riwayat Kredit",       5),
        ("Total Pendapatan",     4),
        ("Rasio Beban Pinjaman", 3),
        ("Cicilan Harian (EMI)", 3),
        ("Tenor Pinjaman",       1),
    ], 1):
        st.markdown(
            f'<div class="tip-row"><span>{n}. {feat}</span>'
            f'<span style="color:{P["gold"]}">{star(s)}</span></div>',
            unsafe_allow_html=True)

    st.markdown(f"<p style='color:{P['gold']};font-weight:700;font-size:.82em;letter-spacing:2px;margin-top:16px'>🎯 INTERPRETASI HASIL</p>", unsafe_allow_html=True)
    for col, ic, msg in [
        (P['green'],  "✅", "Prob ≥ 60% → DISETUJUI"),
        (P['orange'], "⚠️", "Prob 40–59% → BORDERLINE"),
        (P['red'],    "❌", "Prob < 40% → DITOLAK"),
    ]:
        st.markdown(
            f'<div style="background:{P["bg3"]};border-left:3px solid {col};'
            f'border-radius:0 6px 6px 0;padding:7px 12px;margin:5px 0;font-size:.86em">'
            f'{ic} {msg}</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:16px;padding:11px 13px;background:{P['bg3']};
         border-radius:8px;font-size:.8em;color:{P['tx2']};line-height:1.65">
      💡 <b style="color:{P['gold']}">Format angka:</b><br>
      • Ketik <b>5000000</b> atau <b>5.000.000</b><br>
      • Keduanya diterima otomatis
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  INPUT FORM
# ══════════════════════════════════════════════════════════════
st.markdown(f"<p style='color:{P['gold']};font-weight:700;font-size:.82em;letter-spacing:2px;margin-bottom:16px'>📋 DATA PEMOHON PINJAMAN</p>", unsafe_allow_html=True)

col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.markdown(f"<p style='color:{P['tx2']};font-size:.78em;letter-spacing:2px;font-weight:700;margin-bottom:10px'>👤 DATA PERSONAL</p>", unsafe_allow_html=True)
    jenis_kelamin     = st.selectbox("Jenis Kelamin",     ["Laki-laki","Perempuan"])
    status_pernikahan = st.selectbox("Status Pernikahan", ["Menikah","Belum Menikah"])
    jml_tanggungan    = st.selectbox("Jumlah Tanggungan", ["0","1","2","3+"],
                                     help="Anggota keluarga yang ditanggung")
    pendidikan        = st.selectbox("Pendidikan",        ["Sarjana","Bukan Sarjana"])
    wiraswasta        = st.selectbox("Status Pekerjaan",  ["Tidak (Karyawan/PNS)","Ya (Wiraswasta)"])
    wilayah           = st.selectbox("Lokasi Properti",   ["Semi perkotaan","Perkotaan","Pedesaan"],
                                     help="Wilayah lokasi properti yang diajukan")

with col_r:
    st.markdown(f"<p style='color:{P['tx2']};font-size:.78em;letter-spacing:2px;font-weight:700;margin-bottom:10px'>💰 DATA FINANSIAL</p>", unsafe_allow_html=True)

    pendapatan = idr_field(
        "Pendapatan Pemohon (Rp/bulan)",
        default=5_000_000,
        help_text="Gaji/penghasilan bulanan pemohon utama"
    )
    pendapatan_co = idr_field(
        "Pendapatan Co-Pemohon (Rp/bulan)",
        default=0,
        help_text="Isi 0 jika tidak ada co-pemohon"
    )
    jml_pinjaman = idr_field(
        "Jumlah Pinjaman (Rp)",
        default=24_000_000,
        help_text="Total pinjaman yang diajukan"
    )
    tenor = st.selectbox(
        "Tenor Pinjaman",
        [360,180,480,120,84,60,36,12],
        format_func=lambda x: f"{x} hari  ({x//30} bulan)"
    )
    riwayat_kredit = st.radio(
        "Riwayat Kredit",
        ["✅  Baik — Tidak ada tunggakan","❌  Buruk — Memiliki tunggakan"],
        index=0,
        help="Riwayat pembayaran kredit sebelumnya"
    )

# ── Input validation warnings ─────────────────────────────────
ti_raw = pendapatan + pendapatan_co
lr_raw = jml_pinjaman / (ti_raw + 1)
emi_raw = jml_pinjaman / (tenor + 1e-9)

warnings_input = []
if pendapatan < 500_000:
    warnings_input.append(f"⚠️ Pendapatan Rp {fmt(pendapatan)}/bln sangat rendah (di bawah distribusi data training). Prediksi mungkin kurang akurat.")
if lr_raw > 5:
    warnings_input.append(f"⚠️ Rasio Beban Pinjaman = {lr_raw:.1f} — sangat tidak wajar. Pinjaman jauh melebihi kemampuan bayar.")
if jml_pinjaman > LOAN_MAX_TRAIN:
    warnings_input.append(f"⚠️ Jumlah pinjaman Rp {fmt(jml_pinjaman)} melebihi range data training (max ≈ Rp {fmt(LOAN_MAX_TRAIN)}).")
if emi_raw > pendapatan * 0.8 and pendapatan > 0:
    warnings_input.append(f"⚠️ Cicilan harian Rp {fmt(emi_raw)} ≈ {emi_raw*30/pendapatan*100:.0f}% dari pendapatan bulanan — sangat berat.")

if warnings_input:
    st.markdown(f"""
    <div style="background:{P['w_bg']};border:1px solid {P['orange']};border-radius:10px;
         padding:12px 16px;margin:8px 0">
      <p style="color:{P['orange']};font-weight:700;font-size:.85em;letter-spacing:1px;margin:0 0 6px">
        ⚠️ PERHATIAN — INPUT DI LUAR RENTANG NORMAL</p>
      {chr(10).join(f'<p style="color:' + P['orange'] + f';font-size:.85em;margin:3px 0">{w}</p>' for w in warnings_input)}
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PREDICT BUTTON
# ══════════════════════════════════════════════════════════════
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
predict_btn = st.button("🔮  Prediksi Kelayakan Pinjaman", type="primary", use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════
if predict_btn:
    g   = 0 if jenis_kelamin == "Laki-laki" else 1
    m   = 1 if status_pernikahan == "Menikah" else 0
    d   = {"0":0,"1":1,"2":2,"3+":3}[jml_tanggungan]
    e   = 1 if pendidikan == "Sarjana" else 0
    s   = 1 if "Wiraswasta" in wiraswasta else 0
    ch  = 1 if "Baik" in riwayat_kredit else 0

    # ── Area One-Hot (BENAR: urutan ALFABETIS dari get_dummies) ──
    # get_dummies → Area_Pedesaan, Area_Perkotaan, Area_Semi_perkotaan
    ar  = 1 if wilayah == "Pedesaan"       else 0   # Area_Pedesaan
    au  = 1 if wilayah == "Perkotaan"      else 0   # Area_Perkotaan
    asu = 1 if wilayah == "Semi perkotaan" else 0   # Area_Semi_perkotaan

    al  = np.log1p(pendapatan)
    cl2 = np.log1p(pendapatan_co)
    ll  = np.log1p(jml_pinjaman)
    ti  = pendapatan + pendapatan_co
    tl  = np.log1p(ti)
    em  = jml_pinjaman / (tenor + 1e-9)
    eml = np.log1p(em)
    lr  = jml_pinjaman / (ti + 1)

    # FEATURES order matches training notebook exactly:
    # [Jenis_Kelamin, Status_Pernikahan, Jumlah_Tanggungan, Pendidikan, Wiraswasta,
    #  Riwayat_Kredit, Tenor_Pinjaman, Pendapatan_Pemohon_Log, Pendapatan_Pendamping_Log,
    #  Jumlah_Pinjaman_Log, TotalIncome_Log, EMI_Log, LoanIncomeRatio,
    #  Area_Pedesaan, Area_Perkotaan, Area_Semi_perkotaan]
    inp    = np.array([[g, m, d, e, s, ch, tenor,
                        al, cl2, ll, tl, eml, lr,
                        ar, au, asu]])   # ← BENAR: ar, au, asu (alfabetis)
    inp_sc = scaler.transform(inp)
    pred   = model.predict(inp_sc)[0]
    prob   = model.predict_proba(inp_sc)[0]
    p_yes  = prob[1]

    # ── 3-Tier result classification ─────────────────────────
    if p_yes >= 0.60:
        tier, tier_icon = "DISETUJUI",              "✅"
        tier_bg, tier_bdr, tier_tx = P['s_bg'], P['green'],  P['green']
        tier_msg = ("Profil sangat kuat — kelayakan sangat tinggi." if p_yes>=0.80
                    else "Pemohon memenuhi kriteria kelayakan pinjaman.")
    elif p_yes >= 0.40:
        tier, tier_icon = "BORDERLINE — Perlu Pertimbangan", "⚠️"
        tier_bg, tier_bdr, tier_tx = P['w_bg'], P['orange'], P['orange']
        tier_msg = "Persetujuan tidak pasti. Perkuat profil keuangan sebelum mengajukan."
    else:
        tier, tier_icon = "DITOLAK",                "❌"
        tier_bg, tier_bdr, tier_tx = P['e_bg'], P['red'],    P['red']
        tier_msg = "Profil belum memenuhi syarat. Lihat rekomendasi di bawah."

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{P['gold']};font-weight:700;font-size:.82em;letter-spacing:2px;margin-bottom:10px'>🎯 HASIL PREDIKSI</p>", unsafe_allow_html=True)

    # ── Banner ────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{tier_bg};border:2px solid {tier_bdr};border-radius:14px;
         padding:20px 28px;margin-bottom:14px">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
        <span style="font-size:1.8em">{tier_icon}</span>
        <div>
          <p style="color:{tier_tx};font-size:1.4em;font-weight:900;margin:0">{tier}</p>
          <p style="color:{tier_tx};font-size:.9em;margin:0;opacity:.85">{tier_msg}</p>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Probability gauge ─────────────────────────────────────
    # Colour the progress bar via custom overlay
    bar_color = P['green'] if p_yes>=0.60 else (P['orange'] if p_yes>=0.40 else P['red'])
    st.markdown(f"""
    <div style="margin-bottom:4px;display:flex;justify-content:space-between;font-size:.83em">
      <span style="color:{P['red']}">0% Ditolak</span>
      <span style="color:{P['orange']}">40% ─ Borderline ─ 60%</span>
      <span style="color:{P['green']}">100% Disetujui</span>
    </div>
    <div style="background:{P['bg3']};border-radius:8px;height:18px;position:relative;overflow:hidden;margin-bottom:16px;border:1px solid {P['bdr']}">
      <div style="width:{p_yes*100:.1f}%;background:{bar_color};height:100%;border-radius:8px;
           transition:width .4s"></div>
      <div style="position:absolute;top:0;left:39.5%;height:100%;width:1px;background:{P['orange']};opacity:.7"></div>
      <div style="position:absolute;top:0;left:59.5%;height:100%;width:1px;background:{P['green']};opacity:.7"></div>
    </div>""", unsafe_allow_html=True)

    pa, pb = st.columns(2)
    pa.metric("Probabilitas Disetujui", f"{p_yes*100:.1f}%",
              delta=f"{(p_yes-0.5)*100:+.1f}% dari batas 50%")
    pb.metric("Probabilitas Ditolak",   f"{prob[0]*100:.1f}%")

    # ── Factor table ──────────────────────────────────────────
    st.markdown(f"<p style='color:{P['gold']};font-weight:700;font-size:.82em;letter-spacing:2px;margin-top:18px;margin-bottom:8px'>📊 ANALISIS FAKTOR</p>", unsafe_allow_html=True)

    lr_label = "aman ✅" if lr<0.5 else ("perhatian ⚠️" if lr<1 else "berbahaya ❌")
    facts = pd.DataFrame({
        "Faktor"  : ["Riwayat Kredit","Total Pendapatan","Jumlah Pinjaman",
                     "Cicilan Harian (EMI)","Rasio Beban Pinjaman","Tenor"],
        "Nilai"   : [
            "Baik ✅" if ch else "Buruk ❌",
            f"Rp {fmt(ti)}/bln",
            f"Rp {fmt(jml_pinjaman)}",
            f"Rp {fmt(em)}/hari",
            f"{lr:.3f}  ({lr_label})",
            f"{tenor} hari ({tenor//30} bln)"
        ],
        "Status"  : [
            "✅ Mendukung"     if ch              else "❌ Risiko Tinggi",
            "✅ Mencukupi"     if ti>5_000_000    else "⚠️ Perlu Ditingkatkan",
            "✅ Wajar"         if jml_pinjaman<50_000_000 else "⚠️ Besar",
            "✅ Ringan"        if em<100_000      else "⚠️ Berat",
            ("✅ Aman"         if lr<0.5  else
             "⚠️ Perlu Perhatian" if lr<1 else "❌ Terlalu Tinggi"),
            "✅ Standar"       if tenor==360       else "ℹ️ Non-Standar"
        ],
        "Bobot"   : [star(5),star(4),star(3),star(3),star(2),star(1)]
    })
    st.dataframe(facts, use_container_width=True, hide_index=True)

    # Rasio penjelasan
    st.markdown(f"""
    <div style="background:{P['bg3']};border:1px solid {P['bdr']};border-radius:8px;
         padding:9px 14px;font-size:.82em;color:{P['tx2']};margin-top:2px">
      ℹ️ <b style="color:{P['gold']}">Rasio Beban Pinjaman</b> =
      Jumlah Pinjaman ÷ Total Pendapatan (bulan) =
      <b style="color:{P['gold']}">Rp {fmt(jml_pinjaman)} ÷ Rp {fmt(ti)} = {lr:.3f}</b><br>
      Idealnya &lt; 0.5 · Nilai {lr:.3f} {'aman ✅' if lr<0.5 else ('perlu perhatian ⚠️' if lr<1 else 'terlalu tinggi ❌')}
    </div>""", unsafe_allow_html=True)

    # ── RECOMMENDATIONS ───────────────────────────────────────
    st.markdown(f"<p style='color:{P['gold']};font-weight:700;font-size:.82em;letter-spacing:2px;margin-top:20px;margin-bottom:8px'>💡 REKOMENDASI</p>", unsafe_allow_html=True)

    recs = []

    # 1. Riwayat Kredit
    if ch == 0:
        recs.append((P['red'], "❌ Perbaiki Riwayat Kredit (Prioritas Utama)",
            f"Riwayat kredit buruk adalah penolak terbesar. Langkah: "
            f"hubungi semua kreditur untuk negosiasi ulang / pelunasan. "
            f"Setelah 6–12 bulan bersih, ajukan ulang."))

    # 2. Income vs loan ratio
    safe_loan = int(ti * 0.4)
    safe_loan_msg = f"Rp {fmt(safe_loan)}" if safe_loan > 0 else "lebih kecil"
    if lr > 0.5:
        recs.append((P['orange'] if lr<1 else P['red'],
            f"⚠️ Kurangi Jumlah Pinjaman (saat ini Rp {fmt(jml_pinjaman)})",
            f"Rasio {lr:.2f} terlalu tinggi. Dengan pendapatan Rp {fmt(ti)}/bln, "
            f"pinjaman ideal ≤ {safe_loan_msg} (rasio ≤ 0.4). "
            f"Atau tambah co-pemohon untuk meningkatkan total pendapatan."))

    # 3. EMI vs income
    if em*30 > ti*0.5 and ti > 0:
        emi_pct = em*30/ti*100
        recs.append((P['orange'],
            f"⚠️ Cicilan Terlalu Berat ({emi_pct:.0f}% dari pendapatan)",
            f"Cicilan bulanan Rp {fmt(em*30)} = {emi_pct:.0f}% pendapatan. "
            f"Pilih tenor 360 hari (30 thn) untuk cicilan ≈ Rp {fmt(jml_pinjaman/360)}/hari "
            f"(Rp {fmt(jml_pinjaman/360*30)}/bln = {jml_pinjaman/360*30/ti*100:.0f}% pendapatan)."))

    # 4. Low income
    if ti < INCOME_MED_TRAIN * 0.3:
        recs.append((P['orange'],
            f"⚠️ Pendapatan Terlalu Rendah (Rp {fmt(ti)}/bln)",
            f"Pendapatan jauh di bawah profil peminjam tipikal "
            f"(median ≈ Rp {fmt(INCOME_MED_TRAIN)}/bln). "
            f"Tambahkan co-pemohon (pasangan/orang tua) atau tunggu kenaikan gaji."))

    # 5. If approved but borderline
    if pred == 1 and 0.40 <= p_yes < 0.65:
        recs.append((P['blue'],
            "ℹ️ Approval Borderline — Perkuat Dokumen",
            f"Probabilitas {p_yes*100:.0f}% tergolong tipis. Siapkan: slip gaji 3 bulan, "
            f"rekening koran 6 bulan, dan surat keterangan kerja untuk memperkuat pengajuan."))

    # 6. Everything OK
    if not recs and pred == 1 and p_yes >= 0.65:
        recs.append((P['green'],
            "✅ Profil Kuat — Tips Mempertahankan Kelayakan",
            f"Tetap jaga riwayat kredit bersih, hindari utang baru sebelum pinjaman cair, "
            f"dan siapkan DP minimal 20% jika ini KPR."))

    for color, title, desc in recs:
        st.markdown(f"""
        <div style="background:{P['bg3']};border-left:4px solid {color};border-radius:0 10px 10px 0;
             padding:12px 16px;margin:7px 0">
          <p style="color:{color};font-weight:700;font-size:.9em;margin:0 0 5px">{title}</p>
          <p style="color:{P['tx2']};font-size:.86em;margin:0;line-height:1.6">{desc}</p>
        </div>""", unsafe_allow_html=True)

    # ── Expander: Non-monotonic explanation ───────────────────
    with st.expander("🔬 Mengapa probabilitas bisa turun meski pendapatan naik?"):
        st.markdown(f"""
        <div style="color:{P['tx']};font-size:.88em;line-height:1.7">

        <p><b style="color:{P['gold']}">Ini adalah perilaku normal Random Forest</b> dan bukan error.</p>

        <b>Penjelasan teknis:</b>
        <ul style="color:{P['tx2']}">
          <li>Random Forest terdiri dari ratusan <i>decision tree</i> yang masing-masing
              mempartisi ruang fitur secara berbeda.</li>
          <li>Model ini <b>tidak dijamin monoton</b> — artinya pendapatan naik sedikit
              TIDAK selalu berarti probabilitas naik.</li>
          <li>Perubahan kecil (misal Rp 650.000 → 660.000) bisa menggeser titik data
              ke leaf node berbeda di beberapa tree, mengubah vote mayoritas.</li>
        </ul>

        <b>Kenapa ini terjadi lebih sering di nilai ekstrem?</b>
        <ul style="color:{P['tx2']}">
          <li>Pendapatan Rp 650K–660K dengan pinjaman Rp 24 juta menghasilkan
              <b>Rasio Beban Pinjaman ≈ 37</b> (jauh di luar batas normal 0.5).</li>
          <li>Data training hampir tidak punya sampel di range ini →
              prediksi di zona ekstrem tidak stabil.</li>
          <li>Perbedaan 3% (62% vs 59%) di zona ekstrem = <b>noise statistik</b>,
              bukan sinyal yang bermakna.</li>
        </ul>

        <b>Kapan prediksi bisa dipercaya?</b>
        <ul style="color:{P['tx2']}">
          <li>Probabilitas <b>&lt; 30% atau &gt; 70%</b> → sinyal kuat, bisa dipercaya.</li>
          <li>Probabilitas <b>30–70%</b> → borderline, fokus pada perbaikan faktor utama.</li>
          <li>Selalu perhatikan <b>Riwayat Kredit</b> dan <b>Rasio Beban Pinjaman</b>
              sebagai indikator utama, bukan angka probabilitas exact.</li>
        </ul>
        </div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────
st.markdown("<hr style='margin:24px 0 12px'>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align:center;color:{P['tx2']};font-size:.78em;line-height:2">
  🤖 Random Forest Classifier &nbsp;·&nbsp;
  📊 Dataset Loan Prediction — Kaggle &nbsp;·&nbsp;
  🔑 Kurs: 1 INR = Rp 187,22 (Juni 2026) &nbsp;·&nbsp;
  🚀 Streamlit Cloud<br>
  ⚠️ Hasil bersifat indikatif — bukan keputusan resmi lembaga keuangan
</div>""", unsafe_allow_html=True)
