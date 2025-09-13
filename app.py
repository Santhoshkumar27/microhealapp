#!/usr/bin/env python3
# app.py — Micro Science S3 Session Viewer (mobile-friendly, charts, tabs)
#
# Features
# - Reads latest (or selectable) session from S3:
#     microheal/<deviceId>/<YYYY-MM-DD>/<YYYYMMDD_HHMMSS>/
# - Shows Analytics + Presence
# - Photos grid (JPG/PNG)
# - Audio player (WAV)
# - Mobile-friendly layout, tabs, metrics, Bristol scale classification,
#   Bristol distribution bar chart, and multi-session trend (avg Bristol & duration).
# - Big CTA to jump to the latest session automatically.
#
# How to run locally:
#   python3 -m venv .venv
#   source .venv/bin/activate
#   pip install --upgrade pip wheel
#   pip install streamlit boto3 pillow python-dotenv plotly
#   streamlit run app.py
#
# Credentials:
#   Use env vars or a named AWS profile. Do NOT hardcode credentials.
#   export AWS_ACCESS_KEY_ID=...
#   export AWS_SECRET_ACCESS_KEY=...
#   export AWS_REGION=ap-south-1
#   # OR: aws configure --profile s3-uploader
#
# Notes:
# - We ignore any non-date folder like "queued/".
# - Works without pandas; uses Plotly for charts.

import io
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import boto3
from botocore.exceptions import NoCredentialsError, ClientError, BotoCoreError
from PIL import Image
import streamlit as st
import os
import plotly.express as px
import plotly.graph_objects as go

# Optional .env (local only)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --------------------------- S3 helpers ---------------------------

def s3_client(region: Optional[str], profile: Optional[str]):
    """Create an S3 client using env vars or a named profile.
    Returns (session, s3_client, auth_source_str)
    """
    try:
        ak  = os.getenv("AWS_ACCESS_KEY_ID")
        sk  = os.getenv("AWS_SECRET_ACCESS_KEY")
        tok = os.getenv("AWS_SESSION_TOKEN")  # optional
        reg = region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "ap-south-1"

        # 1) Explicit env vars
        if ak and sk:
            session = boto3.Session(
                aws_access_key_id=ak.strip(),
                aws_secret_access_key=sk.strip(),
                aws_session_token=(tok.strip() if tok else None),
                region_name=reg
            )
            return session, session.client("s3"), "env"

        # 2) Named profile
        if profile and profile.strip():
            session = boto3.Session(profile_name=profile.strip(), region_name=reg)
            return session, session.client("s3"), f"profile:{profile.strip()}"

        # 3) Default chain
        session = boto3.Session(region_name=reg)
        return session, session.client("s3"), "default-chain"
    except (BotoCoreError, Exception) as e:
        st.error(f"Failed to create S3 client: {e}")
        st.stop()


def list_subprefixes(client, bucket: str, prefix: str) -> List[str]:
    """Return 'folders' directly under prefix (uses Delimiter='/')."""
    resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
    return [cp["Prefix"] for cp in resp.get("CommonPrefixes", [])]


def is_valid_date_folder(folder: str) -> bool:
    """Check suffix is YYYY-MM-DD"""
    suffix = folder.strip('/').split('/')[-1]
    return (len(suffix) == 10 and suffix[4] == '-' and suffix[7] == '-' and suffix.replace('-', '').isdigit())


def is_valid_session_folder(folder: str) -> bool:
    """Check suffix is YYYYMMDD_HHMMSS"""
    suffix = folder.strip('/').split('/')[-1]
    return (len(suffix) == 15 and suffix[8] == '_' and suffix[:8].isdigit() and suffix[9:].isdigit())


def latest_session_prefix(client, bucket: str, device_id: str) -> Optional[str]:
    """Newest real session under <device>/<YYYY-MM-DD>/<YYYYMMDD_HHMMSS>/ (ignores queued/)."""
    root = f"{device_id.strip()}/"
    dates = [d for d in list_subprefixes(client, bucket, root) if is_valid_date_folder(d)]
    if not dates:
        return None
    dates.sort()
    last_date = dates[-1]
    sessions = [s for s in list_subprefixes(client, bucket, last_date) if is_valid_session_folder(s)]
    if not sessions:
        return None
    sessions.sort()
    return sessions[-1]


def recent_sessions(client, bucket: str, device_id: str, limit: int = 7) -> List[str]:
    """Collect up to N most recent session prefixes across recent date folders."""
    root = f"{device_id.strip()}/"
    dates = [d for d in list_subprefixes(client, bucket, root) if is_valid_date_folder(d)]
    dates.sort(reverse=True)
    out: List[str] = []
    for d in dates:
        sessions = [s for s in list_subprefixes(client, bucket, d) if is_valid_session_folder(s)]
        sessions.sort(reverse=True)
        for s in sessions:
            out.append(s)
            if len(out) >= limit:
                return sorted(out)  # sort ascending for nicer labels
    return sorted(out)


def list_objects_in_prefix(client, bucket: str, prefix: str) -> List[str]:
    keys: List[str] = []
    token = None
    while True:
        kwargs = dict(Bucket=bucket, Prefix=prefix)
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            if not obj["Key"].endswith("/"):
                keys.append(obj["Key"])
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break
    return keys


def download_object_bytes(client, bucket: str, key: str) -> bytes:
    bio = io.BytesIO()
    client.download_fileobj(bucket, key, bio)
    return bio.getvalue()

# --------------------------- Analytics helpers ---------------------------

def parse_bristol_info(aj: Dict[str, Any]) -> Tuple[Optional[float], Dict[int, int]]:
    """
    Extract average Bristol score and histogram from analytics JSON (best-effort).
    Expected keys (any of):
      - avgBristol / bristolAverage / bristol_mean : float
      - bristolHistogram : { "1": n, ..., "7": n }  or list length 7
      - bristolScores : [ints]
    """
    avg = None
    hist: Dict[int, int] = {i: 0 for i in range(1, 8)}

    # average
    for k in ("avgBristol", "bristolAverage", "bristol_mean", "average_bristol"):
        if k in aj and isinstance(aj[k], (int, float)):
            avg = float(aj[k])
            break

    # histogram
    if "bristolHistogram" in aj:
        bh = aj["bristolHistogram"]
        if isinstance(bh, dict):
            for i in range(1, 8):
                val = bh.get(str(i)) or bh.get(i) or 0
                if isinstance(val, int):
                    hist[i] = val
        elif isinstance(bh, list) and len(bh) >= 7:
            for i in range(7):
                try:
                    hist[i+1] = int(bh[i])
                except Exception:
                    pass

    elif "bristolScores" in aj and isinstance(aj["bristolScores"], list):
        for s in aj["bristolScores"]:
            try:
                si = int(s)
                if 1 <= si <= 7:
                    hist[si] += 1
            except Exception:
                pass

    # derive avg if missing and we have histogram
    total = sum(hist.values())
    if avg is None and total > 0:
        avg = sum(score * count for score, count in hist.items()) / float(total)

    return avg, hist


def classify_bristol(avg: Optional[float]) -> str:
    """Return 'Constipation', 'Normal', or 'Diarrhea' based on avg Bristol."""
    if avg is None:
        return "Unknown"
    if avg < 2.5:
        return "Constipation"
    if avg > 4.5:
        return "Diarrhea"
    return "Normal"


def secs_to_hms(sec: Optional[int]) -> str:
    if not isinstance(sec, (int, float)):
        return "—"
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"

#
# --------------------------- UI THEME (mobile-ish) ---------------------------
MOBILE_CSS = """
<style>
:root{
  --mh-bg: #0b0f14;
  --mh-card: #121822;
  --mh-accent: #3b82f6;
  --mh-ok:#10b981;
  --mh-warn:#f59e0b;
  --mh-bad:#ef4444;
  --mh-text:#e5e7eb;
  --mh-dim:#9ca3af;
  --mh-chart1:#22d3ee;
  --mh-chart2:#60a5fa;
  --mh-chart3:#a78bfa;
}
@media (prefers-color-scheme: light){
  :root{
    --mh-bg: #f7f9fc;
    --mh-card: #ffffff;
    --mh-accent: #2563eb;
    --mh-ok:#059669;
    --mh-warn:#b45309;
    --mh-bad:#dc2626;
    --mh-text:#0f172a;
    --mh-dim:#475569;
    --mh-chart1:#0891b2;
    --mh-chart2:#3b82f6;
    --mh-chart3:#7c3aed;
  }
}
html, body, .stApp { background: var(--mh-bg) !important; }
.block-container{ padding-top: 0.75rem; padding-bottom: 4.5rem; max-width: 980px; }
.mh-appbar{
  position: sticky; top: 0; z-index: 100;
  backdrop-filter: blur(6px);
  background: linear-gradient(180deg, rgba(0,0,0,.45), rgba(0,0,0,.15)) !important;
  padding: .6rem .8rem; border-bottom: 1px solid rgba(255,255,255,.07);
}
.mh-title{ display:flex; align-items:center; gap:.5rem; color: var(--mh-text); font-weight:700; }
.mh-badge{ font-size:.8rem; padding:.15rem .5rem; border-radius: 999px; background: var(--mh-accent); color:white; }
.mh-card{
  background: var(--mh-card); border:1px solid rgba(255,255,255,.06);
  border-radius: 16px; padding: 14px 14px;
  box-shadow: 0 6px 18px rgba(0,0,0,.15);
  margin-bottom: 12px;
}
.mh-kv{ display:flex; justify-content:space-between; color:var(--mh-dim); font-size:.9rem; padding:.2rem 0; }
.mh-kv b{ color:var(--mh-text); }
.mh-pill{ display:inline-flex; align-items:center; gap:.35rem; border-radius:999px; padding:.2rem .55rem; font-size:.85rem; }
.mh-pill.ok{ background: rgba(16,185,129,.15); color: var(--mh-ok); }
.mh-pill.warn{ background: rgba(245,158,11,.15); color: var(--mh-warn); }
.mh-pill.bad{ background: rgba(239,68,68,.15); color: var(--mh-bad); }
.mh-section-title{ color: var(--mh-text); font-weight:700; margin:.2rem 0 .5rem; }
.mh-sub{ color: var(--mh-dim); font-size:.85rem; margin-bottom:.5rem; }
.mh-grid{ display:grid; grid-template-columns: repeat( auto-fill, minmax(180px,1fr) ); gap:10px; }
.mh-img{ border-radius:12px; overflow:hidden; border:1px solid rgba(255,255,255,.06); }
/* NEW UI elements */
.mh-icons{ display:flex; gap:.5rem; flex-wrap:wrap; margin:.25rem 0 .75rem; }
.mh-ic{ background: var(--mh-card); border:1px solid rgba(255,255,255,.06); border-radius:12px; padding:.4rem .6rem; display:flex; align-items:center; gap:.45rem; color:var(--mh-text); }
.mh-ic small{ color:var(--mh-dim); }
.mh-selbar{ position: sticky; top: 48px; z-index: 90; padding:.4rem; margin-bottom:.4rem; background: linear-gradient(180deg, rgba(0,0,0,.25), rgba(0,0,0,.05)); border-bottom: 1px solid rgba(255,255,255,.07); }
.mh-bottomnav{
  position: fixed; left:0; right:0; bottom:0; z-index:120;
  background: var(--mh-card); border-top:1px solid rgba(255,255,255,.08);
  display:flex; justify-content:space-around; padding:.35rem .6rem;
}
.mh-bottomnav a{ color: var(--mh-text); text-decoration:none; font-size:.95rem; opacity:.9; }
.mh-bottomnav a .tiny{ display:block; font-size:.7rem; color:var(--mh-dim); }
</style>
"""
def mh_header():
    st.markdown(MOBILE_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="mh-appbar">
          <div class="mh-title">
            <span>🧪 Micro Science</span>
            <span class="mh-badge">Wellness</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
def mh_icon_row(wifi:str, battery:str, presence:str):
    html = f'''
    <div class="mh-icons">
      <div class="mh-ic">📶 <b>{wifi}</b> <small>Wi‑Fi</small></div>
      <div class="mh-ic">🔋 <b>{battery}</b> <small>Battery</small></div>
      <div class="mh-ic">🧍 <b>{presence}</b> <small>Presence</small></div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)

def mh_session_selector(label:str, labels, index:int):
    st.markdown('<div class="mh-selbar">', unsafe_allow_html=True)
    sel = st.selectbox(label, options=labels, index=index, key="session_pick")
    st.markdown('</div>', unsafe_allow_html=True)
    return labels.index(sel)
def pill(text:str, kind:str="ok"):
    return f'<span class="mh-pill {kind}">{text}</span>'
def card_kv(items):
    rows = "".join([f'<div class="mh-kv"><span>{k}</span><b>{v}</b></div>' for k,v in items])
    st.markdown(f'<div class="mh-card">{rows}</div>', unsafe_allow_html=True)

# --------------------------- UI ---------------------------

st.set_page_config(page_title="Micro Science Session Viewer", layout="wide")
mh_header()
st.markdown('<div class="mh-section-title">Latest Analysis</div>', unsafe_allow_html=True)
cta_btn = st.button("Know your gut — Latest analysis", use_container_width=True)

with st.sidebar:
    st.header("S3 Settings")
    bucket = st.text_input("Bucket name", value="microheal")
    region = st.text_input("AWS region (optional)", value="ap-south-1")
    profile = st.text_input("AWS profile (leave blank to use env vars)", value="")
    device_id = st.text_input("Device ID (root folder)", value="1f0298d885aa")
    last_n = st.number_input("Show last N sessions", min_value=1, max_value=30, value=7, step=1)
    st.caption("Use env vars or choose a named profile (e.g., s3-uploader).")

    fetch_btn = st.button("Fetch")
    diag_btn = st.button("Check AWS identity")

# Diagnostics button
if 'diag_btn' in locals() and diag_btn:
    try:
        sess, _, auth_src = s3_client(region, profile)
        sts = sess.client("sts")
        ident = sts.get_caller_identity()
        acct = ident.get("Account", "—")
        arn = ident.get("Arn", "—")
        user = arn.split('/')[-1] if '/' in arn else arn
        st.success("STS get-caller-identity succeeded")
        st.write({"Auth source": auth_src, "Account": acct, "Arn": arn, "User": user})
    except Exception as e:
        st.error(f"STS identity check failed: {e}")
        st.info("If this fails, your credentials or region/profile are not being picked up.")

# Main fetch flow

do_fetch = bool(fetch_btn or ('cta_btn' in locals() and cta_btn))
force_latest = bool('cta_btn' in locals() and cta_btn)

if do_fetch:
    if not bucket or not device_id:
        st.error("Please enter Bucket name and Device ID.")
        st.stop()

    sess, s3, auth_src = s3_client(region, profile)
    st.info(f"Auth source: {auth_src}")

    # Build recent sessions and let the user pick
    try:
        recents = recent_sessions(s3, bucket, device_id, limit=int(last_n))
    except ClientError as e:
        st.error(f"S3 error while listing sessions: {e}")
        st.stop()

    if not recents:
        st.error(f"No sessions found for device `{device_id}`.")
        st.stop()

    # Build short labels
    def short_label(pref: str) -> str:
        parts = pref.strip('/').split('/')
        # Expect: device / YYYY-MM-DD / YYYYMMDD_HHMMSS
        return f"{parts[-2]} · {parts[-1]}"

    if force_latest:
        prefix = recents[-1]
        st.success(f"Auto-selected latest session: `{prefix}`")
    else:
        labels = [short_label(p) for p in recents]
        default_idx = len(labels) - 1
        choice = mh_session_selector("Session", labels, default_idx)
        prefix = recents[choice]
        st.success(f"Selected session: `{prefix}`")

    # Tabs for a clean mobile-friendly UI
    tab_overview, tab_photos, tab_audio, tab_details = st.tabs(["Overview", "Photos", "Audio", "Details"])

    # List objects once
    keys = list_objects_in_prefix(s3, bucket, prefix)
    imgs = [k for k in keys if k.lower().endswith((".jpg", ".jpeg", ".png"))]
    wavs = [k for k in keys if k.lower().endswith(".wav")]
    analytics_key = next((k for k in keys if k.lower().endswith("analytics.json")), None)
    sess_presence_key = next((k for k in keys if k.lower().endswith("session_presence.json")), None)
    presence_jsonl_key = next((k for k in keys if k.lower().endswith("presence.jsonl")), None)

    # ---------- Overview ----------
    with tab_overview:
        st.markdown('<a id="overview"></a>', unsafe_allow_html=True)

        # Quick status icon row (prefill; updated after JSON loads)
        _wifi = "—"
        _batt = "—"
        _pres = "—"

        colM1, colM2 = st.columns(2)

        # Left: Metrics & Classification
        with colM1:
            st.markdown('<div class="mh-section-title">Session Metrics</div>', unsafe_allow_html=True)

            aj = {}
            if analytics_key:
                try:
                    data = download_object_bytes(s3, bucket, analytics_key)
                    aj = json.loads(data.decode("utf-8"))
                except Exception as e:
                    st.warning(f"Failed to read analytics.json: {e}")

            avg_bristol, br_hist = parse_bristol_info(aj)
            classification = classify_bristol(avg_bristol)

            duration = aj.get("sessionDurationSeconds") or aj.get("durationSeconds") or aj.get("session_duration") or 0
            daily_triggers = aj.get("dailyTriggerCount") or aj.get("daily_triggers") or 0
            avg_visit_seconds = aj.get("avgVisitSeconds") or aj.get("averageVisitSeconds") or None

            # Derive icon values (first pass)
            _wifi = aj.get("wifiSsid") or aj.get("networkSsid") or "—"
            _batt_val = aj.get("batteryPct") or aj.get("battery")
            if isinstance(_batt_val, (int, float)):
                _batt = f"{int(_batt_val)}%"
            _pres = "Present" if aj.get("present") is True else ("Absent" if aj.get("present") is False else "—")

            mh_icon_row(_wifi, _batt, _pres)

            status_kind = "ok" if classification=="Normal" else ("warn" if classification=="Constipation" else ("bad" if classification=="Diarrhea" else "warn"))
            card_kv([
                ("Last Visit Duration", secs_to_hms(duration)),
                ("Average Bathroom Time", secs_to_hms(avg_visit_seconds) if avg_visit_seconds else "—"),
                ("Daily Visits", str(daily_triggers)),
                ("Gut Status", f"Classified as: {classification} {pill('Wellness', status_kind)}"),
            ])

            # Bristol distribution — Plotly
            try:
                xs = list(range(1, 8))
                ys = [br_hist[i] for i in xs] if br_hist else [0]*7
                fig = px.bar(
                    x=[str(x) for x in xs],
                    y=ys,
                    labels={'x': 'Bristol Type (1–7)', 'y': 'Count'},
                )
                fig.update_traces(marker_color=[
                    'var(--mh-chart1)','var(--mh-chart1)',
                    'var(--mh-chart2)','var(--mh-chart2)','var(--mh-chart2)',
                    'var(--mh-chart3)','var(--mh-chart3)'
                ])
                fig.update_layout(
                    title="Stool Type Distribution",
                    margin=dict(l=10,r=10,t=40,b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info(f"(Chart skipped: {e})")

        # Right: Presence quick view
        with colM2:
            st.markdown('<div class="mh-section-title">Presence</div>', unsafe_allow_html=True)
            sp = {}
            if sess_presence_key:
                try:
                    data = download_object_bytes(s3, bucket, sess_presence_key)
                    sp = json.loads(data.decode("utf-8"))
                except Exception as e:
                    st.warning(f"Failed to read session_presence.json: {e}")

            last_line = None
            if presence_jsonl_key:
                try:
                    data = download_object_bytes(s3, bucket, presence_jsonl_key).decode("utf-8", "replace")
                    lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
                    last_line = json.loads(lines[-1]) if lines else None
                except Exception as e:
                    st.warning(f"Failed to parse presence.jsonl: {e}")

            # Update presence indicator if JSONs provide a value
            try:
                if isinstance(sp, dict) and 'present' in sp:
                    _pres = "Present" if bool(sp.get('present')) else "Absent"
                if isinstance(last_line, dict) and 'present' in last_line:
                    _pres = "Present" if bool(last_line.get('present')) else "Absent"
            except Exception:
                pass
            mh_icon_row(_wifi, _batt, _pres)

            if sp:
                st.caption("session_presence.json")
                st.json(sp, expanded=False)
            if last_line:
                st.caption("presence.jsonl (last)")
                st.json(last_line, expanded=False)
            if not sp and not last_line:
                st.info("No presence artifacts found.")

        # Trend across recent sessions (avg bristol & duration)
        st.markdown('<div class="mh-section-title">Recent Trend</div>', unsafe_allow_html=True)
        # Load analytics for each recent session (best-effort)
        trend_labels, trend_bristol, trend_dur = [], [], []
        for p in recents:
            try:
                keys_r = list_objects_in_prefix(s3, bucket, p)
                ak = next((k for k in keys_r if k.lower().endswith("analytics.json")), None)
                if not ak:
                    continue
                data = download_object_bytes(s3, bucket, ak)
                ajr = json.loads(data.decode("utf-8"))
                avg_b, _ = parse_bristol_info(ajr)
                dur = ajr.get("sessionDurationSeconds") or ajr.get("durationSeconds") or 0
                trend_labels.append(short_label(p))
                trend_bristol.append(avg_b if avg_b is not None else 0.0)
                trend_dur.append(int(dur) if isinstance(dur, (int, float)) else 0)
            except Exception:
                continue

        if trend_labels:
            try:
                # Avg Bristol trend (Plotly)
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=trend_labels, y=trend_bristol, mode="lines+markers",
                    line=dict(width=3), fill='tozeroy'
                ))
                fig1.update_layout(
                    title="Gut Score per Session",
                    yaxis_title="Avg Bristol",
                    margin=dict(l=10,r=10,t=40,b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig1, use_container_width=True)

                # Duration trend (Plotly)
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=trend_labels, y=trend_dur, mode="lines+markers",
                    line=dict(width=3), fill='tozeroy'
                ))
                fig2.update_layout(
                    title="Time Spent per Session",
                    yaxis_title="Duration (s)",
                    margin=dict(l=10,r=10,t=40,b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig2, use_container_width=True)
            except Exception as e:
                st.info(f"(Trend charts skipped: {e})")
        else:
            st.info("Not enough analytics across recent sessions to plot trends.")

    # ---------- Photos ----------
    with tab_photos:
        st.markdown('<a id="photos"></a>', unsafe_allow_html=True)
        st.markdown('<div class="mh-section-title">Latest Photos</div>', unsafe_allow_html=True)
        if imgs:
            imgs_sorted = sorted(imgs)
            html_imgs = []
            for k in imgs_sorted:
                try:
                    jb = download_object_bytes(s3, bucket, k)
                    img = Image.open(io.BytesIO(jb)).convert("RGB")
                    # Write to in-memory PNG for <img> src as bytes
                    buf = io.BytesIO(); img.save(buf, format="PNG"); b64 = buf.getvalue()
                    import base64
                    b64s = base64.b64encode(b64).decode("ascii")
                    html_imgs.append(f'<div class="mh-card mh-img"><img src="data:image/png;base64,{b64s}" style="width:100%;height:auto;display:block;" /><div class="mh-sub">{Path(k).name}</div></div>')
                except Exception as e:
                    html_imgs.append(f'<div class="mh-card mh-img"><div class="mh-sub">Load failed: {Path(k).name}</div></div>')
            st.markdown('<div class="mh-grid">' + "".join(html_imgs) + '</div>', unsafe_allow_html=True)
        else:
            st.info("No images in this session.")

    # ---------- Audio ----------
    with tab_audio:
        st.markdown('<a id="audio"></a>', unsafe_allow_html=True)
        st.markdown('<div class="mh-section-title">Latest Audio</div>', unsafe_allow_html=True)
        if wavs:
            wavs_sorted = sorted(wavs)
            first = wavs_sorted[0]
            try:
                wb = download_object_bytes(s3, bucket, first)
                st.caption(Path(first).name)
                st.audio(wb, format="audio/wav")
            except Exception as e:
                st.warning(f"Failed to load audio: {e}")
        else:
            st.info("No WAV found in this session.")

    # ---------- Details (raw JSON dumps) ----------
    with tab_details:
        st.markdown('<div class="mh-section-title">Session Keys</div>', unsafe_allow_html=True)
        st.code("\n".join(sorted(keys)) or "(none)")

        if analytics_key:
            try:
                data = download_object_bytes(s3, bucket, analytics_key)
                st.markdown('<div class="mh-section-title">analytics.json</div>', unsafe_allow_html=True)
                st.json(json.loads(data.decode("utf-8")), expanded=False)
            except Exception:
                pass
        if sess_presence_key:
            try:
                data = download_object_bytes(s3, bucket, sess_presence_key)
                st.markdown('<div class="mh-section-title">session_presence.json</div>', unsafe_allow_html=True)
                st.json(json.loads(data.decode("utf-8")), expanded=False)
            except Exception:
                pass
        if presence_jsonl_key:
            try:
                data = download_object_bytes(s3, bucket, presence_jsonl_key).decode("utf-8", "replace")
                st.markdown('<div class="mh-section-title">presence.jsonl (tail)</div>', unsafe_allow_html=True)
                lines = [ln for ln in data.splitlines() if ln.strip()]
                st.code("\n".join(lines[-20:]))
            except Exception:
                pass

    # Footer branding + CTA
    st.markdown(
      '''
      <div style="text-align:center; opacity:.9; margin: 0.75rem 0;">
        <div style="font-size:1.1rem; font-weight:700; color:var(--mh-text);">🧪 Micro Science</div>
        <div style="color:var(--mh-dim); font-size:.85rem; margin:.2rem 0 .5rem;">Your wellness companion</div>
      </div>
      ''',
      unsafe_allow_html=True
    )
    st.button("Click here for latest analysis", use_container_width=True, key="footer_cta")

    # Bottom nav (cosmetic)
    st.markdown(
      '''
      <div class="mh-bottomnav">
        <a href="#overview">Overview<span class="tiny">Summary</span></a>
        <a href="#photos">Photos<span class="tiny">Gallery</span></a>
        <a href="#audio">Audio<span class="tiny">Player</span></a>
      </div>
      ''',
      unsafe_allow_html=True
    )