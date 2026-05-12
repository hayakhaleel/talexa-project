from __future__ import annotations

import cgi
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
import zipfile
import traceback
from datetime import datetime, timedelta, timezone
from html import escape
from http.cookies import SimpleCookie
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote

from wsgiref.simple_server import make_server


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
ASSETS_DIR = APP_DIR / "assets"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database import (
    authenticate_user,
    create_session_record,
    create_user,
    get_max_session_id,
    get_session_record,
    get_storage_paths,
    get_user_by_email,
    init_db,
    update_session_status,
    upsert_storage_paths,
)
from PIPELINE.preprocessing import (
    validate_and_prepare_audio,
    validate_portrait_image,
    validate_source_document,
)
from PIPELINE.session_manager import create_session, get_next_session_number


LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
RATE_LIMIT_WINDOW_SECONDS = 15 * 60
RATE_LIMIT_RULES = {
    "normal": {"capacity": 160, "refill_rate": 160 / RATE_LIMIT_WINDOW_SECONDS},
    "login": {"capacity": 10, "refill_rate": 10 / RATE_LIMIT_WINDOW_SECONDS},
    "generate": {"capacity": 4, "refill_rate": 4 / RATE_LIMIT_WINDOW_SECONDS},
}
RATE_LIMIT_BUCKETS: dict[tuple[str, str], dict[str, float]] = {}
SESSION_MAX_AGE_SECONDS = 3 * 60 * 60


def request_host(environ):
    return environ.get("HTTP_HOST", "").split(":", 1)[0]


def is_secure_request(environ):
    forwarded_proto = environ.get("HTTP_X_FORWARDED_PROTO", "").split(",", 1)[0].strip().lower()
    return environ.get("wsgi.url_scheme") == "https" or environ.get("HTTPS", "").lower() == "on" or forwarded_proto == "https"


def should_redirect_to_https(environ):
    return request_host(environ) not in LOCAL_HOSTS and not is_secure_request(environ)


def https_url(environ):
    host = environ.get("HTTP_HOST", "")
    path = environ.get("PATH_INFO", "/")
    query = environ.get("QUERY_STRING", "")
    return f"https://{host}{path}{'?' + query if query else ''}"


def security_headers(environ):
    headers = [
        ("Cache-Control", "no-store"),
        ("Pragma", "no-cache"),
        ("X-Content-Type-Options", "nosniff"),
        ("X-Frame-Options", "DENY"),
        ("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; form-action 'self'; base-uri 'self'"),

    ]
    if is_secure_request(environ):
        headers.append(("Strict-Transport-Security", "max-age=31536000; includeSubDomains"))
    return headers


def cookie_security_suffix(environ):
    return "; Secure" if is_secure_request(environ) else ""


def client_ip(environ):
    forwarded_for = environ.get("HTTP_X_FORWARDED_FOR", "").split(",", 1)[0].strip()
    return forwarded_for or environ.get("REMOTE_ADDR", "unknown")


def rate_limit_scope(environ):
    path = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "GET")
    if path == "/login" and method == "POST":
        return "login"
    if path == "/generate" and method == "POST":
        return "generate"
    return "normal"


def token_bucket_allowed(environ):
    scope = rate_limit_scope(environ)
    rule = RATE_LIMIT_RULES[scope]
    key = (client_ip(environ), scope)
    now = time.time()
    bucket = RATE_LIMIT_BUCKETS.get(key, {"tokens": rule["capacity"], "updated_at": now})
    elapsed = now - bucket["updated_at"]
    tokens = min(rule["capacity"], bucket["tokens"] + elapsed * rule["refill_rate"])
    allowed = tokens >= 1
    if allowed:
        tokens -= 1
    RATE_LIMIT_BUCKETS[key] = {"tokens": tokens, "updated_at": now}
    return allowed, scope


def parse_cookies(environ):
    raw_cookie = environ.get("HTTP_COOKIE", "")
    parsed = SimpleCookie(raw_cookie)
    return {key: unquote(morsel.value) for key, morsel in parsed.items()}


def parse_post_data(environ):
    length = int(environ.get("CONTENT_LENGTH") or 0)
    raw_data = environ["wsgi.input"].read(length).decode("utf-8")
    parsed = parse_qs(raw_data)
    return {key: values[0] for key, values in parsed.items()}


def parse_multipart(environ):
    return cgi.FieldStorage(fp=environ["wsgi.input"], environ=environ, keep_blank_values=True)


def clean_text(value, max_length=200):
    cleaned = "".join(character for character in str(value).strip() if character.isprintable())
    return cleaned.replace("<", "").replace(">", "")[:max_length]


def query_notice(environ):
    query = parse_qs(environ.get("QUERY_STRING", ""))
    return query.get("notice", [""])[0]


def query_int(environ, name):
    query = parse_qs(environ.get("QUERY_STRING", ""))
    value = query.get(name, [""])[0]
    return int(value) if value.isdigit() else None


def redirect(environ, start_response, location, extra_headers=None):
    headers = [("Location", location)]
    if extra_headers:
        headers.extend(extra_headers)
    headers.extend(security_headers(environ))
    start_response("302 Found", headers)
    return [b""]


def plain_text(environ, start_response, status, text):
    body = text.encode("utf-8")
    start_response(status, [("Content-Type", "text/plain; charset=utf-8"), ("Content-Length", str(len(body)))] + security_headers(environ))
    return [body]


def html_response(environ, start_response, html_text, status="200 OK", extra_headers=None):
    body = html_text.encode("utf-8")
    headers = [("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(body)))]
    if extra_headers:
        headers.extend(extra_headers)
    headers.extend(security_headers(environ))
    start_response(status, headers)
    return [body]


def binary_response(environ, start_response, file_name, content_type, data):
    headers = [
        ("Content-Type", content_type),
        ("Content-Length", str(len(data))),
        ("Content-Disposition", f'attachment; filename="{file_name}"'),
    ]
    headers.extend(security_headers(environ))
    start_response("200 OK", headers)
    return [data]


def serve_static(environ, start_response):
    path = environ.get("PATH_INFO", "/")
    if path.startswith("/assets/"):
        base_dir = ASSETS_DIR
        relative_path = path.replace("/assets/", "", 1)
    else:
        base_dir = STATIC_DIR
        relative_path = path.replace("/static/", "", 1)

    file_path = (base_dir / relative_path).resolve()
    if not str(file_path).startswith(str(base_dir.resolve())) or not file_path.is_file():
        return plain_text(environ, start_response, "404 Not Found", "Static file not found.")

    content_type = {
        ".css": "text/css; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(file_path.suffix.lower(), "application/octet-stream")

    data = file_path.read_bytes()
    start_response("200 OK", [("Content-Type", content_type), ("Content-Length", str(len(data)))] + security_headers(environ))
    return [data]


def current_user(environ):
    cookies = parse_cookies(environ)
    user_id = cookies.get("talexa_user_id")
    user_email = cookies.get("talexa_user_email")
    if not user_id or not user_id.isdigit() or not user_email:
        return None
    return {"user_ID": int(user_id), "Email": user_email}


def require_login(environ, start_response):
    user = current_user(environ)
    if user:
        return user
    redirect(environ, start_response, "/login?notice=Please%20log%20in%20first.")
    return None


def auth_cookie_headers(environ, user):
    suffix = cookie_security_suffix(environ)
    return [
        ("Set-Cookie", f"talexa_user_id={int(user['user_ID'])}; Path=/; Max-Age={SESSION_MAX_AGE_SECONDS}; HttpOnly; SameSite=Lax{suffix}"),
        ("Set-Cookie", f"talexa_user_email={quote(str(user['Email']))}; Path=/; Max-Age={SESSION_MAX_AGE_SECONDS}; HttpOnly; SameSite=Lax{suffix}"),
    ]


def clear_auth_cookie_headers(environ):
    suffix = cookie_security_suffix(environ)
    return [
        ("Set-Cookie", f"talexa_user_id=; Path=/; Max-Age=0; HttpOnly; SameSite=Lax{suffix}"),
        ("Set-Cookie", f"talexa_user_email=; Path=/; Max-Age=0; HttpOnly; SameSite=Lax{suffix}"),
    ]


def page_layout(title, body, user=None, notice="", wide=False, refresh_seconds=None):
    user_block = ""
    if user:
        user_block = f"""
          <div class="user-bar">
            <span class="user-email">{escape(user['Email'])}</span>
            <a href="/upload">Upload</a>
            <a href="/logout" class="logout">Logout</a>
          </div>
        """
    notice_block = f'<div class="notice">{escape(notice)}</div>' if notice else ""
    refresh_tag = f'<meta http-equiv="refresh" content="{int(refresh_seconds)}" />' if refresh_seconds else ""
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    {refresh_tag}
    <title>{escape(title)}</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link rel="stylesheet" href="/static/app.css" />
  </head>
  <body>
    <div class="shell">
      <header class="topbar">
        <a class="brand" href="/upload">TALEXA<span class="brand-dot"></span></a>
        {user_block}
      </header>
      {notice_block}
      {body}
    </div>
    <script>
      document.querySelectorAll('.robot-wrap img').forEach(function(img) {{
        img.addEventListener('click', function() {{ showToast('Hello! I am your AI lecturer.'); }});
      }});
      function showToast(msg) {{
        var t = document.getElementById('talexa-toast');
        if (!t) {{
          t = document.createElement('div');
          t.id = 'talexa-toast';
          t.style.cssText = 'position:fixed;bottom:20px;right:20px;background:#0d1f6e;color:#fff;padding:10px 18px;border-radius:12px;font-size:0.82rem;font-weight:700;border-left:4px solid #4db8e8;opacity:0;transform:translateY(10px);transition:all .3s;pointer-events:none;z-index:999;font-family:Space Grotesk,sans-serif;';
          document.body.appendChild(t);
        }}
        t.textContent = msg;
        t.style.opacity = '1'; t.style.transform = 'translateY(0)';
        setTimeout(function() {{ t.style.opacity = '0'; t.style.transform = 'translateY(10px)'; }}, 2400);
      }}
    </script>
  </body>
</html>"""


def auth_shell(title_text, subtitle_text, form_html):
    return f"""
    <main class="auth-page">
      <section class="auth-grid-talexa">
        <section class="hero-panel">
          <p class="eyebrow">AI-Powered Education</p>
          <h1>Turn your notes into an <span class="ac1">AI lecture.</span><br/>Like <span class="ac2">magic.</span></h1>
          <p class="hero-subtitle">Upload your slides or textbook, a portrait and voice sample &mdash; Talexa generates the full lecture for you.</p>
          <div class="hero-steps">
            <div class="hero-step"><span class="hero-step-num n1">1</span> Upload your PDF &amp; audio sample</div>
            <div class="hero-step"><span class="hero-step-num n2">2</span> Choose language &amp; content type</div>
            <div class="hero-step"><span class="hero-step-num n3">3</span> Download your AI-generated lecture</div>
          </div>
          <div class="robot-wrap"><img src="/assets/robot.png" alt="Talexa AI lecturer" /></div>
        </section>
        <section class="auth-form-panel">
          <div class="badge-row">
            <span class="badge badge-blue">Grad Project 2026</span>
            <span class="badge badge-cyan">End-to-end AI</span>
          </div>
          <h2 class="form-heading">{escape(title_text)}</h2>
          <p class="form-sub">{escape(subtitle_text)}</p>
          {form_html}
        </section>
      </section>
    </main>
    """


def login_page(notice=""):
    form = """
      <form method="post" action="/login" class="talexa-form" onsubmit="handleLogin(event)">
        <div class="field-wrap">
          <label for="login-email">Email address</label>
          <input type="email" id="login-email" name="email" placeholder="you@university.edu" required oninput="validateEmail('login-email','login-email-notice')" />
          <div class="field-notice" id="login-email-notice"></div>
        </div>
        <div class="field-wrap">
          <label for="login-pw">Password</label>
          <div class="pw-wrap">
            <input type="password" id="login-pw" name="password" placeholder="&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;" required />
            <button type="button" class="pw-toggle" onclick="togglePw('login-pw',this)">Show</button>
          </div>
        </div>
        <div class="field-notice" id="login-form-notice"></div>
        <button type="submit" class="button primary big" style="margin-top:4px;">Log in &rarr;</button>
      </form>
      <div class="divider"><span>or</span></div>
      <a class="button outline big full" href="/sign-up">Create an account</a>
      <script>
        function togglePw(id,btn){var i=document.getElementById(id);i.type=i.type==='password'?'text':'password';btn.textContent=i.type==='password'?'Show':'Hide';}
        function validateEmail(fid,nid){var v=document.getElementById(fid).value,n=document.getElementById(nid);if(v&&!v.includes('@')){n.textContent='Enter a valid email.';n.className='field-notice err';}else{n.textContent='';n.className='field-notice';}}
        function handleLogin(e){var em=document.getElementById('login-email').value,pw=document.getElementById('login-pw').value,n=document.getElementById('login-form-notice');if(!em||!em.includes('@')){e.preventDefault();n.textContent='Please enter a valid email.';n.className='field-notice err';}else if(!pw){e.preventDefault();n.textContent='Please enter your password.';n.className='field-notice err';}}
      </script>
    """
    return page_layout("TALEXA \u2014 Login", auth_shell("Welcome back.", "Sign in to your Talexa account to continue.", form), notice=notice)


def signup_page(notice=""):
    form = """
      <form method="post" action="/sign-up" class="talexa-form" onsubmit="handleSignup(event)">
        <div class="field-wrap">
          <label for="su-email">Email address</label>
          <input type="email" id="su-email" name="email" placeholder="you@university.edu" required oninput="validateEmail('su-email','su-email-notice')" />
          <div class="field-notice" id="su-email-notice"></div>
        </div>
        <div class="field-wrap">
          <label for="su-pw">Password</label>
          <div class="pw-wrap">
            <input type="password" id="su-pw" name="password" placeholder="Choose a strong password" required oninput="checkStrength()" />
            <button type="button" class="pw-toggle" onclick="togglePw('su-pw',this)">Show</button>
          </div>
          <div class="strength-bar"><div class="strength-fill" id="strength-fill"></div></div>
          <div class="strength-label" id="strength-label"></div>
        </div>
        <div class="field-notice" id="su-form-notice"></div>
        <button type="submit" class="button primary big" style="margin-top:4px;">Create account &rarr;</button>
      </form>
      <div class="divider"><span>or</span></div>
      <a class="button outline big full" href="/login">Back to log in</a>
      <script>
        function togglePw(id,btn){var i=document.getElementById(id);i.type=i.type==='password'?'text':'password';btn.textContent=i.type==='password'?'Show':'Hide';}
        function validateEmail(fid,nid){var v=document.getElementById(fid).value,n=document.getElementById(nid);if(v&&!v.includes('@')){n.textContent='Enter a valid email.';n.className='field-notice err';}else{n.textContent='';n.className='field-notice';}}
        function checkStrength(){var pw=document.getElementById('su-pw').value,f=document.getElementById('strength-fill'),l=document.getElementById('strength-label'),s=0;if(pw.length>=8)s++;if(/[A-Z]/.test(pw))s++;if(/[0-9]/.test(pw))s++;if(/[^A-Za-z0-9]/.test(pw))s++;var p=[0,25,50,75,100][s],c=['','#e24b4a','#ef9f27','#4db8e8','#1d9e75'],lb=['','Weak','Fair','Good','Strong'];f.style.width=p+'%';f.style.background=c[s]||'#e2e4ef';l.textContent=s>0?lb[s]:'';l.style.color=c[s]||'var(--muted)';}
        function handleSignup(e){var em=document.getElementById('su-email').value,pw=document.getElementById('su-pw').value,n=document.getElementById('su-form-notice');if(!em||!em.includes('@')){e.preventDefault();n.textContent='Please enter a valid email.';n.className='field-notice err';}else if(pw.length<6){e.preventDefault();n.textContent='Password must be at least 6 characters.';n.className='field-notice err';}}
      </script>
    """
    return page_layout("TALEXA \u2014 Sign Up", auth_shell("Create your account.", "Join Talexa and start generating AI lectures.", form), notice=notice)


def terms_page(user, notice=""):
    body = """
    <main class="terms-page">
      <section class="terms-panel">
        <h2>Terms &amp; Conditions</h2>
        <p class="terms-lead">Click each item to acknowledge it, then accept to continue.</p>
        <p class="terms-count" id="tcount">0 of 4 acknowledged</p>
        <ul class="terms-list" id="terms-list">
          <li onclick="checkTerm(this)"><div class="term-check"></div>Your uploads (images, portraits, textbooks, audio) are only used to generate your AI lecture. Nothing else.</li>
          <li onclick="checkTerm(this)"><div class="term-check"></div>All files are processed by our system solely to create the requested lecture content.</li>
          <li onclick="checkTerm(this)"><div class="term-check"></div>Your content stays private &mdash; we never share your uploads with third parties.</li>
          <li onclick="checkTerm(this)"><div class="term-check"></div>By uploading, you confirm you have the legal right to use the content and it does not violate copyright or privacy laws.</li>
        </ul>
        <div class="terms-actions">
          <form method="post" action="/terms/accept" id="terms-form">
            <button type="submit" class="button primary big" id="terms-btn" disabled style="opacity:.4;cursor:not-allowed;">Accept &amp; Continue &rarr;</button>
          </form>
          <a class="button outline big" href="/logout">Back to Login</a>
        </div>
        <script>
          var checked = 0;
          function checkTerm(el) {
            if (el.classList.contains('checked')) { el.classList.remove('checked'); el.querySelector('.term-check').textContent = ''; checked = Math.max(0, checked - 1); }
            else { el.classList.add('checked'); el.querySelector('.term-check').textContent = '\u2713'; checked++; }
            document.getElementById('tcount').textContent = checked + ' of 4 acknowledged';
            var btn = document.getElementById('terms-btn');
            if (checked === 4) { btn.disabled = false; btn.style.opacity = '1'; btn.style.cursor = 'pointer'; }
            else { btn.disabled = true; btn.style.opacity = '.4'; btn.style.cursor = 'not-allowed'; }
          }
        </script>
      </section>
    </main>
    """
    return page_layout("TALEXA \u2014 Terms", body, user=user, notice=notice)


def upload_page(environ, user, notice=""):
    session_id = query_int(environ, "session_id")
    session_record = get_session_record(session_id) if session_id else None
    session_status = str(session_record["status"]) if session_record else ""
    slides_output, video_output = existing_output_paths(session_id)
    session_chip = ""
    waiting_block = ""
    refresh_seconds = None
    if video_output and video_output.exists():
        waiting_block = ""
        refresh_seconds = None
    elif session_status == "completed":
        waiting_block = '<div class="generation-status working">Lecture ready &mdash; locating your files&hellip;</div>'
        refresh_seconds = 5
    elif session_status == "assembling":
        waiting_block = '<div class="generation-status working">Assembling your lecture video&hellip;</div>'
        refresh_seconds = 20
    elif session_status == "processing":
        waiting_block = '<div class="generation-status working">Generating lecture assets&hellip; this page updates automatically.</div>'
        refresh_seconds = 20
    elif session_status == "failed":
        waiting_block = '<div class="generation-status error">Generation failed. Please try again.</div>'
    elif session_id:
        waiting_block = '<div class="generation-status working">Generating&hellip; checking for your lecture video.</div>'
        refresh_seconds = 20
    download_links = render_downloads(session_id, slides_output, video_output)
    body = f"""
    <main class="upload-page">
      {session_chip}
      <section class="upload-layout">
        <section class="hero-panel upload-hero">
          <p class="eyebrow">Generate a Lecture</p>
          <h1>Upload your <span class="ac1">materials</span> and let Talexa <span class="ac2">cook.</span></h1>
          <p class="hero-subtitle">Provide a PDF, your portrait and a voice sample &mdash; Talexa handles slides, narration, and a full video lecture.</p>
          <div class="hero-steps">
            <div class="hero-step" id="step-pdf"><span class="hero-step-num n1">1</span> Your PDF slides or textbook</div>
            <div class="hero-step" id="step-img"><span class="hero-step-num n2">2</span> A portrait photo of you</div>
            <div class="hero-step" id="step-wav"><span class="hero-step-num n3">3</span> A short WAV voice sample</div>
          </div>
          <div class="robot-wrap"><img src="/assets/robot.png" alt="Talexa AI lecturer" /></div>
        </section>
        <section class="upload-panel">
          <h2 class="upload-panel-title">New generation</h2>
          <p class="upload-panel-sub">Fill all fields and hit Generate.</p>
          <div class="progress-track"><div class="progress-fill" id="prog-fill"></div></div>
          <p class="progress-label" id="prog-label">0 of 3 files ready</p>
          <form method="post" action="/generate" enctype="multipart/form-data" class="upload-form" onsubmit="return checkAllFiles()">
            <div class="upload-card" id="card-pdf">
              <div class="upload-card-header">
                <div class="upload-card-icon">&#128196;</div>
                <div><div class="upload-card-label">Text PDF</div><small>Slides deck or textbook chapter</small></div>
              </div>
              <input type="file" name="text_pdf" accept="application/pdf" required onchange="markFile('pdf',this)" />
              <div style="margin-top:10px;">
                <p class="radio-section-label">Content type</p>
                <div class="radio-row">
                  <label><input type="radio" name="text_type" value="slides" checked /> Slides</label>
                  <label><input type="radio" name="text_type" value="textbook" /> Textbook</label>
                </div>
              </div>
            </div>
            <div class="upload-card" id="card-img">
              <div class="upload-card-header">
                <div class="upload-card-icon">&#128247;</div>
                <div><div class="upload-card-label">Portrait image</div><small>PNG, JPG, JPEG, or WEBP</small></div>
              </div>
              <input type="file" name="portrait_png" accept="image/png,image/jpeg,image/webp" required onchange="markFile('img',this)" />
            </div>
            <div class="upload-card" id="card-wav">
              <div class="upload-card-header">
                <div class="upload-card-icon">&#127908;</div>
                <div><div class="upload-card-label">Voice sample</div><small>WAV only &mdash; 5+ seconds of clear speech</small></div>
              </div>
              <input type="file" name="audio_wav" accept="audio/wav,.wav" required onchange="markFile('wav',this)" />
            </div>
            <div class="upload-card">
              <p class="radio-section-label">Output language</p>
              <div class="radio-row">
                <label><input type="radio" name="language" value="english" checked /> English</label>
                <label><input type="radio" name="language" value="arabic" /> Arabic</label>
              </div>
            </div>
            <button type="submit" class="button primary big" id="gen-btn">Generate Lecture &rarr;</button>
          </form>
          {waiting_block}
          {download_links}
          <script>
            var files = {{pdf:false,img:false,wav:false}};
            function markFile(key, inp) {{
              if (inp.files && inp.files.length > 0) {{
                files[key] = true;
                document.getElementById('card-'+key).classList.add('ready');
                var step = {{pdf:'step-pdf',img:'step-img',wav:'step-wav'}}[key];
                if (step) {{ var s=document.getElementById(step); if(s){{s.style.color='#7dd4f5';s.style.background='rgba(77,184,232,0.12)';}} }}
              }}
              var done = Object.values(files).filter(Boolean).length;
              document.getElementById('prog-fill').style.width = Math.round(done/3*100)+'%';
              document.getElementById('prog-label').textContent = done+' of 3 files ready';
            }}
            function checkAllFiles() {{
              var done = Object.values(files).filter(Boolean).length;
              if (done < 3) {{ alert('Please upload all 3 files before generating.'); return false; }}
              var btn = document.getElementById('gen-btn');
              btn.textContent = 'Generating\u2026'; btn.disabled = true; btn.style.opacity = '0.6';
              return true;
            }}
          </script>
        </section>
      </section>
    </main>
    """
    return page_layout("TALEXA \u2014 Generate", body, user=user, notice=notice, refresh_seconds=refresh_seconds)


def render_downloads(session_id, slides_output, video_output):
    if not session_id:
        return ""
    links = []
    if video_output and video_output.exists():
        links.append(f'<a class="download-button dl-video" href="/download/lecture?session_id={session_id}">&#9654; Download Lecture Video</a>')
    if slides_output and slides_output.exists() and slides_output.is_file():
        links.append(f'<a class="download-button dl-slides" href="/download/slides?session_id={session_id}">&#128196; Download Lecture Slides</a>')
    if not links:
        return ""
    return f'<div class="download-stack">{"".join(links)}</div>'


def handle_login(environ, start_response):
    if environ.get("REQUEST_METHOD") == "GET":
        return html_response(environ, start_response, login_page(query_notice(environ)))

    form = parse_post_data(environ)
    email = clean_text(form.get("email", ""), 320)
    password = form.get("password", "").strip()
    if not email or not password:
        return html_response(environ, start_response, login_page("Please enter both email and password."), "400 Bad Request")

    is_valid, message, user = authenticate_user(email, password)
    if not is_valid or user is None:
        if user is None and "No account" in message:
            message = "This email does not exist. Please sign up first."
        return html_response(environ, start_response, login_page(message), "401 Unauthorized")

    return redirect(environ, start_response, "/terms", auth_cookie_headers(environ, user))


def handle_signup(environ, start_response):
    if environ.get("REQUEST_METHOD") == "GET":
        return html_response(environ, start_response, signup_page(query_notice(environ)))

    form = parse_post_data(environ)
    email = clean_text(form.get("email", ""), 320)
    password = form.get("password", "").strip()
    if not email or not password:
        return html_response(environ, start_response, signup_page("Please enter both email and password."), "400 Bad Request")
    if get_user_by_email(email):
        return html_response(environ, start_response, signup_page("An account already exists for this email."), "409 Conflict")
    if create_user(email, password):
        return redirect(environ, start_response, "/login?notice=Account%20created%20successfully.%20You%20can%20now%20log%20in.")
    return html_response(environ, start_response, signup_page("Unable to create the account. Please try again."), "500 Internal Server Error")


def handle_generate(environ, start_response):
    user = require_login(environ, start_response)
    if not user:
        return [b""]

    session_id = None
    session_record_created = False
    try:
        form = parse_multipart(environ)
        text_type = clean_text(field_value(form, "text_type", "slides"), 20).lower()
        language = clean_text(field_value(form, "language", "english"), 20).lower()

        if text_type not in {"slides", "textbook"}:
            return redirect(environ, start_response, "/upload?notice=Please%20choose%20a%20valid%20text%20type.")
        if language not in {"english", "arabic"}:
            return redirect(environ, start_response, "/upload?notice=Please%20choose%20a%20valid%20language.")

        text_file = form["text_pdf"] if "text_pdf" in form else None
        portrait_file = form["portrait_png"] if "portrait_png" in form else None
        audio_file = form["audio_wav"] if "audio_wav" in form else None
        if not uploaded_field_has_file(text_file) or not uploaded_field_has_file(portrait_file) or not uploaded_field_has_file(audio_file):
            return redirect(environ, start_response, "/upload?notice=Please%20upload%20the%20text%20PDF,%20portrait,%20and%20audio%20WAV.")

        session_id = max(get_next_session_number(PROJECT_ROOT), get_max_session_id() + 1)

        source_upload_path = save_uploaded_field(text_file, ".pdf")
        portrait_upload_path = save_uploaded_field(portrait_file, ".png")
        audio_upload_path = save_uploaded_field(audio_file, ".wav")

        validated_source_path = validate_source_document(str(source_upload_path), text_type)
        validated_portrait_path = validate_portrait_image(str(portrait_upload_path))
        validated_audio_path = validate_and_prepare_audio(str(audio_upload_path))

        create_session_record(
            session_id=session_id,
            user_id=int(user["user_ID"]),
            text_type=text_type,
            language=language,
            status="processing",
            expire_time=(datetime.now(timezone.utc) + timedelta(seconds=SESSION_MAX_AGE_SECONDS)).isoformat(),
        )
        session_record_created = True

        pipeline_session = create_session(
            PROJECT_ROOT,
            pdf_file_path=str(validated_source_path),
            audio_file_path=str(validated_audio_path),
            portrait_file_path=str(validated_portrait_path),
            session_number=session_id,
        )

        upsert_storage_paths(
            user_id=int(user["user_ID"]),
            session_id=session_id,
            image_path=str(pipeline_session.stored_portrait_path) if pipeline_session.stored_portrait_path else None,
            audio_path=str(pipeline_session.stored_audio_path) if pipeline_session.stored_audio_path else None,
            slides_path=str(pipeline_session.stored_pdf_path) if text_type == "slides" else None,
            textbook_path=str(pipeline_session.stored_pdf_path) if text_type == "textbook" else None,
        )
        worker = threading.Thread(
            target=run_generation_job,
            kwargs={
                "session_id": session_id,
                "user_id": int(user["user_ID"]),
                "pipeline_session": pipeline_session,
                "text_type": text_type,
                "language": language,
            },
            daemon=True,
        )
        worker.start()

        return redirect(
            environ,
            start_response,
            f"/upload?session_id={session_id}&notice=Generation%20started.%20This%20page%20will%20keep%20checking%20for%20the%20outputs.",
        )
    except Exception as exc:
        if session_id and session_record_created:
            update_session_status(session_id, "failed")
        target = f"/upload?session_id={session_id}" if session_id else "/upload"
        return redirect(environ, start_response, f"{target}&notice={quote('Generation failed: ' + str(exc))}" if "?" in target else f"{target}?notice={quote('Generation failed: ' + str(exc))}")


def field_value(form, name, default=""):
    field = form[name] if name in form else None
    if field is None:
        return default
    value = field.value
    return value if isinstance(value, str) else default


def uploaded_field_has_file(field):
    return field is not None and bool(getattr(field, "filename", ""))


def save_uploaded_field(field, suffix):
    staging_dir = Path(tempfile.gettempdir()) / "talexa_upload_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(field.filename or "upload").name
    target_path = staging_dir / f"{uuid.uuid4().hex}_{safe_name}"
    if suffix and target_path.suffix.lower() != suffix.lower():
        target_path = target_path.with_suffix(suffix)

    with target_path.open("wb") as handle:
        shutil.copyfileobj(field.file, handle)
    return target_path


def run_generation_job(session_id, user_id, pipeline_session, text_type, language):
    try:
        update_session_status(session_id, "processing")

        if text_type == "textbook":
            from PIPELINE.run_textbook_pipeline import run_textbook_pipeline

            result = run_textbook_pipeline(pipeline_session, lecture_language=language)
        elif text_type == "slides":
            from PIPELINE.run_slides_pipeline import run_slides_pipeline

            result = run_slides_pipeline(pipeline_session, lecture_language=language)
        else:
            raise ValueError("text_type must be either 'textbook' or 'slides'.")

        slides_output_path = result.get("slides_pdf_path")
        upsert_storage_paths(
            user_id=user_id,
            session_id=session_id,
            image_path=str(pipeline_session.stored_portrait_path) if pipeline_session.stored_portrait_path else None,
            audio_path=str(pipeline_session.stored_audio_path) if pipeline_session.stored_audio_path else None,
            slides_path=str(pipeline_session.stored_pdf_path) if text_type == "slides" else None,
            textbook_path=str(pipeline_session.stored_pdf_path) if text_type == "textbook" else None,
            slides_output_path=str(slides_output_path) if slides_output_path else None,
            video_output_path=None,
        )

        update_session_status(session_id, "assembling")

        from PIPELINE.assembly import assemble_session_video

        assembly_result = assemble_session_video(
            session_id=pipeline_session.session_number,
            language=language,
        )
        video_output_path = assembly_result.get("video_output_path")

        upsert_storage_paths(
            user_id=user_id,
            session_id=session_id,
            image_path=str(pipeline_session.stored_portrait_path) if pipeline_session.stored_portrait_path else None,
            audio_path=str(pipeline_session.stored_audio_path) if pipeline_session.stored_audio_path else None,
            slides_path=str(pipeline_session.stored_pdf_path) if text_type == "slides" else None,
            textbook_path=str(pipeline_session.stored_pdf_path) if text_type == "textbook" else None,
            slides_output_path=str(slides_output_path) if slides_output_path else None,
            video_output_path=str(video_output_path) if video_output_path else None,
        )
        update_session_status(session_id, "completed")
    except Exception as exc:
        update_session_status(session_id, "failed")
        error_path = pipeline_session.output_dir / "generation_error.txt"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(traceback.format_exc(), encoding="utf-8")


def existing_output_paths(session_id):
    if not session_id:
        return None, None

    fallback_slides_path, fallback_video_path = discover_session_outputs(session_id)
    record = get_storage_paths(session_id)
    if record is None:
        return fallback_slides_path, fallback_video_path

    slides_path = Path(record["slides_output_path"]) if record["slides_output_path"] else None
    video_path = Path(record["video_output_path"]) if record["video_output_path"] else None
    if slides_path is None or not slides_path.exists():
        slides_path = fallback_slides_path
    if video_path is None or not video_path.exists():
        video_path = fallback_video_path
    return slides_path, video_path


def discover_session_outputs(session_id):
    output_dir = PROJECT_ROOT / "sessions" / str(session_id) / "output"
    if not output_dir.exists():
        return None, None

    mp4_candidates = sorted(
        output_dir.glob("*.mp4"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    pdf_candidates = sorted(
        output_dir.glob("*.pdf"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return (
        pdf_candidates[0] if pdf_candidates else None,
        mp4_candidates[0] if mp4_candidates else None,
    )


def zip_directory(directory):
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file():
                archive.write(file_path, arcname=file_path.relative_to(directory))
    return memory_file.getvalue()


def handle_download(environ, start_response, kind):
    user = require_login(environ, start_response)
    if not user:
        return [b""]
    session_id = query_int(environ, "session_id")
    slides_output, video_output = existing_output_paths(session_id)
    target = slides_output if kind == "slides" else video_output
    if target is None or not target.exists():
        return redirect(environ, start_response, f"/upload?session_id={session_id or ''}&notice=Requested%20output%20was%20not%20found.")
    if target.is_dir():
        return binary_response(environ, start_response, f"session_{session_id}_lecture.zip", "application/zip", zip_directory(target))
    content_type = "application/pdf" if kind == "slides" else "video/mp4"
    return binary_response(environ, start_response, target.name, content_type, target.read_bytes())


def application(environ, start_response):
    try:
        path = environ.get("PATH_INFO", "/")
        method = environ.get("REQUEST_METHOD", "GET")
        if should_redirect_to_https(environ):
            return redirect(environ, start_response, https_url(environ))
        allowed, limit_scope = token_bucket_allowed(environ)
        if not allowed:
            return plain_text(environ, start_response, "429 Too Many Requests", f"Too many {limit_scope} requests. Please try again later.")
        if path.startswith("/static/") or path.startswith("/assets/"):
            return serve_static(environ, start_response)
        if path == "/":
            return redirect(environ, start_response, "/upload" if current_user(environ) else "/login")
        if path == "/login":
            return handle_login(environ, start_response)
        if path == "/sign-up":
            return handle_signup(environ, start_response)
        if path == "/logout":
            return redirect(environ, start_response, "/login", clear_auth_cookie_headers(environ))
        if path == "/terms":
            user = require_login(environ, start_response)
            if not user:
                return [b""]
            return html_response(environ, start_response, terms_page(user, query_notice(environ)))
        if path == "/terms/accept" and method == "POST":
            user = require_login(environ, start_response)
            if not user:
                return [b""]
            return redirect(environ, start_response, "/upload")
        if path == "/upload":
            user = require_login(environ, start_response)
            if not user:
                return [b""]
            return html_response(environ, start_response, upload_page(environ, user, query_notice(environ)))
        if path == "/generate" and method == "POST":
            return handle_generate(environ, start_response)
        if path == "/download/slides":
            return handle_download(environ, start_response, "slides")
        if path == "/download/lecture":
            return handle_download(environ, start_response, "lecture")
        return plain_text(environ, start_response, "404 Not Found", "Page not found.")
    except Exception as exc:
        return plain_text(environ, start_response, "500 Internal Server Error", f"An internal error occurred: {exc}")


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", "8000"))
    print(f"TALEXA running at http://0.0.0.0:{port}", flush=True)
    with make_server("0.0.0.0", port, application) as httpd:
        httpd.serve_forever()