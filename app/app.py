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
        ("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; form-action 'self'; base-uri 'self'"),
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
            <span>{escape(user['Email'])}</span>
            <a href="/upload">Upload</a>
            <a href="/logout">Logout</a>
          </div>
        """
    notice_block = f'<div class="notice">{escape(notice)}</div>' if notice else ""
    shell_class = "shell wide" if wide else "shell"
    refresh_tag = f'<meta http-equiv="refresh" content="{int(refresh_seconds)}" />' if refresh_seconds else ""
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    {refresh_tag}
    <title>{escape(title)}</title>
    <link rel="stylesheet" href="/static/app.css" />
  </head>
  <body>
    <div class="{shell_class}">
      <header class="topbar">
        <a class="brand" href="/upload">TALEXA</a>
        {user_block}
      </header>
      {notice_block}
      {body}
    </div>
  </body>
</html>"""


def auth_shell(title_text, form_html):
    return f"""
    <main class="auth-page">
      <section class="auth-grid-talexa">
        <section class="hero-panel robot-only-panel">
          <div class="robot-wrap"><img src="/assets/robot.png" alt="Talexa robot" /></div>
        </section>
        <section class="auth-form-panel">
          <h2>{escape(title_text)}</h2>
          {form_html}
        </section>
      </section>
    </main>
    """


def login_page(notice=""):
    form = """
      <form method="post" action="/login" class="talexa-form">
        <input type="email" name="email" placeholder="Email address" required />
        <input type="password" name="password" placeholder="Password" required />
        <button type="submit" class="button outline big">log in</button>
      </form>
      <div class="divider"><span>or</span></div>
      <a class="button outline big full" href="/sign-up">sign up</a>
    """
    return page_layout("TALEXA Login", auth_shell("Welcome to Talexa! Your personal AI lecture", form), notice=notice, wide=True)


def signup_page(notice=""):
    form = """
      <form method="post" action="/sign-up" class="talexa-form">
        <input type="email" name="email" placeholder="Email" required />
        <input type="password" name="password" placeholder="Password" required />
        <button type="submit" class="button outline big">create account</button>
      </form>
      <div class="divider"><span>or</span></div>
      <a class="button outline big full" href="/login">back to log in</a>
    """
    return page_layout("TALEXA Sign Up", auth_shell("Create account:", form), notice=notice, wide=True)


def terms_page(user, notice=""):
    body = """
    <main class="terms-page">
      <section class="terms-panel">
        <h2>TERMS AND CONDITIONS</h2>
        <ul>
          <li>By using our platform, you agree that any images, portraits, textbooks, and audio files you upload will be processed solely for generating AI-generated slides and lectures.</li>
          <li>All uploaded files are used by the system to process your content and generate the requested lecture.</li>
          <li>Your uploaded content is never shared with third parties.</li>
          <li>By uploading any material, you confirm that you have the legal right to use such content and that it does not violate copyright, intellectual property, or privacy laws.</li>
        </ul>
        <form method="post" action="/terms/accept">
          <button type="submit" class="button outline big">ACCEPT TERMS AND CONDITIONS</button>
        </form>
        <a class="button outline big full" href="/logout">BACK TO LOGIN</a>
      </section>
    </main>
    """
    return page_layout("TALEXA Terms", body, user=user, notice=notice, wide=True)


def upload_page(environ, user, notice=""):
    session_id = query_int(environ, "session_id")
    session_record = get_session_record(session_id) if session_id else None
    session_status = str(session_record["status"]) if session_record else ""
    slides_output, video_output = existing_output_paths(session_id)
    session_chip = f'<div class="session-chip">Current Session: {session_id}</div>' if session_id else ""
    waiting_block = ""
    refresh_seconds = None
    if session_status == "assembling":
        waiting_block = '<div class="generation-status">lecture is being assembled</div>'
        refresh_seconds = 20
    elif session_status == "processing":
        waiting_block = '<div class="generation-status">Generating lecture assets... this page will keep checking for progress.</div>'
        refresh_seconds = 20
    elif session_status == "failed":
        waiting_block = '<div class="generation-status error">Generation failed. Please review the session files or try again.</div>'
    elif session_id and not (video_output and video_output.exists()):
        waiting_block = '<div class="generation-status">Generating... this page will keep checking for the lecture video.</div>'
        refresh_seconds = 20
    download_links = render_downloads(session_id, slides_output, video_output)
    body = f"""
    <main class="upload-page">
      {session_chip}
      <section class="upload-layout">
        <section class="hero-panel upload-hero robot-only-panel">
          <div class="robot-wrap upload-robot"><img src="/assets/robot.png" alt="Talexa robot" /></div>
        </section>
        <section class="upload-panel">
          <form method="post" action="/generate" enctype="multipart/form-data" class="upload-form">
            <label class="upload-card">
              <span>Upload Text</span>
              <input type="file" name="text_pdf" accept="application/pdf" required />
              <small>PDF only</small>
              <div class="radio-row">
                <label><input type="radio" name="text_type" value="slides" checked /> Slides</label>
                <label><input type="radio" name="text_type" value="textbook" /> Textbook Chapter</label>
              </div>
            </label>
            <label class="upload-card">
              <span>Upload Portrait</span>
              <input type="file" name="portrait_png" accept="image/png,image/jpeg,image/webp" required />
              <small>PNG, JPG, JPEG, or WEBP</small>
            </label>
            <label class="upload-card">
              <span>Upload Audio Sample</span>
              <input type="file" name="audio_wav" accept="audio/wav,.wav" required />
              <small>WAV only</small>
            </label>
            <div class="upload-card">
              <span>Choose Language</span>
              <div class="radio-row">
                <label><input type="radio" name="language" value="english" checked /> English</label>
                <label><input type="radio" name="language" value="arabic" /> Arabic</label>
              </div>
            </div>
            <button type="submit" class="button outline generate-button">GENERATE</button>
          </form>
          {waiting_block}
          {download_links}
        </section>
      </section>
    </main>
    """
    return page_layout("TALEXA Upload", body, user=user, notice=notice, wide=True, refresh_seconds=refresh_seconds)


def render_downloads(session_id, slides_output, video_output):
    if not session_id:
        return ""
    links = []
    if video_output and video_output.exists():
        links.append(f'<a class="button outline download-button" href="/download/lecture?session_id={session_id}">Download Lecture Video</a>')
    if slides_output and slides_output.exists() and slides_output.is_file():
        links.append(f'<a class="button outline download-button" href="/download/slides?session_id={session_id}">Download Lecture Slides</a>')
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
        error_path.write_text(str(exc), encoding="utf-8")


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
    print(f"TALEXA running at http://127.0.0.1:{port}", flush=True)
    with make_server("127.0.0.1", port, application) as httpd:
        httpd.serve_forever()
