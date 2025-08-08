import sqlite3, pathlib, smtplib, ssl
# from passlib.hash import bcrypt
import bcrypt
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
import streamlit as st
from datetime import datetime, timedelta
from email.mime.text import MIMEText

DB_PATH = pathlib.Path(__file__).with_name("users.db")
SECRET = "replace-me-with-something-secret-and-random"
TOKEN_TTL = 1800  # seconds (30 min)

serializer = URLSafeTimedSerializer(SECRET)

# ---------- DB helpers ----------
def _init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS users (
                 username TEXT PRIMARY KEY,
                 password_hash TEXT NOT NULL,
                 email TEXT UNIQUE NOT NULL
               );"""
        )

_init_db()

def add_user(username, plain_password, email):
    h = bcrypt.hashpw(plain_password.encode(), bcrypt.gensalt()).decode()
    # h = bcrypt.hash(plain_password)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO users(username, password_hash, email) VALUES(?,?,?)",
            (username, h, email),
        )

def verify_user(username, plain_password):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE username=?", (username,)
        ).fetchone()
    if row:
        # return bcrypt.verify(plain_password, row[0])
        return bcrypt.checkpw(plain_password.encode(), row[0])

    return False

def get_email(username):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT email FROM users WHERE username=?", (username,)).fetchone()
    return row[0] if row else None

# ---------- Reset token ----------
def generate_reset_token(username):
    return serializer.dumps(username, salt="reset")

def verify_reset_token(token, max_age=TOKEN_TTL):
    try:
        username = serializer.loads(token, salt="reset", max_age=max_age)
        return username
    except (BadSignature, SignatureExpired):
        return None

# ---------- Gmail sender for reset ----------
def send_reset_email(service, to_email, reset_link):
    subject = "Password Reset – Recruiter Dashboard"
    body = f"""\
Hi,

You requested a password reset.
Click the link below (valid 30 min):

{reset_link}

If you did not request this, please ignore this e-mail.
"""
    from gmail_helper import send_gmail
    send_gmail(service, to_email, subject, body)