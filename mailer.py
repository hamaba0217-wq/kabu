# -*- coding: utf-8 -*-
"""
メール送信

Gmail の「アプリパスワード」を使ってSMTP送信します。
通常のGmailログインパスワードでは送れません（Googleが2022年に廃止したため）。

必要な環境変数
  MAIL_FROM      送信元Gmailアドレス
  MAIL_PASSWORD  Googleで発行したアプリパスワード（16桁）
  MAIL_TO        送信先アドレス（自分宛でOK。カンマ区切りで複数可）
"""

from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formatdate

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "465"))


class MailNotConfigured(Exception):
    pass


def send(subject: str, html: str, text: str = "") -> None:
    sender = os.environ.get("MAIL_FROM")
    password = os.environ.get("MAIL_PASSWORD")
    to = os.environ.get("MAIL_TO") or sender

    if not (sender and password and to):
        raise MailNotConfigured(
            "メール設定が足りません。MAIL_FROM / MAIL_PASSWORD / MAIL_TO を設定してください。"
        )

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    msg["Date"] = formatdate(localtime=True)
    msg.set_content(text or "HTMLメールに対応した環境でご覧ください。")
    msg.add_alternative(html, subtype="html")

    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as s:
        s.login(sender, password)
        s.send_message(msg)


# ---------------------------------------------------------------------------
# HTML の組み立て
# ---------------------------------------------------------------------------

STYLE = """
<style>
 body { font-family: -apple-system, "Hiragino Sans", "Meiryo", sans-serif;
        font-size: 14px; color: #222; line-height: 1.7; }
 .box  { border-left: 4px solid #888; padding: 8px 14px; margin: 14px 0;
         background: #fafafa; }
 .green  { border-color: #2e7d32; } .yellow { border-color: #f9a825; }
 .red    { border-color: #c62828; }
 table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }
 th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: right; }
 th { background: #f0f0f0; text-align: center; }
 td.l, th.l { text-align: left; }
 .note { color: #666; font-size: 12px; margin-top: 20px;
         border-top: 1px solid #ddd; padding-top: 10px; }
 .warn { color: #c62828; font-weight: bold; }
</style>
"""


def df_to_html(df, left_cols=()) -> str:
    if df is None or len(df) == 0:
        return "<p>該当なし</p>"
    head = "".join(
        f'<th class="l">{c}</th>' if c in left_cols else f"<th>{c}</th>"
        for c in df.columns
    )
    rows = []
    for _, r in df.iterrows():
        tds = "".join(
            f'<td class="l">{r[c]}</td>' if c in left_cols else f"<td>{r[c]}</td>"
            for c in df.columns
        )
        rows.append(f"<tr>{tds}</tr>")
    return f"<table><tr>{head}</tr>{''.join(rows)}</table>"


def wrap(title: str, body: str, footer: str = "") -> str:
    return f"""<html><head><meta charset="utf-8">{STYLE}</head><body>
<h2>{title}</h2>
{body}
<div class="note">
{footer}
このメールは自動送信です。記載内容は投資助言ではありません。
売買の判断と結果責任はご自身にあります。
</div>
</body></html>"""
