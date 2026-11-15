#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import math
import random
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from decimal import Decimal, getcontext, ROUND_HALF_UP
from collections import Counter, defaultdict

import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, ContextTypes
)

getcontext().prec = 50
PI = Decimal("3.14159265358979323846264338327950288419716939937510")
MULT = Decimal("7.30548851")
LIMIT = Decimal(27)

API_BASE = "https://jnd28.pw/api/kj?limit={}"
TEMA_RANGE = list(range(28))
ALL_TYPES = ["大单", "大双", "小单", "小双"]

# 你的频道
CHANNEL_USERNAME = "@qw888nb" 

# ══════════ 基础 ══════════
def classify(n):
    size = "大" if n >= 14 else "小"
    par = "单" if n % 2 else "双"
    return size + par

def charity_compute(a, b, c, S):
    inner = Decimal(a) + Decimal(b) + Decimal(c) + Decimal(S) * PI
    val = inner * MULT
    steps = 0
    while val > LIMIT:
        val /= PI
        steps += 1
    return int(val), steps

# ══════════ 极速异步数据拉取 ══════════
async def fetch_history():
    async with httpx.AsyncClient(timeout=10.0) as client:
        url = API_BASE.format(20)
        try:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            obj = resp.json()
            if obj.get("code") != 0: return None
            hist = []
            for item in obj.get("data", []):
                qh = int(item["qh"])
                A, B, C = map(int, item["pc28"].split("+"))
                he = int(item["he"])
                hist.append((qh, A, B, C, he))
            return sorted(hist, key=lambda x: x[0])
        except Exception as e:
            print(f"API拉取失败: {e}")
            return None

# ══════════ 特码 8 方法 ══════════
def tema_charity(a, b, c):
    val = (Decimal(a)/100 + Decimal(b)/100 + Decimal(c)/100 * Decimal(a) + Decimal(b) + Decimal(c))
    val = val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    s = str(val); ip, fp = s.split(".")
    i1 = int(ip[-1]); i2 = int(ip[-2]) if len(ip) >= 2 else 0
    d1 = int(fp[0]); d2 = int(fp[1])
    cands = [d1+d2, i1+d1, i2+d2, i1+d2]
    return [v for v in cands if 0 <= v <= 27]

def tema_bayes(h, alpha=1.0):
    c = Counter(he for _,_,_,_,he in h); n = sum(c.values())
    p = {v: (c.get(v,0)+alpha)/(n+alpha*28) for v in TEMA_RANGE}
    return sorted(p.items(), key=lambda x: x[1], reverse=True)

def tema_ewma(h, lam=0.85):
    n = len(h)
    w = [lam**(n-1-i) for i in range(n)]; tw = sum(w)
    p = {v: 0.0 for v in TEMA_RANGE}
    for i,(_,_,_,_,he) in enumerate(h): p[he] += w[i]/tw
    return sorted(p.items(), key=lambda x: x[1], reverse=True)

def tema_markov(h, alpha=0.5):
    if len(h) < 2: return tema_bayes(h)
    tr = defaultdict(Counter)
    for i in range(len(h)-1): tr[h[i][4]][h[i+1][4]] += 1
    row = tr.get(h[-1][4], Counter()); n = sum(row.values())
    p = {v: (row.get(v,0)+alpha)/(n+alpha*28) for v in TEMA_RANGE}
    return sorted(p.items(), key=lambda x: x[1], reverse=True)

def tema_knn(h, k=10, window=5):
    if len(h) < window + k: return tema_bayes(h)
    vals = [he for _,_,_,_,he in h]
    target = vals[-window:]
    cands = []
    for i in range(window, len(vals)):
        feats = vals[i-window:i]
        d = math.sqrt(sum((a-b)**2 for a,b in zip(feats, target)))
        cands.append((d, vals[i]))
    cands.sort(key=lambda x: x[0])
    w = Counter()
    for d, v in cands[:k]: w[v] += 1.0/(d+1e-6)
    total = sum(w.values()) or 1
    return sorted([(v, c/total) for v,c in w.items()], key=lambda x: x[1], reverse=True)

def tema_mod(h):
    a, b, c = h[-1][1], h[-1][2], h[-1][3]
    S = a + b + c
    ops = [S%28, (a+b)%28, (b+c)%28, (a+c)%28, (a*b+c)%28, (S*7)%28, abs(a-b)+c, abs(b-c)+a]
    cnt = Counter(v for v in ops if 0 <= v <= 27)
    total = sum(cnt.values()) or 1
    return sorted([(v, c/total) for v,c in cnt.items()], key=lambda x: x[1], reverse=True)

def tema_kalman(h):
    if not h: return [(14, 0.04)]
    x = h[0][4]; P = 1.0
    for _,_,_,_,he in h[1:]:
        K = P/(P+1.0)
        x = x + K*(he-x)
        P = (1-K)*P + 0.1
    center = max(0, min(27, int(round(x))))
    return [(center, 0.5), (max(0,center-1), 0.2), (min(27,center+1), 0.2)]

def tema_mc(h, n_sim=1000, seed=42):
    c = Counter(he for _,_,_,_,he in h); n = sum(c.values())
    if n == 0: return [(14, 0.04)]
    p = {v: (c.get(v,0)+1.0)/(n+28) for v in TEMA_RANGE}
    types = list(p.keys()); weights = [p[v] for v in types]
    rng = random.Random(seed); sim = Counter()
    for _ in range(n_sim): sim[rng.choices(types, weights=weights)[0]] += 1
    return sorted([(v, sim.get(v,0)/n_sim) for v in TEMA_RANGE], key=lambda x: x[1], reverse=True)

WEIGHTS = {"charity":0.10, "bayes":0.15, "ewma":0.12, "markov":0.12, "knn":0.10, "mod":0.08, "kalman":0.08, "mc":0.10, "uniform":0.15}

def fuse_tema(h):
    latest = h[-1]
    methods = {
        "charity": [(v, 1.0/max(1,len(tema_charity(latest[1],latest[2],latest[3])))) for v in tema_charity(latest[1],latest[2],latest[3])],
        "bayes":  tema_bayes(h), "ewma":   tema_ewma(h), "markov": tema_markov(h),
        "knn":    tema_knn(h), "mod":    tema_mod(h), "kalman": tema_kalman(h),
        "mc":     tema_mc(h), "uniform":[(v, 1/28) for v in TEMA_RANGE],
    }
    fused = {v: 0.0 for v in TEMA_RANGE}
    for name, items in methods.items():
        w = WEIGHTS.get(name, 0)
        total = sum(p for _,p in items) or 1
        for v, p in items: fused[v] += w * (p/total)
    total = sum(fused.values()) or 1
    return sorted([(v, p/total) for v,p in fused.items()], key=lambda x: x[1], reverse=True)

def dual_group(h):
    latest, prev, prev2 = h[-1], h[-2], h[-3]
    n, _ = charity_compute(latest[1], prev[2], prev2[3], latest[4])
    base = classify(n)
    s, p = base[0], base[1]
    d = n % 10
    same = [s+"单", s+"双"] if d <= 4 else ["大"+p, "小"+p]
    dim = "大小" if d <= 4 else "单双"
    guard = [t for t in same if t != base][0]
    return {"main": base, "guard": guard, "dim": dim, "n": n}

def miss_streak(h, top_n=1, lookback=12):
    base = top_n/28.0
    streak = 0
    for i in range(len(h)-1, max(0, len(h)-lookback-1), -1):
        train = h[:i]
        if len(train) < 10: break
        top = [v for v,_ in fuse_tema(train)[:top_n]]
        if h[i][4] in top: break
        streak += 1
    prob = (1-base)**streak if streak > 0 else 1.0
    return {"streak": streak, "prob": prob, "base": base}

# ══════════ Bot 核心逻辑 ══════════
CACHE = {"history": None, "ts": 0}

async def get_history():
    import time
    if CACHE["history"] is None or time.time() - CACHE["ts"] > 120:
        h = await fetch_history()
        if h:
            CACHE["history"] = h
            CACHE["ts"] = time.time()
    return CACHE["history"]

async def is_user_joined(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user_id = update.effective_user.id
    try:
        member = await context.bot.get_chat_member(chat_id=CHANNEL_USERNAME, user_id=user_id)
        return member.status in ['member', 'administrator', 'creator']
    except Exception:
        return False 

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_user_joined(update, context):
        kb = [[InlineKeyboardButton("👉 点击加入频道", url=f"https://t.me/{CHANNEL_USERNAME.replace('@', '')}")],
              [InlineKeyboardButton("✅ 我已加入，重新验证", callback_data="check_join")]]
        await update.message.reply_text("⚠️ 请先加入频道！\n加入后点击验证。", reply_markup=InlineKeyboardMarkup(kb))
        return

    kb = [
        [InlineKeyboardButton("🎯 特码 Top-6", callback_data="tema6")],
        [InlineKeyboardButton("🎯 特码 Top-3", callback_data="tema3")],
        [InlineKeyboardButton("🔪 双组/杀组", callback_data="dual")],
        [InlineKeyboardButton("📉 连挂分析", callback_data="streak")],
    ]
    await update.message.reply_text("请选择分析类型：", reply_markup=InlineKeyboardMarkup(kb))

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try: await q.answer()
    except: pass

    if q.data == "check_join":
        if await is_user_joined(update, context):
            kb = [
                [InlineKeyboardButton("🎯 特码 Top-6", callback_data="tema6")],
                [InlineKeyboardButton("🎯 特码 Top-3", callback_data="tema3")],
                [InlineKeyboardButton("🔪 双组/杀组", callback_data="dual")],
                [InlineKeyboardButton("📉 连挂分析", callback_data="streak")],
            ]
            await q.edit_message_text("✅ 验证成功！请选择：", reply_markup=InlineKeyboardMarkup(kb))
        else:
            try: await q.answer("❌ 验证失败，请先加入频道！", show_alert=True)
            except: pass
        return

    if not await is_user_joined(update, context):
        await q.edit_message_text("⚠️ 请先加入频道后再使用！发送 /start 重新验证。")
        return

    await q.edit_message_text("⚡ 极速计算中...")
    h = await get_history()
    
    if not h or len(h) < 5:
        await q.edit_message_text("❌ 数据拉取失败或期数不足，请稍后重试")
        return

    latest = h[-1]
    header = f"最新期 {latest[0]}\n{latest[1]}+{latest[2]}+{latest[3]}={latest[4]}（{classify(latest[4])}）\n\n"

    if q.data == "tema6":
        ranked = fuse_tema(h)
        text = header + "🎯 特码 Top-6\n\n"
        for i, (v, p) in enumerate(ranked[:6], 1): text += f"{i}. 特码 {v}（{classify(v)}）\n"
    elif q.data == "tema3":
        ranked = fuse_tema(h)
        text = header + "🎯 特码 Top-3\n\n"
        for i, (v, p) in enumerate(ranked[:3], 1): text += f"{i}. 特码 {v}（{classify(v)}）\n"
    elif q.data == "dual":
        dg = dual_group(h)
        text = header + f"🔪 双组分析\n\n维度：{dg['dim']}\n主推：{dg['main']}\n防守：{dg['guard']}\n\n理论基准：50%"
    elif q.data == "streak":
        s1 = miss_streak(h, top_n=1)
        text = header + f"📉 连挂分析\n\nTop-1：连挂 {s1['streak']} 期\n随机下出现概率 {s1['prob']*100:.1f}%"
    else: text = "未知操作"

    text += "\n\n⚠️ 基准不变，算法不提高长期命中率"
    await q.edit_message_text(text)


# ══════════ 适配 Render 的假 Web 服务器 ══════════
class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running!")

def run_dummy_server():
    port = int(os.environ.get("PORT", "8080"))
    server = HTTPServer(('0.0.0.0', port), SimpleHandler)
    server.serve_forever()

# ══════════ 主入口 ══════════
def main():
    token = os.environ.get("BOT_TOKEN", "").strip()
    if not token:
        print("错误：未找到 BOT_TOKEN 环境变量！")
        return
        
    # 启动假 Web 服务器（在后台运行，防止 Render 报无端口错误）
    threading.Thread(target=run_dummy_server, daemon=True).start()
    
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(on_button))
    print("Bot 已启动！")
    
    # 启动 Telegram 机器人
    app.run_polling()

if __name__ == "__main__":
    main()
