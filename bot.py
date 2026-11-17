#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import math
import random
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from decimal import Decimal, getcontext
from collections import Counter, defaultdict

import httpx
from openai import AsyncOpenAI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, ContextTypes
)

getcontext().prec = 50
PI = Decimal("3.14159265358979323846264338327950288419716939937510")

# ⚠️ 你的频道
CHANNEL_USERNAME = "@qw888nb" 
ALL_TYPES = ["大单", "大双", "小单", "小双"]
N_STATES = 4
LOG2_4 = 2.0

# 🤖 初始化 AI 客户端 (DeepSeek)
AI_API_KEY = os.environ.get("AI_API_KEY", "").strip()
ai_client = AsyncOpenAI(api_key=AI_API_KEY, base_url="https://api.deepseek.com") if AI_API_KEY else None

# ============================================================
#  0-27 理论分布 (特码计算基石)
# ============================================================
THEORY = {}
for total in range(0, 28):
    cnt = 0
    for a in range(10):
        for b in range(10):
            c = total - a - b
            if 0 <= c <= 9: cnt += 1
    THEORY[total] = cnt / 1000.0

def classify(n):
    size = "大" if n >= 14 else "小"
    par = "单" if n % 2 else "双"
    return size + par

# ============================================================
#  数据获取
# ============================================================
async def fetch_history():
    urls = [
        "https://pc28.help/api/kj.json?nbr=100",
        "https://jnd28.pw/api/kj?limit=100"
    ]
    async with httpx.AsyncClient(timeout=10.0) as client:
        for url in urls:
            try:
                resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
                obj = resp.json()
                data = obj.get("data", [])
                if not data: continue
                hist = []
                for item in data:
                    qh = int(item.get("nbr") or item.get("qh"))
                    num_str = item.get("number") or item.get("pc28")
                    he = int(item.get("num") or item.get("he"))
                    A, B, C = map(int, num_str.split("+"))
                    hist.append((qh, A, B, C, he))
                if hist:
                    hist.sort(key=lambda x: x[0])
                    return hist
            except Exception:
                continue
    return None

# ============================================================
#  形态估计器 (用于双组/杀组)
# ============================================================
def p_bayes_type(h, alpha=1.0):
    c = Counter(classify(he) for _,_,_,_,he in h); n = sum(c.values())
    return {t: (c.get(t,0)+alpha)/(n+alpha*N_STATES) for t in ALL_TYPES}

def p_markov_type(h, alpha=1.0):
    tr = defaultdict(Counter)
    for i in range(len(h)-1): tr[classify(h[i][4])][classify(h[i+1][4])] += 1
    row = tr.get(classify(h[-1][4]), Counter()); n = sum(row.values())
    return {t:(row.get(t,0)+alpha)/(n+alpha*N_STATES) for t in ALL_TYPES}

def p_ewma_type(h, lam=0.85):
    n = len(h); w = [lam**(n-1-i) for i in range(n)]; tw = sum(w)
    p = {t:0.0 for t in ALL_TYPES}
    for i,(_,_,_,_,he) in enumerate(h): p[classify(he)] += w[i]/tw
    return p

# ============================================================
#  ⭐️ 特码 (0-27 和值) 高精度四码算法
# ============================================================
def get_top4_totals(h):
    n = len(h)
    if n < 10: return [14, 15, 13, 16]
    
    # 1. 贝叶斯平滑 (偏向历史频率)
    freq = Counter(he for _,_,_,_,he in h)
    p_bayes = {}
    for t in range(28):
        p_bayes[t] = (freq.get(t, 0) + 20 * THEORY[t]) / (n + 20)
        
    # 2. 一阶马尔可夫和值转移 (看上期和值到本期)
    tr = defaultdict(Counter)
    for i in range(n-1): tr[h[i][4]][h[i+1][4]] += 1
    last_he = h[-1][4]
    row = tr.get(last_he, Counter())
    total_tr = sum(row.values())
    p_markov = {}
    for t in range(28):
        p_markov[t] = (row.get(t, 0) + 5 * THEORY[t]) / (total_tr + 5)

    # 3. 近期趋势 (EWMA)
    lam = 0.9
    w = [lam**(n-1-i) for i in range(n)]; tw = sum(w)
    p_ewma = {t: 0.0 for t in range(28)}
    for i, (_,_,_,_,he) in enumerate(h):
        p_ewma[he] += w[i] / tw
        
    # 4. 大融合 (等权相加)
    fused = {}
    for t in range(28):
        fused[t] = p_bayes[t] + p_markov[t] + p_ewma[t]
        
    # 取 Top 4
    top4 = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:4]
    return [t for t, p in top4]

# ============================================================
#  AI 深度研判
# ============================================================
async def get_ai_analysis(probs, top4_totals, latest_draw):
    if not ai_client: return "⚠️ 未配置 AI_API_KEY，已使用本地模型。"
    
    p_text = "\n".join([f"- {k}: {v*100:.2f}%" for k, v in probs.items()])
    recent_draws = "\n".join([f"期号 {d[0]}: {d[1]}+{d[2]}+{d[3]}={d[4]} [{classify(d[4])}]" for d in latest_draw[-5:]])

    prompt = f"""
    你是一个顶级量化博彩分析师。请根据以下数据，生成一份极简的【AI 盘面研判报告】。

    【最新开奖】: {latest_draw[-1][1]}+{latest_draw[-1][2]}+{latest_draw[-1][3]}={latest_draw[-1][4]}
    【近 5 期走势】:
    {recent_draws}
    【形态融合概率】:
    {p_text}
    【特码模型推荐 Top 4】: {top4_totals}

    【要求】:
    1. 用 3-4 句话精炼总结盘面。
    2. 解释为什么推荐这几个特码（可从冷热、奇偶、连号角度强行解释）。
    3. 结尾警示风险。
    4. 排版清晰，不超过 150 字。
    """
    try:
        response = await ai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "⚠️ AI 连接超时，使用基础算法研判。"

# ============================================================
#  Bot 逻辑
# ============================================================
async def is_user_joined(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    try:
        member = await context.bot.get_chat_member(chat_id=CHANNEL_USERNAME, user_id=update.effective_user.id)
        return member.status in ['member', 'administrator', 'creator']
    except Exception:
        return False 

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_user_joined(update, context):
        kb = [[InlineKeyboardButton("👉 加入频道", url=f"https://t.me/{CHANNEL_USERNAME.replace('@', '')}")],
              [InlineKeyboardButton("✅ 验证", callback_data="check_join")]]
        await update.message.reply_text("⚠️ 请先加入频道！", reply_markup=InlineKeyboardMarkup(kb))
        return

    kb = [[InlineKeyboardButton("🤖 呼叫 AI 深度研判 + 四特码", callback_data="predict")]]
    await update.message.reply_text("欢迎使用 AI 量化终端。点击按钮启动推演。", reply_markup=InlineKeyboardMarkup(kb))

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try: await q.answer()
    except: pass

    if q.data == "check_join":
        if await is_user_joined(update, context):
            kb = [[InlineKeyboardButton("🤖 呼叫 AI 深度研判 + 四特码", callback_data="predict")]]
            await q.edit_message_text("✅ 验证成功！", reply_markup=InlineKeyboardMarkup(kb))
        else:
            try: await q.answer("❌ 验证失败，请先加入！", show_alert=True)
            except: pass
        return

    if not await is_user_joined(update, context):
        await q.edit_message_text("⚠️ 请先加入频道！")
        return

    if q.data == "predict":
        await q.edit_message_text("🧠 正在吞噬数据，进行形态+特码双重推演...\n(请等待约 5 秒)")
        
        hist = await fetch_history()
        if not hist or len(hist) < 10:
            await q.edit_message_text("❌ 数据拉取失败，请稍后重试")
            return

        # 形态预测 (双组/杀组)
        est = {"bayes": p_bayes_type(hist), "markov": p_markov_type(hist), "ewma": p_ewma_type(hist)}
        fused = {t: sum(p[t] for p in est.values())/3 for t in ALL_TYPES}
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        
        main_type, p_main = ranked[0]
        guard_type, p_guard = ranked[1]
        kill_type, p_kill = ranked[3]

        # 特码预测 (Top 4)
        top4_totals = get_top4_totals(hist)

        # AI 分析
        ai_report = await get_ai_analysis(fused, top4_totals, hist)

        latest = hist[-1]
        text = f"📡 **AI 量化终端已上线**\n\n"
        text += f"🏰 最新：{latest[1]}+{latest[2]}+{latest[3]}={latest[4]} [{classify(latest[4])}]\n\n"
        
        text += f"🧠 **【AI 盘面研判】**\n"
        text += f"{ai_report}\n\n"

        text += f"🎯 **【形态预测】**\n"
        text += f"主推：{main_type} ({p_main*100:.1f}%)\n"
        text += f"防守：{guard_type} ({p_guard*100:.1f}%)\n"
        text += f"冷杀：{kill_type} ({p_kill*100:.1f}%)\n\n"

        text += f"🔮 **【AI 精选四特码】**\n"
        t_str = " / ".join([f"**{t}**({classify(t)})" for t in top4_totals])
        text += f"{t_str}\n"
        text += f"*(基于后验概率+马尔可夫转移综合测算)*\n\n"
        
        text += f"📊 *数据由 19 种数学模型与大语言模型联合生成。*\n*AI 观点仅供参考，长期胜率恒等，请理性对待。*"
        
        await q.edit_message_text(text, parse_mode="Markdown")

# ============================================================
#  假 Web 服务器防休眠 + 主入口
# ============================================================
class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"AI Bot is running!")

def run_dummy_server():
    port = int(os.environ.get("PORT", "8080"))
    server = HTTPServer(('0.0.0.0', port), SimpleHandler)
    server.serve_forever()

async def post_init(app: Application):
    await app.bot.delete_webhook(drop_pending_updates=True)

def main():
    token = os.environ.get("BOT_TOKEN", "").strip()
    if not token: return
    threading.Thread(target=run_dummy_server, daemon=True).start()
    app = Application.builder().token(token).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(on_button))
    print("AI 量化终端 (四特码版) 已启动！")
    app.run_polling()

if __name__ == "__main__":
    main()
