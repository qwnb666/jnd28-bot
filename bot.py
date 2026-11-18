#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from collections import Counter, defaultdict

import httpx
from openai import AsyncOpenAI
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, ContextTypes
)

CHANNEL_USERNAME = "@qw888nb" 
ALL_TYPES = ["大单", "大双", "小单", "小双"]

AI_API_KEY = os.environ.get("AI_API_KEY", "").strip()
ai_client = AsyncOpenAI(api_key=AI_API_KEY, base_url="https://api.deepseek.com") if AI_API_KEY else None

async def fetch_history():
    urls = ["https://pc28.help/api/kj.json?nbr=100", "https://jnd28.pw/api/kj?limit=100"]
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

def classify(n):
    size = "大" if n >= 14 else "小"
    par = "单" if n % 2 else "双"
    return size + par

# ============================================================
# 🏆 5 大策略引擎 (用于投票)
# ============================================================
def model_trend(h): # 顺势长龙派
    recent = [classify(he) for _,_,_,_,he in h[-5:]]
    return recent[-1]

def model_reversion(h): # 均值回归派
    recent = [classify(he) for _,_,_,_,he in h[-10:]]
    cnt = Counter(recent)
    return cnt.most_common(1)[0][0]

def model_markov(h): # 马尔可夫转移派
    tr = defaultdict(Counter)
    for i in range(len(h)-1): tr[classify(h[i][4])][classify(h[i+1][4])] += 1
    last = classify(h[-1][4])
    row = tr.get(last, Counter())
    if not row: return classify(h[-1][4])
    return row.most_common(1)[0][0]

def model_bayes(h): # 朴素贝叶斯派
    c = Counter(classify(he) for _,_,_,_,he in h[-20:])
    return c.most_common(1)[0][0]

def model_alternate(h): # 极值反脆弱派 (连出4把变盘)
    recent = [classify(he) for _,_,_,_,he in h[-4:]]
    if len(set(recent)) == 1:
        return recent[-1] # 如果连出，投它一票（长龙顺热）
    return model_reversion(h)

# ============================================================
# ⚖️ 共识投票与熔断机制
# ============================================================
def get_consensus_prediction(h):
    models = {
        "顺势长龙": model_trend,
        "均值回归": model_reversion,
        "马氏转移": model_markov,
        "贝叶斯流": model_bayes,
        "极值反脆弱": model_alternate
    }
    
    # 收集所有模型想要 "杀" 的形态
    kill_votes = []
    model_picks = {}
    
    for name, func in models.items():
        try:
            kill_target = func(h)
            kill_votes.append(kill_target)
            model_picks[name] = kill_target
        except:
            pass

    # 统计票数
    vote_counter = Counter(kill_votes)
    most_common_target, votes = vote_counter.most_common(1)[0]

    # 判断共识程度
    if votes >= 4:
        consensus_level = "🔥 绝对共识 (4/5)"
        action = f"高胜率出击！建议主杀：**{most_common_target}**"
        main = [t for t in ALL_TYPES if t != most_common_target][0]
        is_chaos = False
    elif votes == 3:
        consensus_level = "⚖️ 多数共识 (3/5)"
        action = f"常规防守！建议主杀：**{most_common_target}**"
        main = [t for t in ALL_TYPES if t != most_common_target][0]
        is_chaos = False
    else:
        consensus_level = "🚨 意见分歧 (最大票数 < 3)"
        action = "**建议观望**：模型意见极度混乱，强行下注极易连黑！"
        most_common_target = "无"
        main = "无"
        is_chaos = True

    # 找防守形态
    guard = "无"
    if not is_chaos:
        remain = [t for t in ALL_TYPES if t != most_common_target]
        guard = remain[-1]

    # 精选两个特码（基于主推形态）
    temas = []
    if main != "无":
        def get_totals(p):
            if p == "大双": return [14, 16, 18, 20, 22, 24, 26]
            if p == "大单": return [15, 17, 19, 21, 23, 25, 27]
            if p == "小双": return [2, 4, 6, 8, 10, 12]
            if p == "小单": return [1, 3, 5, 7, 9, 11, 13]
            return []
        pool = get_totals(main)
        recent_sums = [he for _,_,_,_,he in h[-10:]]
        cold_pool = sorted(pool, key=lambda t: recent_sums.count(t))
        temas = sorted(cold_pool[:2])

    return {
        "models": model_picks,
        "votes": dict(vote_counter),
        "target": most_common_target,
        "level": consensus_level,
        "action": action,
        "main": main,
        "guard": guard,
        "temas": temas,
        "is_chaos": is_chaos
    }

async def get_ai_comment(consensus):
    if not ai_client: return "⚠️ AI 未配置，本地共识算法已执行。"
    
    prompt = f"""
    你是一个首席量化风控官。当前系统执行了“多模型共识投票机制”。
    5个独立算法的投票详情：{consensus['models']}。
    最终共识：{consensus['level']}。
    系统建议：{consensus['action']}。
    
    请用 2 句话点评此结果，语气要像华尔街风控官，强调纪律和概率，不超过 80 字。
    """
    try:
        response = await ai_client.chat.completions.create(
            model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.5, max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except:
        return "⚠️ AI 连接超时。"

# ============================================================
# Bot 逻辑
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

    kb = [[InlineKeyboardButton("⚖️ 启动 5 模型共识投票", callback_data="predict")]]
    await update.message.reply_text("欢迎使用共识投票系统。只有在多模型达成共识时，才会发出高胜率指令。", reply_markup=InlineKeyboardMarkup(kb))

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try: await q.answer()
    except: pass

    if q.data == "check_join":
        if await is_user_joined(update, context):
            kb = [[InlineKeyboardButton("⚖️ 启动 5 模型共识投票", callback_data="predict")]]
            await q.edit_message_text("✅ 验证成功！", reply_markup=InlineKeyboardMarkup(kb))
        else:
            try: await q.answer("❌ 验证失败，请先加入！", show_alert=True)
            except: pass
        return

    if not await is_user_joined(update, context):
        await q.edit_message_text("⚠️ 请先加入频道！")
        return

    if q.data == "predict":
        await q.edit_message_text("🧠 正在收集 5 大流派的独立预测，并启动共识熔断机制...")
        
        hist = await fetch_history()
        if not hist or len(hist) < 20:
            await q.edit_message_text("❌ 数据拉取失败或不足")
            return

        consensus = get_consensus_prediction(hist)
        ai_comment = await get_ai_comment(consensus)

        latest = hist[-1]
        text = f"🏛️ **情雾 10.0 · 共识熔断版**\n\n"
        text += f"🏰 最新：{latest[1]}+{latest[2]}+{latest[3]}={latest[4]} [{classify(latest[4])}]\n\n"
        
        # 投票面板
        text += f"📊 **【5 大流派独立投票】**\n"
        for name, target in consensus['models'].items():
            text += f"· {name}：杀 `{target}`\n"
        
        text += f"\n⚖️ **【投票结果统计】**\n"
        for target, v in consensus['votes'].items():
            text += f"`{target}` 获得了 {v} 票\n"
        
        text += f"\n🚦 **【系统决议：{consensus['level']}】**\n"
        text += f"{consensus['action']}\n\n"

        if not consensus['is_chaos']:
            text += f"🎯 **【交易指令】**\n"
            text += f"高概率杀组：`{consensus['target']}`\n"
            text += f"主推形态：`{consensus['main']}`\n"
            text += f"防守形态：`{consensus['guard']}`\n"
            
            if consensus['temas']:
                t_str = " / ".join([f"**{t}**({classify(t)})" for t in consensus['temas']])
                text += f"极冷特码：{t_str}\n"
        else:
            text += f"🛑 **【交易指令】**\n**空仓观望！不要下注！**\n"

        text += f"\n🧠 **【AI 风控点评】**\n"
        text += f"{ai_comment}\n"
        
        await q.edit_message_text(text, parse_mode="Markdown")

# ============================================================
# Web Server & Main
# ============================================================
class SimpleHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running!")

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
    print("共识熔断版 Bot 已启动！")
    app.run_polling()

if __name__ == "__main__":
    main()
