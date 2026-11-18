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

# ⚠️ 你的频道
CHANNEL_USERNAME = "@qw888nb" 
ALL_TYPES = ["大单", "大双", "小单", "小双"]

# 🤖 初始化 AI 客户端
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
#  🛡️ 核心策略：降维防反猎杀模型
# ============================================================
def get_strategy(h):
    if len(h) < 20:
        return {"error": "历史数据不足，无法执行降维策略"}

    # 近 10 期和 30 期形态统计
    recent_10 = [classify(he) for _,_,_,_,he in h[-10:]]
    recent_30 = [classify(he) for _,_,_,_,he in h[-30:]]
    cnt_10 = Counter(recent_10)
    cnt_30 = Counter(recent_30)
    
    last_type = recent_10[-1]
    
    # 1. 检测长龙（连号）
    streak = 0
    for t in reversed(recent_10):
        if t == last_type:
            streak += 1
        else:
            break
            
    # 2. 寻找“极热”与“极冷”
    hottest_30 = cnt_30.most_common(1)[0][0]
    coldest_30 = min(ALL_TYPES, key=lambda t: cnt_30.get(t, 0))

    # 3. 制定杀组策略
    # 逻辑 A：如果连出 >= 4 把，直接杀另外两个形态（防长龙断点），绝不抄底。
    # 逻辑 B：如果没有长龙，顺着 30 期最热的杀（顺热策略），或者杀最冷的（均值回归）。
    if streak >= 4:
        kill_patterns = [t for t in ALL_TYPES if t != last_type and t != recent_10[-2]]
        strategy_desc = f"【长龙预警】已连出 {streak} 把 {last_type}，模型强制顺热，杀无关形态。"
    else:
        # 杀掉最热门的那个形态（因为热门形态被过度消耗）
        kill_target = hottest_30
        kill_patterns = [kill_target]
        strategy_desc = f"【均值回归】近期 {kill_target} 出现过于频繁，模型判定其将被冷却。"

    # 4. 制定双组策略（从剩下的里面挑，避开连号和极冷）
    remain_types = [t for t in ALL_TYPES if t not in kill_patterns]
    
    # 选出主推（近期偏热，但不是最热）和防守（近期偏冷，准备反弹）
    remain_types.sort(key=lambda t: cnt_10.get(t, 0), reverse=True)
    main_type = remain_types[0]
    guard_type = remain_types[-1] if len(remain_types) > 1 else remain_types[0]

    # 5. 特码精选（只出 2 个，降低难度）
    # 在主推形态里，找前 30 期最冷的两个和值
    def get_pattern_totals(pattern):
        if pattern == "大双": return [14, 16, 18, 20, 22, 24, 26]
        if pattern == "大单": return [15, 17, 19, 21, 23, 25, 27]
        if pattern == "小双": return [2, 4, 6, 8, 10, 12]
        if pattern == "小单": return [1, 3, 5, 7, 9, 11, 13]
        return []
        
    pool = get_pattern_totals(main_type)
    recent_30_sums = [he for _,_,_,_,he in h[-30:]]
    cold_pool = sorted(pool, key=lambda t: recent_30_sums.count(t))
    top2_totals = sorted(cold_pool[:2]) # 拿最冷的两个

    return {
        "streak": streak,
        "kill": kill_patterns,
        "main": main_type,
        "guard": guard_type,
        "totals": top2_totals,
        "desc": strategy_desc,
        "cnt_30": dict(cnt_30)
    }

async def get_ai_comment(strategy):
    if not ai_client: return "⚠️ 未配置 AI_API_KEY，本地算法已执行。"
    prompt = f"""
    你是一个实战派量化博彩分析师。当前模型执行了“降维防守策略”。
    当前盘面：近30期形态频次统计 {strategy['cnt_30']}。
    当前连号状态：连出 {strategy['streak']} 把。
    策略动作：杀 {strategy['kill']}，主推 {strategy['main']}，防守 {strategy['guard']}。
    精选特码：{strategy['totals']}。
    
    请用 3 句话点评这个策略逻辑，语言要犀利、冷静，强调“防守大于预测”，结尾附带风险提示。不超过 100 字。
    """
    try:
        response = await ai_client.chat.completions.create(
            model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except:
        return "⚠️ AI 连接超时。"

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

    kb = [[InlineKeyboardButton("🛡️ 获取降维防守策略", callback_data="predict")]]
    await update.message.reply_text("请选择操作。当前为【降维防反猎杀版】，主打高胜率防守。", reply_markup=InlineKeyboardMarkup(kb))

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try: await q.answer()
    except: pass

    if q.data == "check_join":
        if await is_user_joined(update, context):
            kb = [[InlineKeyboardButton("🛡️ 获取降维防守策略", callback_data="predict")]]
            await q.edit_message_text("✅ 验证成功！", reply_markup=InlineKeyboardMarkup(kb))
        else:
            try: await q.answer("❌ 验证失败，请先加入！", show_alert=True)
            except: pass
        return

    if not await is_user_joined(update, context):
        await q.edit_message_text("⚠️ 请先加入频道！")
        return

    if q.data == "predict":
        await q.edit_message_text("⚡ 正在分析长龙与形态频次...")
        
        hist = await fetch_history()
        if not hist or len(hist) < 20:
            await q.edit_message_text("❌ 数据拉取失败或不足 20 期，请稍后重试")
            return

        strategy = get_strategy(hist)
        if "error" in strategy:
            await q.edit_message_text(f"❌ {strategy['error']}")
            return

        ai_comment = await get_ai_comment(strategy)

        latest = hist[-1]
        text = f"📡 **情雾 · 降维防反猎杀版**\n\n"
        text += f"🏰 最新：{latest[1]}+{latest[2]}+{latest[3]}={latest[4]} [{classify(latest[4])}]\n"
        text += f"📊 走势判断：{strategy['desc']}\n\n"
        
        text += f"🧠 **【AI 策略点评】**\n"
        text += f"{ai_comment}\n\n"

        kill_str = " / ".join(strategy['kill'])
        text += f"❌ **【杀组区间】**\n必杀：`{kill_str}` *(高胜率防守)*\n\n"

        text += f"🎯 **【双组防守】**\n"
        text += f"主推：`{strategy['main']}`\n"
        text += f"防守：`{strategy['guard']}`\n\n"

        text += f"🔮 **【精选双特码】**\n"
        t_str = " / ".join([f"**{t}**({classify(t)})" for t in strategy['totals']])
        text += f"{t_str}\n"
        text += f"*(从主推形态中精选极冷号)*\n\n"
        
        text += f"⚠️ *策略仅为防守参考。如果模型连黑，请立即停手观望！*"
        
        await q.edit_message_text(text, parse_mode="Markdown")

# ============================================================
#  假 Web 服务器防休眠 + 主入口
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
    print("降维防反猎杀版 Bot 已启动！")
    app.run_polling()

if __name__ == "__main__":
    main()
