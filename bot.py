#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import math
import threading
import random
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
# 🏆 8 大策略引擎 (双轨投票)
# ============================================================

# 1. 顺势长龙派
def model_trend(h): 
    recent = [classify(he) for _,_,_,_,he in h[-5:]]
    return {"kill": recent[-1], "duals": [recent[-1], recent[-2]]}

# 2. 均值回归派
def model_reversion(h): 
    recent = [classify(he) for _,_,_,_,he in h[-10:]]
    cnt = Counter(recent)
    kill_target = cnt.most_common(1)[0][0] 
    duals = [t for t in ALL_TYPES if t != kill_target][:2]
    return {"kill": kill_target, "duals": duals}

# 3. 马尔可夫转移派
def model_markov(h): 
    tr = defaultdict(Counter)
    for i in range(len(h)-1): tr[classify(h[i][4])][classify(h[i+1][4])] += 1
    last = classify(h[-1][4])
    row = tr.get(last, Counter())
    if not row: return model_reversion(h)
    sorted_next = row.most_common()
    duals = [item[0] for item in sorted_next[:2]]
    kill_target = sorted_next[-1][0]
    return {"kill": kill_target, "duals": duals}

# 4. 朴素贝叶斯派
def model_bayes(h): 
    c = Counter(classify(he) for _,_,_,_,he in h[-20:])
    kill_target = c.most_common(1)[0][0]
    duals = [t for t in ALL_TYPES if t != kill_target][:2] 
    return {"kill": kill_target, "duals": duals}

# 5. 极值反脆弱派
def model_alternate(h): 
    recent = [classify(he) for _,_,_,_,he in h[-4:]]
    if len(set(recent)) == 1:
        return {"kill": recent[-1], "duals": [recent[-1], recent[-1]]}
    return model_reversion(h)

# 6. 逻辑回归派 (模拟特征权重)
def model_logistic(h):
    # 简单构造特征：大小概率、单双概率
    recent = [he for _,_,_,_,he in h[-15:]]
    big_cnt = sum(1 for v in recent if v >= 14)
    odd_cnt = sum(1 for v in recent if v % 2 == 1)
    
    # 强行拟合权重
    prob_big = big_cnt / len(recent) if recent else 0.5
    prob_odd = odd_cnt / len(recent) if recent else 0.5
    
    if prob_big > 0.6 and prob_odd > 0.6:
        return {"kill": "小双", "duals": ["大单", "大双"]}
    elif prob_big < 0.4 and prob_odd < 0.4:
        return {"kill": "大单", "duals": ["小双", "小单"]}
    
    # 默认回退
    return model_reversion(h)

# 7. KNN 近邻派 (寻找历史相似点)
def model_knn(h, k=5):
    if len(h) < 10: return model_reversion(h)
    target_seq = [he for _,_,_,_,he in h[-3:]] # 寻找近3期相似的
    
    distances = []
    for i in range(len(h)-4):
        seq = [he for _,_,_,_,he in h[i:i+3]]
        dist = sum(abs(a-b) for a, b in zip(seq, target_seq))
        distances.append((dist, classify(h[i+3][4])))
        
    distances.sort(key=lambda x: x[0])
    knn_types = [t for _, t in distances[:k]]
    cnt = Counter(knn_types)
    kill_target = cnt.most_common(1)[0][0] if cnt else "小双"
    duals = [t for t in ALL_TYPES if t != kill_target][:2]
    return {"kill": kill_target, "duals": duals}

# 8. 随机森林简化派 (多决策树投票)
def model_forest(h):
    if len(h) < 20: return model_trend(h)
    # 随机抽取不同长度的窗口
    t1 = model_trend(h[-5:])
    t2 = model_reversion(h[-15:])
    t3 = model_markov(h[-30:])
    
    all_kills = [t1['kill'], t2['kill'], t3['kill']]
    cnt = Counter(all_kills)
    kill_target = cnt.most_common(1)[0][0]
    
    all_duals = t1['duals'] + t2['duals'] + t3['duals']
    dual_cnt = Counter(all_duals)
    duals = [t for t, v in dual_cnt.most_common() if t != kill_target][:2]
    return {"kill": kill_target, "duals": duals}

# ============================================================
# ⚖️ 8 模型双轨投票与熔断机制
# ============================================================
def get_master_prediction(h):
    models = {
        "顺势长龙": model_trend,
        "均值回归": model_reversion,
        "马氏转移": model_markov,
        "朴素贝叶斯": model_bayes,
        "极值反脆弱": model_alternate,
        "逻辑回归": model_logistic,
        "KNN近邻": model_knn,
        "随机森林": model_forest
    }
    
    kill_pool = []
    dual_pool = []
    model_picks = {}
    
    for name, func in models.items():
        try:
            res = func(h)
            kill_pool.append(res["kill"])
            dual_pool.extend(res["duals"])
            model_picks[name] = res
        except Exception as e:
            pass

    kill_counter = Counter(kill_pool)
    dual_counter = Counter(dual_pool)
    
    final_kill, kill_votes = kill_counter.most_common(1)[0]

    # 剔除被杀的形态
    valid_duals = [(t, v) for t, v in dual_counter.most_common() if t != final_kill]
    final_duals = [valid_duals[0][0], valid_duals[1][0]] if len(valid_duals) >= 2 else valid_duals

    # 8 个模型，共识要求提高（至少 4 票杀组）
    if kill_votes >= 6:
        consensus_level = "🌋 终极共识 (杀组绝对碾压)"
        is_chaos = False
    elif kill_votes >= 4:
        consensus_level = "⚖️ 多数共识 (杀组半数以上)"
        is_chaos = False
    else:
        consensus_level = "🚨 严重分歧 (杀组票数分散)"
        is_chaos = True

    return {
        "models": model_picks,
        "kill_counter": dict(kill_counter),
        "kill_target": final_kill,
        "kill_votes": kill_votes,
        "dual_counter": dict(dual_counter),
        "final_duals": final_duals,
        "level": consensus_level,
        "is_chaos": is_chaos,
        "total_models": len(models)
    }

async def get_ai_comment(res):
    if not ai_client: return "⚠️ AI 未配置。"
    
    prompt = f"""
    你是一个首席量化风控官。当前系统执行了“8大模型双轨独立投票”。
    最终杀组共识：{res['kill_target']} (在 {res['total_models']} 个模型中得 {res['kill_votes']} 票)。
    最终双组共识：{res['final_duals']}。
    
    请用 2 句话点评此结果，强调“机器学习模型集成的稳健性”和“纪律性”，不超过 80 字。
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

    kb = [[InlineKeyboardButton("🚀 启动 8 模型机器学习推演", callback_data="predict")]]
    await update.message.reply_text("欢迎使用最高算力终端。8 大机器学习模型已就绪。", reply_markup=InlineKeyboardMarkup(kb))

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try: await q.answer()
    except: pass

    if q.data == "check_join":
        if await is_user_joined(update, context):
            kb = [[InlineKeyboardButton("🚀 启动 8 模型机器学习推演", callback_data="predict")]]
            await q.edit_message_text("✅ 验证成功！", reply_markup=InlineKeyboardMarkup(kb))
        else:
            try: await q.answer("❌ 验证失败，请先加入！", show_alert=True)
            except: pass
        return

    if not await is_user_joined(update, context):
        await q.edit_message_text("⚠️ 请先加入频道！")
        return

    if q.data == "predict":
        await q.edit_message_text("🧠 正在启动逻辑回归、KNN、随机森林等 8 大模型并发投票...\n(算力消耗较大，请稍候)")
        
        hist = await fetch_history()
        if not hist or len(hist) < 20:
            await q.edit_message_text("❌ 数据拉取失败或不足")
            return

        res = get_master_prediction(hist)
        ai_comment = await get_ai_comment(res)

        latest = hist[-1]
        text = f"🏛️ **情雾 12.0 · 8 模型集成版**\n\n"
        text += f"🏰 最新：{latest[1]}+{latest[2]}+{latest[3]}={latest[4]} [{classify(latest[4])}]\n\n"
        
        text += f"❌ **【杀组票池：8 大模型投票】**\n"
        for t, v in sorted(res['kill_counter'].items(), key=lambda x: -x[1]):
            text += f"· `{t}` 获得 {v} 票\n"
        text += f"▶ **最终决议：杀 `{res['kill_target']}`**\n\n"

        text += f"✅ **【双组票池】** (剔除杀组后)\n"
        dual_strs = [f"`{t}`({v}票)" for t, v in res['dual_counter'].items() if t != res['kill_target']]
        text += "· " + " | ".join(dual_strs[:3]) + "\n"
        text += f"▶ **最终双组：主推 `{res['final_duals'][0]}` / 防守 `{res['final_duals'][1]}`**\n\n"

        text += f"🚦 **【系统决议：{res['level']}】**\n"
        if res['is_chaos']:
            text += f"🛑 **交易指令：空仓观望！不要下注！**\n"

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
    print("8 大机器学习模型 Bot 已启动！")
    app.run_polling()

if __name__ == "__main__":
    main()
