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

# ⚠️ 你的频道
CHANNEL_USERNAME = "@qw888nb" 
ALL_TYPES = ["大单", "大双", "小单", "小双"]
N_STATES = 4

# 🤖 初始化 AI 客户端
AI_API_KEY = os.environ.get("AI_API_KEY", "").strip()
ai_client = AsyncOpenAI(api_key=AI_API_KEY, base_url="https://api.deepseek.com") if AI_API_KEY else None

# ============================================================
#  0-27 理论概率分布
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

def get_pattern_totals(pattern: str) -> list:
    if pattern == "大双": return [14, 16, 18, 20, 22, 24, 26]
    if pattern == "大单": return [15, 17, 19, 21, 23, 25, 27]
    if pattern == "小双": return [2, 4, 6, 8, 10, 12]
    if pattern == "小单": return [1, 3, 5, 7, 9, 11, 13]
    return list(range(28))

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
#  形态估计器 (双组/杀组)
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
#  🚀 顶级量化特码算法 (融合 7 大高级数理统计模型)
# ============================================================
def get_top4_advanced(h, allowed_patterns):
    n = len(h)
    if n < 20: return [14, 15, 13, 16]
    
    vals = [he for _,_,_,_,he in h]
    
    # 1. KL 散度计算 (Kullback-Leibler Divergence)
    # 衡量最近20期分布与理论分布的差异
    recent_20 = vals[-20:]
    recent_freq = Counter(recent_20)
    kl_div = 0.0
    for t in range(28):
        p = (recent_freq.get(t, 0) + 1) / 21.0
        q = THEORY[t] + 1e-6
        kl_div += p * math.log(p / q)
    # 差异越小，说明越接近随机；差异越大，说明近期走势越极端
    # 我们利用 KL 散度动态分配“趋势跟踪”和“均值回归”的权重
    weight_trend = min(max(kl_div / 2.0, 0.2), 0.8) 
    
    # 2. Z-Score 均值回归 (Mean Reversion)
    mean_val = sum(vals) / n
    variance = sum((x - mean_val) ** 2 for x in vals) / n
    std_dev = math.sqrt(variance) if variance > 0 else 1.0
    z_scores = {t: (t - mean_val) / std_dev for t in range(28)}
    
    # 3. AR(1) 自回归时间序列
    # X_t = c + phi * X_{t-1} + epsilon
    ar_pred = mean_val
    if n >= 2:
        x_prev = vals[-1]
        numerator = sum((vals[i] - mean_val) * (vals[i-1] - mean_val) for i in range(1, n))
        denominator = sum((vals[i-1] - mean_val) ** 2 for i in range(1, n))
        phi = numerator / denominator if denominator != 0 else 0.5
        ar_pred = mean_val + phi * (x_prev - mean_val)

    # 4. 马尔可夫转移矩阵 (Markov Chain)
    tr = defaultdict(Counter)
    for i in range(n-1): tr[vals[i]][vals[i+1]] += 1
    last_he = vals[-1]
    row = tr.get(last_he, Counter())
    total_tr = sum(row.values())
    
    # 5. MCMC (马尔可夫链蒙特卡洛) 采样验证
    # 基于转移矩阵，抽样 1000 次，统计频率
    mcmc_counts = Counter()
    current_state = last_he
    for _ in range(1000):
        possible_nexts = list(row.keys())
        if possible_nexts:
            weights = [row[k] for k in possible_nexts]
            current_state = random.choices(possible_nexts, weights=weights, k=1)[0]
        else:
            current_state = random.randint(0, 27)
        mcmc_counts[current_state] += 1

    # 6. 贝叶斯特征交叉 (Naive Bayes 特征工程)
    # 特征：大小 (0/1), 单双 (0/1), 尾数质合 (0/1)
    def get_feat(n):
        return (1 if n >= 14 else 0, 1 if n % 2 else 0, 1 if n % 10 in [2,3,5,7] else 0)
    feat_counts = Counter(get_feat(v) for v in vals)
    
    # ================= 融合计算 =================
    fused = {}
    for t in range(28):
        p_theory = THEORY[t]
        
        # 马尔可夫平滑概率
        p_markov = (row.get(t, 0) + 2 * p_theory) / (total_tr + 2)
        
        # MCMC 概率
        p_mcmc = (mcmc_counts.get(t, 0) + 5) / 1005.0
        
        # Z-score 概率 (距离均值越远，在特定趋势下回补概率越大)
        # 如果趋势权重高，我们倾向于跟踪中位数；如果权重低，我们倾向于均值回归
        z_score_factor = math.exp(-0.5 * (z_scores[t] ** 2)) 
        
        # AR(1) 距离衰减
        ar_factor = 1.0 / (1.0 + abs(t - ar_pred))
        
        # 朴素贝叶斯特征概率
        tf = get_feat(t)
        p_nb = 1.0
        for i in range(3):
            feat_val = tf[i]
            count = sum(1 for v in vals if get_feat(v)[i] == feat_val)
            p_nb *= (count + 1) / (n + 2) # 拉普拉斯平滑
            
        # 综合评分
        score = (
            p_theory * (1 - weight_trend) + 
            p_markov * weight_trend * 1.5 +
            p_mcmc * weight_trend * 1.5 +
            z_score_factor * 0.2 +
            ar_factor * 0.2 +
            p_nb * 0.1
        )
        
        fused[t] = score

    # 7. 形态共振强制过滤
    allowed_totals = set()
    for pat in allowed_patterns:
        allowed_totals.update(get_pattern_totals(pat))
        
    filtered = [(t, p) for t, p in fused.items() if t in allowed_totals]
    filtered.sort(key=lambda x: x[1], reverse=True)
    
    if len(filtered) < 4:
        all_sorted = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        for t, p in all_sorted:
            if t not in [x[0] for x in filtered]:
                filtered.append((t, p))
            if len(filtered) == 4: break

    return [t for t, p in filtered[:4]]

# ============================================================
#  AI 深度研判 (极致装逼版)
# ============================================================
async def get_ai_analysis(probs, top4_totals, latest_draw, main_p, guard_p):
    if not ai_client: return "⚠️ 未配置 AI_API_KEY，已使用本地模型。"
    
    p_text = "\n".join([f"- {k}: {v*100:.2f}%" for k, v in probs.items()])
    recent_draws = "\n".join([f"期号 {d[0]}: {d[1]}+{d[2]}+{d[3]}={d[4]} [{classify(d[4])}]" for d in latest_draw[-5:]])

    prompt = f"""
    你是一个顶级量化博彩分析师，精通高级数理统计与机器学习。
    请根据数据生成极简的【AI 深度量化研判报告】。

    【最新开奖】: {latest_draw[-1][1]}+{latest_draw[-1][2]}+{latest_draw[-1][3]}={latest_draw[-1][4]}
    【近 5 期走势】:
    {recent_draws}
    【形态融合概率】:
    {p_text}
    【主推形态】: {main_p}，【防守形态】: {guard_p}
    【七模型融合特码 4 码】: {top4_totals}

    【要求】:
    1. 用 3-4 句话总结盘面。必须强行穿插以下至少 3 个术语：“KL散度动态调权”、“AR(1)自回归”、“Z-Score均值回归”、“MCMC抽样验证”、“朴素贝叶斯特征交叉”。
    2. 解释为什么推荐这 4 个特码。
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
        return "⚠️ AI 连接超时，使用七维数理模型研判完成。"

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

    kb = [[InlineKeyboardButton("🔬 启动七维数理模型推演", callback_data="predict")]]
    await update.message.reply_text("欢迎使用顶级量化终端。点击按钮启动多维特征工程推演。", reply_markup=InlineKeyboardMarkup(kb))

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try: await q.answer()
    except: pass

    if q.data == "check_join":
        if await is_user_joined(update, context):
            kb = [[InlineKeyboardButton("🔬 启动七维数理模型推演", callback_data="predict")]]
            await q.edit_message_text("✅ 验证成功！", reply_markup=InlineKeyboardMarkup(kb))
        else:
            try: await q.answer("❌ 验证失败，请先加入！", show_alert=True)
            except: pass
        return

    if not await is_user_joined(update, context):
        await q.edit_message_text("⚠️ 请先加入频道！")
        return

    if q.data == "predict":
        await q.edit_message_text("🧠 正在执行 MCMC 抽样与 KL 散度计算...\n(算力消耗较大，请等待约 5 秒)")
        
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

        # 特码预测 (七维高级公式 4码)
        top4_totals = get_top4_advanced(hist, [main_type, guard_type])

        # AI 分析
        ai_report = await get_ai_analysis(fused, top4_totals, hist, main_type, guard_type)

        latest = hist[-1]
        text = f"📡 **顶级量化 AI 终端已上线**\n\n"
        text += f"🏰 最新：{latest[1]}+{latest[2]}+{latest[3]}={latest[4]} [{classify(latest[4])}]\n\n"
        
        text += f"🧠 **【AI 深度数理研判】**\n"
        text += f"{ai_report}\n\n"

        text += f"🎯 **【形态预测】**\n"
        text += f"主推：{main_type} ({p_main*100:.1f}%)\n"
        text += f"防守：{guard_type} ({p_guard*100:.1f}%)\n"
        text += f"冷杀：{kill_type} ({p_kill*100:.1f}%)\n\n"

        text += f"🔮 **【七维数理共振 4 特码】**\n"
        t_str = " / ".join([f"**{t}**({classify(t)})" for t in top4_totals])
        text += f"{t_str}\n"
        text += f"*(MCMC+Z-Score+AR自回归+朴素贝叶斯融合)*\n\n"
        
        text += f"📊 *数据由复杂数理统计与大语言模型联合生成。*\n*理性对待随机事件。*"
        
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
    print("七维数理量化 Bot 已启动！")
    app.run_polling()

if __name__ == "__main__":
    main()
