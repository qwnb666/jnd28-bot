#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from collections import Counter
from typing import List, Dict, Optional

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, ContextTypes
)

# ==================== 配置 ====================
API_URL = "https://pc28.help/api/kj.json?nbr=300"
HISTORY_LEN = 300
BACKTEST_N = 100
ALPHA = 20.0

# ⚠️ 替换成你的频道用户名（带@） ⚠️
CHANNEL_USERNAME = "@qw888nb" 

# ============================================================
#  理论分布
# ============================================================
def build_theoretical_dist() -> Dict[int, float]:
    dist = {}
    for total in range(0, 28):
        cnt = 0
        for a in range(10):
            for b in range(10):
                c = total - a - b
                if 0 <= c <= 9:
                    cnt += 1
        dist[total] = cnt / 1000.0
    return dist

THEORY = build_theoretical_dist()

# ============================================================
#  单期开奖数据类
# ============================================================
class Draw:
    def __init__(self, period: int, nums: List[int], total: int,
                 pattern: str, edge: str = "中", remainder: str = "", shape: str = "杂六"):
        self.period = period
        self.a, self.b, self.c = nums[0], nums[1], nums[2]
        self.nums = nums
        self.total = total
        self.pattern = pattern
        self.edge = edge
        self.remainder = remainder
        self.shape = shape

# ============================================================
#  工具函数
# ============================================================
def detect_shape(nums: List[int]) -> str:
    a, b, c = nums[0], nums[1], nums[2]
    if a == b == c: return '豹子'
    if a == b or b == c or a == c: return '对子'
    s = sorted(nums)
    if s[2] - s[0] == 2 and len(set(s)) == 3: return '顺子'
    return '杂六'

def get_pattern_by_sum(s: int) -> str:
    if s <= 13: return "小单" if s % 2 == 1 else "小双"
    else: return "大单" if s % 2 == 1 else "大双"

def get_pattern_totals(pattern: str) -> List[int]:
    if pattern == "大双": return [14, 16, 18, 20, 22, 24, 26]
    elif pattern == "大单": return [15, 17, 19, 21, 23, 25, 27]
    elif pattern == "小双": return [2, 4, 6, 8, 10, 12]
    elif pattern == "小单": return [1, 3, 5, 7, 9, 11, 13]
    return [10]

# ============================================================
#  V4.4 预测器
# ============================================================
class V44Predictor:
    def __init__(self, history: List[Draw]):
        self.history = history

    def _bayesian_weights(self, lookback: Optional[int] = None) -> Dict[int, float]:
        if lookback is None: lookback = min(100, len(self.history))
        recent = [r.total for r in self.history[-lookback:]]
        n = len(recent)
        if n == 0: return dict(THEORY)
        freq = Counter(recent)
        posterior = {}
        for t in range(1, 28):
            theo = THEORY.get(t, 0)
            posterior[t] = (freq.get(t, 0) + ALPHA * theo) / (n + ALPHA)
        return posterior

    def _pattern_conditional(self) -> str:
        pattern_theory = {
            "大双": sum(THEORY[t] for t in [14, 16, 18, 20, 22, 24, 26]),
            "大单": sum(THEORY[t] for t in [15, 17, 19, 21, 23, 25, 27]),
            "小双": sum(THEORY[t] for t in [2, 4, 6, 8, 10, 12]),
            "小单": sum(THEORY[t] for t in [1, 3, 5, 7, 9, 11, 13]),
        }
        scores = dict(pattern_theory)
        recent_10 = [r.pattern for r in self.history[-10:]]
        recent_30 = [r.pattern for r in self.history[-30:]]
        cnt10 = Counter(recent_10)
        cnt30 = Counter(recent_30)
        for p in scores:
            obs10 = cnt10.get(p, 0) / len(recent_10) if recent_10 else 0
            obs30 = cnt30.get(p, 0) / len(recent_30) if recent_30 else 0
            scores[p] = 0.4 * pattern_theory[p] + 0.3 * obs30 + 0.3 * obs10

        sizes = [r.pattern[0] for r in self.history[-4:]]
        odds = [r.pattern[1] for r in self.history[-4:]]
        size_streak = len(sizes) >= 4 and len(set(sizes)) == 1
        odd_streak = len(odds) >= 4 and len(set(odds)) == 1
        if size_streak:
            for p in scores:
                if p[0] == sizes[-1]: scores[p] *= 0.85
                else: scores[p] *= 1.08
        if odd_streak:
            for p in scores:
                if p[1] == odds[-1]: scores[p] *= 0.95
                else: scores[p] *= 1.05

        recent_30_sizes = [r.pattern[0] for r in self.history[-30:]]
        if recent_30_sizes:
            big_ratio = recent_30_sizes.count('大') / len(recent_30_sizes)
            if big_ratio > 0.6:
                for p in scores:
                    if p[0] == '大': scores[p] *= 0.85
                    else: scores[p] *= 1.1
            elif big_ratio < 0.4:
                for p in scores:
                    if p[0] == '小': scores[p] *= 0.85
                    else: scores[p] *= 1.1

        last_shape = self.history[-1].shape
        if last_shape == '豹子':
            for p in scores:
                if p[0] == '小': scores[p] *= 1.2
        elif last_shape == '顺子':
            for p in scores:
                if p[0] == '小': scores[p] *= 1.1
        return max(scores, key=scores.get)

    def predict(self) -> Dict:
        if len(self.history) < 5:
            return {'total': 14, 'pattern': '大双', 'top5_totals': [13, 14, 15]}
        posterior = self._bayesian_weights()
        main_pattern = self._pattern_conditional()
        candidates = get_pattern_totals(main_pattern)
        best_total = max(candidates, key=lambda t: posterior.get(t, 0))
        top5 = sorted(posterior.items(), key=lambda x: -x[1])[:5]
        top5_totals = [t for t, _ in top5]
        return {
            'total': best_total, 'pattern': main_pattern,
            'top5_totals': top5_totals, 'posterior': {t: round(posterior[t], 4) for t in range(1, 28)}
        }

# ============================================================
#  特码选择 + 双组
# ============================================================
def pick_two_totals(pattern: str, main_total: int, posterior: Dict[int, float]) -> List[int]:
    candidates = get_pattern_totals(pattern)
    if not candidates: return [main_total]
    sorted_c = sorted(candidates, key=lambda t: -posterior.get(t, 0))
    picked = [sorted_c[0]]
    span = max(candidates) - min(candidates)
    min_gap = max(2, span // 4)
    for t in sorted_c[1:]:
        if abs(t - picked[0]) >= min_gap:
            picked.append(t)
            break
    else:
        if len(sorted_c) > 1: picked.append(sorted_c[1])
    return sorted(picked)

def build_two_group(main_pattern: str, main_total: int, posterior: Dict[int, float]) -> Dict:
    all_patterns = ["大双", "大单", "小双", "小单"]
    pattern_scores = {}
    for p in all_patterns:
        if p == main_pattern: continue
        totals = get_pattern_totals(p)
        pattern_scores[p] = max(posterior.get(t, 0) for t in totals)
    alt = max(pattern_scores, key=pattern_scores.get)

    main_totals = pick_two_totals(main_pattern, main_total, posterior)
    alt_totals = pick_two_totals(alt, main_total, posterior)
    return {
        "main_pattern": main_pattern, "alt_pattern": alt,
        "main_totals": main_totals, "alt_totals": alt_totals,
        "group_text": f"双组推荐：{main_pattern}({'、'.join(map(str, main_totals))}) / {alt}({'、'.join(map(str, alt_totals))})"
    }

# ============================================================
#  数据获取与转换
# ============================================================
def fetch_history_data() -> List[Dict]:
    try:
        res = requests.get(API_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15, verify=False)
        data = json.loads(res.text)
        out = []
        for item in data["data"][:HISTORY_LEN]:
            period = int(item["nbr"])
            n1, n2, n3 = map(int, item["number"].split("+"))
            sum_s = int(item["num"])
            pat = get_pattern_by_sum(sum_s)
            shape = detect_shape([n1, n2, n3])
            out.append({"period": period, "a": n1, "b": n2, "c": n3, "total": sum_s, "pattern": pat, "shape": shape})
        out.sort(key=lambda x: x["period"])
        return out
    except Exception as e:
        print(f"获取数据失败: {e}")
        return []

def dicts_to_draws(history_data: List[Dict]) -> List[Draw]:
    return [Draw(period=d['period'], nums=[d['a'], d['b'], d['c']], total=d['total'], pattern=d['pattern'], shape=d.get('shape', '杂六')) for d in history_data]

# ============================================================
#  回测
# ============================================================
def backtest(history_data: List[Dict], n_test: int = BACKTEST_N) -> str:
    if len(history_data) < 40: return "历史数据不足"
    start = max(30, len(history_data) - n_test)
    hits_main = hits_alt = hits_both = 0
    total = 0
    for i in range(start, len(history_data)):
        train = history_data[:i]
        actual = history_data[i]['total']
        pred = V44Predictor(dicts_to_draws(train)).predict()
        two = build_two_group(pred['pattern'], pred['total'], pred['posterior'])
        in_main = actual in two['main_totals']
        in_alt = actual in two['alt_totals']
        if in_main: hits_main += 1
        if in_alt: hits_alt += 1
        if in_main or in_alt: hits_both += 1
        total += 1
    return f"主组命中率: {hits_main/total*100:.1f}%\n副组命中率: {hits_alt/total*100:.1f}%\n双组联合命中率: {hits_both/total*100:.1f}%"

# ============================================================
#  Bot 核心逻辑
# ============================================================
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
        [InlineKeyboardButton("🎯 预测下期 (V4.4)", callback_data="predict")],
        [InlineKeyboardButton("🧪 回测最近100期", callback_data="backtest")],
    ]
    await update.message.reply_text("请选择操作类型：", reply_markup=InlineKeyboardMarkup(kb))

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    try: await q.answer()
    except: pass

    if q.data == "check_join":
        if await is_user_joined(update, context):
            kb = [
                [InlineKeyboardButton("🎯 预测下期 (V4.4)", callback_data="predict")],
                [InlineKeyboardButton("🧪 回测最近100期", callback_data="backtest")],
            ]
            await q.edit_message_text("✅ 验证成功！请选择操作类型：", reply_markup=InlineKeyboardMarkup(kb))
        else:
            try: await q.answer("❌ 验证失败，请先加入频道！", show_alert=True)
            except: pass
        return

    if not await is_user_joined(update, context):
        await q.edit_message_text("⚠️ 请先加入频道后再使用！发送 /start 重新验证。")
        return

    if q.data == "predict":
        await q.edit_message_text("⚡ 正在拉取最新数据并计算...")
        history_list = fetch_history_data()
        if not history_list:
            await q.edit_message_text("❌ 数据拉取失败，请稍后重试")
            return
        
        pred = V44Predictor(dicts_to_draws(history_list)).predict()
        two = build_two_group(pred['pattern'], pred['total'], pred['posterior'])
        
        latest = history_list[-1]
        text = f"🏰 最新期号：{latest['period']}\n"
        text += f"🏰 开奖号：{latest['a']}+{latest['b']}+{latest['c']}={latest['total']} [{latest['pattern']}]\n\n"
        text += f"🎯 主组形态(条件概率)：{two['main_pattern']}\n"
        text += f"🔢 主组特码：{'、'.join(map(str, two['main_totals']))}\n\n"
        text += f"🎯 副组形态：{two['alt_pattern']}\n"
        text += f"🔢 副组特码：{'、'.join(map(str, two['alt_totals']))}\n\n"
        text += f"📌 双组推荐：\n{two['group_text']}\n\n"
        text += f"📊 后验Top5和值：{pred['top5_totals']}\n"
        text += "\n⚠️ 纯数学游戏，娱乐参考"
        
        await q.edit_message_text(text)

    elif q.data == "backtest":
        await q.edit_message_text("🧪 正在拉取历史数据并进行回测，请稍候...")
        history_list = fetch_history_data()
        if not history_list:
            await q.edit_message_text("❌ 数据拉取失败，请稍后重试")
            return
        
        bt_result = backtest(history_list, n_test=BACKTEST_N)
        text = f"📊 最近 {BACKTEST_N} 期回测结果：\n\n{bt_result}\n\n⚠️ 历史表现不代表未来"
        await q.edit_message_text(text)

# ══════════ 适配 Railway 的假 Web 服务器 ══════════
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
        
    threading.Thread(target=run_dummy_server, daemon=True).start()
    
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(on_button))
    print("V4.4 Bot 已启动！")
    app.run_polling()

if __name__ == "__main__":
    main()
