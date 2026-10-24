# -*- coding: utf-8 -*-
"""
AI 群聊后端服务

两种对话模式:
1. 依次回答(chat):   豆包 / 通义千问 / DeepSeek 依次独立回复用户
2. 讨论模式(discuss): 三位 AI 先轮流发表观点(opinion) -> 交叉讨论(discuss)
                     -> 主持人总结(summary)，讨论过程全程在群里可见

运行方式:
    python server.py
然后浏览器访问 http://127.0.0.1:5000

配置真实 API（可选，不配置则为演示模式）:
    set DOUBAO_API_KEY=你的豆包(火山方舟)API Key
    set QWEN_API_KEY=你的通义千问(百炼)API Key
    set DEEPSEEK_API_KEY=你的DeepSeek API Key
"""
import os
import random

import requests
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR)

# ---------------- 三个大模型的配置（均为 OpenAI 兼容接口） ----------------
MODELS = [
    {
        "key": "doubao",
        "name": "豆包",
        "env_key": "DOUBAO_API_KEY",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        "model": os.environ.get("DOUBAO_MODEL", "doubao-1-5-pro-32k-250115"),
        "persona": "风格活泼亲切、口语化，回答简洁",
    },
    {
        "key": "qwen",
        "name": "通义千问",
        "env_key": "QWEN_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": os.environ.get("QWEN_MODEL", "qwen-plus"),
        "persona": "风格条理清晰、善用分点，严谨专业",
    },
    {
        "key": "deepseek",
        "name": "DeepSeek",
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        "persona": "风格理性深入、注重逻辑分析",
    },
]


def model_status():
    """返回每个模型是否配置了真实 API Key"""
    return [{"key": m["key"], "name": m["name"], "live": bool(os.environ.get(m["env_key"]))} for m in MODELS]


def transcript_text(transcript):
    """把讨论记录转成文本，供 prompt 使用"""
    return "\n".join(f"{t.get('name', '成员')}：{t.get('content', '')}" for t in transcript)


def build_messages(model_cfg, phase, question, history, transcript):
    """根据对话阶段构造发给大模型的消息"""
    base_system = (
        f"你是{model_cfg['name']}，正在一个 AI 群聊里发言，{model_cfg['persona']}。"
        "群里有另外两位 AI 成员和一位用户。"
    )
    if phase == "chat":  # 依次回答模式
        return [{"role": "system", "content": base_system + "请直接回答用户问题，150 字以内。"}] + history

    if phase == "opinion":  # 讨论第一轮：发表观点
        prompt = f"群里有用户提问：「{question}」。请发表你的看法，开门见山，80 字以内。"
    elif phase == "discuss":  # 讨论第二轮：交叉讨论
        prompt = (
            f"群里正在讨论问题：「{question}」。\n"
            f"以下是到目前为止的讨论记录：\n{transcript_text(transcript)}\n"
            "请回应其他成员的观点（可以赞同、补充或反驳），体现出讨论感，80 字以内。"
        )
    elif phase == "summary":  # 讨论总结（主持人）
        prompt = (
            f"群里就问题「{question}」进行了讨论，完整记录：\n{transcript_text(transcript)}\n"
            "你是本次讨论的主持人，请总结：三方共识、主要分歧、最终结论建议。"
            "用「共识 / 分歧 / 结论」三段式输出，150 字以内。"
        )
    else:
        prompt = question
    return [
        {"role": "system", "content": base_system},
        {"role": "user", "content": prompt},
    ]


def call_real_api(model_cfg, phase, question, history, transcript):
    """调用真实大模型接口（OpenAI 兼容格式），未配置 Key 返回 None"""
    api_key = os.environ.get(model_cfg["env_key"])
    if not api_key:
        return None
    try:
        resp = requests.post(
            model_cfg["base_url"],
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model_cfg["model"],
                "messages": build_messages(model_cfg, phase, question, history, transcript),
                "temperature": 0.7,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"（接口调用失败：{e}）"


# ---------------- 演示模式：未配置 Key 时生成模拟回复 ----------------
def mock_chat(model_key, question):
    q = question.strip().rstrip("？?。.")
    if model_key == "doubao":
        templates = [
            f"哈哈这个问题有意思～关于「{q}」，我的看法是：先从最简单的角度理解，别把它想复杂了。抓住核心点，剩下的都是细节问题，动手试一下就知道啦！",
            f"哇，「{q}」这个话题我喜欢！简单说哈：先明确你真正想达到的目标，然后拆成小步骤一个个来。有具体场景的话可以再说说，我帮你细化～",
        ]
    elif model_key == "qwen":
        templates = [
            f"关于「{q}」，我从三个方面分析：\n1. 首先明确问题的本质和目标；\n2. 其次梳理现有的主流方案及其优缺点；\n3. 最后结合你的实际情况选择最合适的路径。\n如果需要，我可以针对其中任何一点展开详细说明。",
            f"针对「{q}」，我的结构化建议如下：\n一、先调研，收集足够信息，避免拍脑袋决策；\n二、做对比，列出候选方案的评估矩阵；\n三、小步验证，用最小成本试错。\n希望这个框架对你有帮助。",
        ]
    else:
        templates = [
            f"从逻辑上拆解「{q}」：问题的复杂度主要取决于约束条件。如果约束明确，可以用演绎法逐步推导最优解；如果约束模糊，建议先用归纳法从案例中找规律。本质上这是一个权衡问题，关键在于确定你的优化目标。",
            f"分析一下「{q}」：\n先说结论——大多数情况下没有银弹。核心思路是：1) 把问题形式化，明确输入输出；2) 识别瓶颈所在；3) 针对性地选择方法。实践中 80% 的效果往往来自 20% 的关键决策。",
        ]
    return random.choice(templates)


def mock_opinion(model_key, question):
    q = question.strip().rstrip("？?。.")
    if model_key == "doubao":
        return f"关于「{q}」，我的观点很直接：别想太多，先干起来！做个最小化的尝试，问题会在实践中自己暴露出来。"
    if model_key == "qwen":
        return f"关于「{q}」，我认为首先要明确目标和评估标准，然后把问题结构化拆解，逐个击破，避免眉毛胡子一把抓。"
    return f"关于「{q}」，我的观点是：先形式化问题，明确约束条件。大部分困难来自问题定义模糊，而不是缺少方法。"


def mock_discuss(model_key, question, transcript):
    my_name = next(m["name"] for m in MODELS if m["key"] == model_key)
    others = [t for t in transcript if t.get("name") != my_name]
    other = random.choice(others).get("name", "另一位") if others else "另一位"
    if model_key == "doubao":
        return random.choice([
            f"我挺同意{other}的～不过我补充一句：标准再清晰，不动手也是白搭，咱们先跑起来再迭代嘛！",
            f"{other}说的框架我认可，但大家别陷在规划里出不来呀，我的经验是边做边调整效率最高～",
        ])
    if model_key == "qwen":
        return random.choice([
            f"我部分认同{other}的观点，但要指出：缺乏评估标准的实践容易跑偏。建议先定义'做到什么程度算解决'，再投入行动。",
            f"补充{other}一点：实践中也要设置检查节点，定期对照目标复盘，否则'边做边改'可能演变成无方向的试错。",
        ])
    return random.choice([
        f"对{other}的看法我持保留意见：这默认了问题边界是清晰的。但现实中约束往往动态变化，我坚持先把问题形式化，否则讨论容易失焦。",
        f"{other}说的行动优先有道理，但从概率上看，未经分析的试错期望成本更高。折中方案：花 20% 时间分析，再快速验证。",
    ])


def mock_summary(question, transcript):
    q = question.strip().rstrip("？?。.")
    return random.choice([
        (
            f"【共识】关于「{q}」，三位一致认为：需要先明确目标，再通过实践验证，过程中保持迭代。\n"
            "【分歧】豆包主张行动优先、边做边学；通义千问强调先建立评估标准和结构；DeepSeek 坚持先形式化问题再动手。\n"
            "【结论】建议采用折中路径：用少量时间界定问题与验收标准，然后快速做一个最小化验证，根据结果迭代调整。"
        ),
        (
            f"【共识】关于「{q}」，大家都认可'目标先行、小步快跑、持续修正'这个大方向。\n"
            "【分歧】分歧主要在先后顺序：先规划还是先行动，以及分析应该投入多大比重。\n"
            "【结论】最终建议：按'2 分分析、6 分实践、2 分复盘'的节奏推进，既不盲目行动，也不陷入过度规划。"
        ),
    ])


def mock_generate(model_key, phase, question, history, transcript):
    if phase == "opinion":
        return mock_opinion(model_key, question)
    if phase == "discuss":
        return mock_discuss(model_key, question, transcript)
    if phase == "summary":
        return mock_summary(question, transcript)
    last_user_msg = next((m["content"] for m in reversed(history) if m["role"] == "user"), question)
    return mock_chat(model_key, last_user_msg)


# ---------------- 路由 ----------------
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"models": model_status()})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    model_key = data.get("model")
    phase = data.get("phase", "chat")          # chat / opinion / discuss / summary
    history = data.get("history", [])          # 依次回答模式的上下文
    question = data.get("question", "")        # 讨论模式的原始问题
    transcript = data.get("transcript", [])    # 讨论模式到目前为止的记录

    model_cfg = next((m for m in MODELS if m["key"] == model_key), None)
    if not model_cfg:
        return jsonify({"error": "unknown model"}), 400

    reply = call_real_api(model_cfg, phase, question, history, transcript)
    if reply is None:  # 未配置 Key，走演示模式
        reply = mock_generate(model_key, phase, question, history, transcript)

    return jsonify({"model": model_key, "name": model_cfg["name"], "content": reply})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 50)
    print(f"AI 群聊服务已启动: http://127.0.0.1:{port}")
    print("支持模式: 依次回答 / 讨论模式(观点->讨论->总结)")
    for s in model_status():
        mode = "真实API" if s["live"] else "演示模式"
        print(f"  - {s['name']}: {mode}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=False)
