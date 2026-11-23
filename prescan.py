#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zh-paper-audit / prescan.py — 中文论文文本自检

查的是"文本本身的形态"，不是"AI 生成概率"：

  1. 结构模板   中文序数骨架（其一/第一/首先/一是/(1)/①②）、首句同构、稿内重复论证
  2. 引用严谨度 无具体归属的权威（"有学者认为"）、来源清单
  3. 改写守恒   数字 token、来源、论点强度、论证覆盖、逻辑方向——改完没丢东西
  4. 文本体检   判断这份稿子像不像论文；不像就先警告，避免阈值误用

不查：AIGC 检测分数。检测器算的是语言模型困惑度，与本工具输出的指标没有
换算关系。本工具不承诺任何平台的分数，也不承诺"改完会降"。

用法：
  python3 prescan.py 稿子.md                      单稿体检
  python3 prescan.py 稿子.md --json out.json      存档
  python3 prescan.py 稿子.md --txt 纯文本.txt      导出可直接上传检测平台的纯文本
  python3 prescan.py 原稿.md --diff 改后.md        前后对比 + 五项守恒
  python3 prescan.py 原稿.md --diff v2.md --base v1.md   第二轮返修：本轮/累计双轨
  python3 prescan.py 稿子.md --genre lab          换文体口径（academic/lab/essay）
  python3 prescan.py --list-genres                看有哪些口径（不需要文件）
  python3 prescan.py --version                    版本与 JSON schema 版本

设计原则：
  - 只报可计数、可定位的事实，不代替人做"该不该改"的判断
  - 纯标准库，无网络、无子进程、无 eval，只读输入文件
  - 输出确定：同一文件两次运行结果完全一致
  - 输出安全：报告/JSON/纯文本绝不覆盖任何输入文件；写出使用同目录临时文件原子替换
  - 编码安全：UTF-8（含 BOM）优先，中文 Windows 的 GB18030 自动回退；解不开就报错，不静默乱码
"""

import sys
import os
import re
import json
import argparse
import statistics
import tempfile
from collections import Counter, OrderedDict
from difflib import SequenceMatcher

__version__ = "2.4.0"
SCHEMA_VERSION = 1

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ============================================================
# 词表与正则（可按学科自行增删）
# ============================================================

# 序数骨架：中文写作里最典型的"预喂结构"痕迹
ORDINAL_RULES = [
    ("其一/其二", r"其[一二三四五六七八九十](?=[，,、。：；\s]|$)"),
    ("第N，(顿逗)", r"第[一二三四五六七八九十]+(?=[，,、])"),
    ("首先/其次/再次/最后", r"(?:首先|其次|再次|最后|最终)[，,]"),
    # 「一是……，二是……」是标准并列骨架，多数不带逗号紧跟内容，
    # 所以要求「X是」后接 1–25 字再遇句读，而不是要求紧跟逗号。
    # 负向断言排除「统一是 / 唯一是 / 单一是」这类普通表述。
    ("一是/二是", r"(?<![统唯单])[一二三四五六七八九]是[^，。；！？\n]{1,25}[，。；]"),
    ("一方面/另一方面", r"(?:另一方面|一方面)[，,]"),
    ("(N) 括号编号", r"(?<![图表格式第次])[（(]\d{1,2}[）)](?!\s*[a-zA-Z%])"),
    ("①②③ 圈号", r"[①②③④⑤⑥⑦⑧⑨⑩]"),
    ("行首字母序号", r"(?m)^[ \t]*[A-Za-z][\.、][ \t]"),
]

# 强信号：几乎只在营销/宣传语里出现，学术文本中一律可疑
STRONG_ASSERT = [
    "高度契合", "完美契合", "深度契合", "自然耦合", "天然契合", "恰好切入", "精准切入",
    "奠定基础", "奠定坚实", "注入活力", "注入新动能", "彰显", "凸显", "赋能", "闭环",
    "抓手", "护城河", "生态位", "毋庸置疑", "众所周知", "显而易见", "不可否认",
    "重新定义", "颠覆", "全新范式", "新范式", "里程碑", "破圈", "出圈",
    "无缝衔接", "深度融合", "全面赋能", "强力支撑", "有力保障",
]

# 弱信号：本身合理，需看上下文与密度，只报计数不建议直接改
WEAK_ASSERT = [
    "恰好", "天然", "自然", "显著", "充分", "极大", "有效", "成功", "完美",
    "意味着", "体现出", "折射出", "反映了", "本质上是", "归根结底",
]

# 借来的权威：出现即需在 40 字内找到具体人名/机构/年份/文献号
VAGUE_AUTHORITY = [
    r"有学者(?:认为|指出|提出|主张)",
    r"(?:研究|学界|业内|业界|专家|学者)(?:普遍)?(?:认为|指出|表明|显示)",
    r"一般认为", r"有人认为", r"众所周知",
    r"被[《“\"]?[^》”\"]{2,30}[》”\"]?(?:评价|誉为|称为)为",
    r"相关研究(?:表明|显示|指出)",
    r"大量(?:研究|实践)(?:表明|显示|证明)",
    r"国内外(?:学者|研究)(?:表明|显示|指出)",
]

# 元语言标记：出现这些词，说明作者是在"讨论"某个说法，而不是"使用"它。
# 这是 use / mention 区分：引号内、表格内、含元语言词的句子是在「讨论」某个说法，
# 不是作者在「使用」它。不做这层区分会把引用核验报告里的条目全判成风险。
META_WORDS = ["表述", "原文", "改成", "改为", "写作", "引用", "所谓", "这句话",
              "这一说法", "的说法", "改写", "措辞", "误判", "例句", "表述为",
              "写成", "说的是", "提到的", "原句", "标题里"]

# 常见机构/报告/媒体库：命中即计入"来源清单"，用于前后守恒。
#
# 这张表原本是一篇商科论文的定制产物（贝恩/艾瑞/巴黎世家…），换个学科就全是
# 死词——实测工科文本命中 0。所以改成「内置保底 + data/sources.*.txt 外部扩展」：
# 换领域只需新建一个文本文件，不用改代码。
BUILTIN_SOURCES = [
    "贝恩", "Bain", "Newzoo", "艾瑞", "美团", "Live Nation", "WGSN", "rEvolution",
    "DRESSX", "前瞻产业研究院", "前瞻", "麦肯锡", "McKinsey", "德勤", "Deloitte",
    "普华永道", "PwC", "安永", "EY", "毕马威", "KPMG", "尼尔森", "Nielsen",
    "Statista", "IDC", "Gartner", "QuestMobile", "艾媒", "易观", "CBNData",
    "Vogue Business", "Business of Fashion", "WWD", "Forbes", "Bloomberg",
    "彭博", "路透", "Reuters", "经济学人", "Economist", "哈佛商业评论",
    "国家统计局", "CNNIC", "中国互联网络信息中心", "Sensor Tower",
    "Steam", "Twitch", "YouTube", "TikTok", "微博", "抖音", "小红书", "哔哩哔哩",
    "B站", "英雄联盟", "VALORANT", "无畏契约", "巴黎世家", "Balenciaga", "Gucci",
    "Louis Vuitton", "LV", "KFC", "肯德基", "海底捞",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
# data/ 可能和脚本同级（本仓库布局），也可能在上一级（scripts/ 布局）。
# 两种都支持，这样脚本单独拷到任何地方都能跑。
_DATA_CANDIDATES = [os.path.join(_HERE, "data"), os.path.join(os.path.dirname(_HERE), "data")]
DATA_DIR = next((p for p in _DATA_CANDIDATES if os.path.isdir(p)), _DATA_CANDIDATES[0])

# 运行时词表：内置保底 + 外部文件合并，就地更新以保持引用不变
SOURCE_LIB = []


def load_source_lib(only=None, extra=()):
    """合并内置表与 data/sources.*.txt，就地更新 SOURCE_LIB。

    only 给定时只加载指定文件（--sources），用于"我的领域我自己管"；
    否则加载 data/ 下全部 sources.*.txt。data/ 目录不存在时退回内置表
    （单文件拷走的场景），但显式指定的词表文件坏/缺会直接报错——
    用户点名要的文件不能静默丢掉，否则来源守恒是在用半张词表跑的。
    """
    merged = OrderedDict((s, True) for s in BUILTIN_SOURCES)
    files = []
    if only:
        files = list(only)
    elif os.path.isdir(DATA_DIR):
        files = sorted(
            os.path.join(DATA_DIR, f)
            for f in os.listdir(DATA_DIR)
            if f.startswith("sources.") and f.endswith(".txt")
        )
    loaded = 0
    for p in files:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                merged[s] = True
                loaded += 1
    for s in extra:
        merged[s] = True
    SOURCE_LIB[:] = list(merged.keys())
    return loaded


# 模块加载时就初始化一次：测试与实际运行必须用同一份词表
load_source_lib()

# 论点强度词：用于检测"弱断言被改写成强断言"（本次踩过的坑）
WEAK_CLAIM = ["有助于", "可能", "或许", "在一定程度上", "倾向于", "有望", "或可",
              "暗示", "提示", "似乎", "大概", "未必", "尚不确定", "有待", "尚未"]
STRONG_CLAIM = ["证明", "证实", "必然", "一定会", "完全", "彻底", "根本", "毫无疑问",
                "决定", "确保", "必将", "一定能够", "充分证明", "确凿"]

# 数字 token：前后稿用同一正则，保证守恒核验只反映真实增删
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?%?")
DASH_RE = re.compile(r"—+")
HAN_RE = re.compile(r"[\u4e00-\u9fff]")

# 段落合理长度按文体给——不同文体的"正常段长"差得很远，套错会大面积误报
GENRE_PROFILES = {
    "academic": {"para": (120, 350), "note": "人文／商科论述：一段承载一个完整论点"},
    "lab":      {"para": (50, 220),  "note": "工科实验报告：一段一个步骤或一个结果"},
    "essay":    {"para": (80, 280),  "note": "评论／案例分析叙述"},
}
PARA_IDEAL = list(GENRE_PROFILES["academic"]["para"])
PARA_TOO_SHORT = int(PARA_IDEAL[0] * 0.67)
PARA_TOO_LONG = int(PARA_IDEAL[1] * 1.43)
CURRENT_GENRE = "academic"

# 原稿真实 AIGC 分数，由 --baseline 传入。
# 为什么必须显式传入：v1 最大的失误是不问起点就开改——
# 如果原稿本来就是 8%（自己写却被误判），一通改写可能把它推到 20%。
BASELINE = None


# ============================================================
# 基础解析
# ============================================================

def read_text(path):
    """读入文本：UTF-8 优先，中文 Windows 常见 GB18030 自动回退。

    旧实现用 errors="replace"，编码错的稿子会静默变成一串 U+FFFD，
    报告仍然照常输出——这是最坏的失败方式：错得无声。这里改成严格解码，
    解不开就报错；BOM（UTF-8/UTF-16）也会被正确剥掉。
    """
    with open(path, "rb") as f:
        data = f.read()

    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("gb18030")
    except UnicodeDecodeError:
        # 再试一次允许替换，只为生成包含替换符的提示；真实内容不会用于报告
        preview = data.decode("gb18030", errors="replace")
        raise UnicodeDecodeError("utf-8/gb18030", data, 0, min(len(data), 24),
                                 "无法解码（前 24 字节预览：%r）" % preview[:24])


def same_path(a, b):
    """两个路径是否指向同一文件（大小写与分隔符差异不算不同）。"""
    return bool(a) and bool(b) and os.path.normcase(os.path.abspath(a)) == \
        os.path.normcase(os.path.abspath(b))


def check_path_collision(args):
    """输出文件绝不允许覆盖任何输入文件，也不允许互相覆盖。

    这不是形式检查：--out 指到原稿上时，报告写入会直接吃掉论文；
    --txt 与 --out 同路径时，后写的报告会覆盖先写的纯文本。
    """
    inputs = [("输入稿", args.file)]
    if args.diff:
        inputs.append(("改后稿", args.diff))
    if args.base:
        inputs.append(("本轮起点", args.base))
    outputs = [(name, path) for name, path in (
        ("--out", args.out), ("--json", args.json), ("--txt", args.txt)) if path]
    for out_name, out_path in outputs:
        for in_name, in_path in inputs:
            if same_path(out_path, in_path):
                raise ValueError("%s（%s）不能指向%s（%s）：会覆盖输入文件"
                                 % (out_name, out_path, in_name, in_path))
    for i, (name_a, path_a) in enumerate(outputs):
        for name_b, path_b in outputs[i + 1:]:
            if same_path(path_a, path_b):
                raise ValueError("%s 与 %s 不能指向同一个文件（%s）：后写的会覆盖先写的"
                                 % (name_a, name_b, path_a))


def write_output(path, content):
    """原子写出：写到同目录临时文件，成功后一次性替换目标。

    半途失败时目标文件保持原样，不会留下半份报告。
    """
    target = os.path.abspath(path)
    parent = os.path.dirname(target)
    if not os.path.isdir(parent):
        raise OSError("输出目录不存在：%s" % parent)
    fd, tmp = tempfile.mkstemp(prefix=".prescan-", suffix=".tmp", dir=parent)
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            f.write(content.rstrip() + "\n")
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def strip_markdown(t):
    """去掉 markdown 语法但保留段落结构。全角标点不动。"""
    t = t.replace("\ufeff", "").replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
    t = re.sub(r"```.*?```", "", t, flags=re.S)
    t = re.sub(r"^#{1,6}[ \t]+", "", t, flags=re.M)
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"\*(.+?)\*", r"\1", t)
    t = re.sub(r"`(.+?)`", r"\1", t)
    t = re.sub(r"^\s*>\s?", "", t, flags=re.M)
    t = re.sub(r"\[(.+?)\]\((?:[^)]*)\)", r"\1", t)
    # markdown 列表符号必须去掉：导出给检测平台的纯文本带上 "- " 会被当成
    # 非自然排版，也可能干扰检测器的分段判断
    t = re.sub(r"^[ \t]*[-*+][ \t]+", "", t, flags=re.M)
    t = re.sub(r"^[ \t]*\d{1,2}[.)][ \t]+", "", t, flags=re.M)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def split_paras(t):
    return [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]


def split_sentences(t):
    s = re.split(r"(?<=[。！？!?])", t)
    return [x.strip() for x in s if len(HAN_RE.findall(x)) >= 2]


def han_count(s):
    return len(HAN_RE.findall(s))


def is_heading_line(line):
    """判断是否为小节标题行（标题里的冒号不计入正文冒号）。"""
    s = line.strip()
    if not s:
        return False
    if re.match(r"^[（(][一二三四五六七八九十\d]+[）)]", s):
        return True
    if re.match(r"^[一二三四五六七八九十]+[、\.]", s):
        return True
    if re.match(r"^第[一二三四五六七八九十\d]+[章节部分]", s):
        return True
    if len(s) <= 40 and s.endswith("：") is False and "：" in s and len(s.split("：")[0]) <= 14:
        # 「（一）产品（Product）：功能性联名…」这类，前段短且无句读
        head = s.split("：")[0]
        if not re.search(r"[，。；,;]", head):
            return True
    return False


def is_heading_para(p):
    """整段是不是一个小节标题。标题段不参与段长/对称统计——否则中位数会被拉垮。"""
    s = p.strip()
    if is_heading_line(s):
        return True
    # 正文段不会只有十几个字且没有句读
    if han_count(s) < 30 and not re.search(r"[。；，,;]", s):
        return True
    return False


def quote_ranges(t):
    """所有被引号/书名号包裹的字符区间。用于区分"使用"与"提及"。"""
    spans = []
    for pat in (r"「[^」]{0,300}」", r"“[^”]{0,300}”", r'"[^"\n]{0,300}"',
                r"『[^』]{0,300}』", r"《[^》]{0,100}》", r"‘[^’]{0,100}’"):
        for m in re.finditer(pat, t):
            spans.append((m.start(), m.end()))
    return spans


def mention_reason(t, start, end, spans):
    """判断这处命中是"在讨论某个说法"还是"在使用它"。返回原因或 None。"""
    for s, e in spans:
        if s <= start and end <= e:
            return "在引号或书名号内"
    s0 = t.rfind("。", 0, start) + 1
    e0 = t.find("。", end)
    sent = t[s0: e0 if e0 != -1 else len(t)]
    for w in META_WORDS:
        if w in sent:
            return "元语言讨论（含“%s”）" % w
    # 表格行：| … | … | 属引用汇总，不是正文论断
    line_start = t.rfind("\n", 0, start) + 1
    line_end = t.find("\n", end)
    line = t[line_start: line_end if line_end != -1 else len(t)]
    if line.count("|") >= 2:
        return "在表格内"
    return None


# ============================================================
# 各项检测
# ============================================================

def detect_ordinals(t):
    """中文序数骨架：两工具共同盲区。返回明细与按段聚集情况。"""
    hits = []
    for name, pat in ORDINAL_RULES:
        for m in re.finditer(pat, t):
            start = max(0, m.start() - 12)
            hits.append({
                "kind": name,
                "text": m.group(0),
                "context": t[start:m.end() + 22].replace("\n", " "),
                "pos": m.start(),
            })
    hits.sort(key=lambda x: x["pos"])

    # 同段内出现 ≥2 个不同序数 → 模板骨架的强证据
    paras = split_paras(t)
    clusters = []
    offset = 0
    for idx, p in enumerate(paras):
        cnt = [h for h in hits if offset <= h["pos"] < offset + len(p) + 2]
        if len(cnt) >= 2:
            clusters.append({
                "para_index": idx + 1,
                "count": len(cnt),
                "kinds": sorted({h["kind"] for h in cnt}),
                "preview": p[:56],
            })
        offset += len(p) + 2
    return hits, clusters


def detect_punct(t, raw_text):
    """标点密度。分母用千汉字。"""
    han = max(han_count(t), 1)
    body_lines = [ln for ln in raw_text.split("\n") if ln.strip() and not is_heading_line(ln)]
    body = "\n".join(body_lines)

    def dens(n):
        return round(n / han * 1000, 3)

    dash = len(DASH_RE.findall(t))
    return OrderedDict([
        ("破折号（——）", {"count": dash, "per_k": dens(dash), "ref": "人类写作约 0.80 /千字（口径参考 lieflat-less-ai-tone）"}),
        ("顿号（、）", {"count": t.count("、"), "per_k": dens(t.count("、")), "ref": "本口径为 /千汉字，与他家口径不同，只看前后变化"}),
        ("分号（；）", {"count": t.count("；"), "per_k": dens(t.count("；")), "ref": "分模型指纹：GPT 偏好 / Opus 回避，不作普适规则"}),
        ("冒号-正文", {"count": body.count("："), "per_k": dens(body.count("：")), "ref": "标题行已剔除"}),
        ("冒号-标题", {"count": sum(ln.count("：") for ln in raw_text.split("\n") if is_heading_line(ln)), "per_k": 0, "ref": "小节标题自带，属正常结构"}),
    ])


def detect_assertions(t):
    strong = [(w, t.count(w)) for w in STRONG_ASSERT if t.count(w) > 0]
    weak = [(w, t.count(w)) for w in WEAK_ASSERT if t.count(w) > 0]
    return strong, weak


def detect_vague_authority(t):
    """借来的权威：命中后检查 40 字内有无具体归属（人名/机构/年份/文献号）。"""
    raw_hits = []
    for pat in VAGUE_AUTHORITY:
        for m in re.finditer(pat, t):
            raw_hits.append((m.start(), m.end(), m.group(0)))
    # 去重叠：「有学者认为」与「学者认为」指向同一处，按起点升序、保留较长者
    raw_hits.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    kept, last_end = [], -1
    for s, e, txt in raw_hits:
        if s >= last_end:
            kept.append((s, e, txt))
            last_end = e

    out = []
    spans = quote_ranges(t)
    for s, e, txt in kept:
        window = t[max(0, s - 30): e + 45]
        has_anchor = bool(re.search(
            r"[（(]\s*(?:\d{4}|[A-Z][a-z]+|[\u4e00-\u9fff]{2,4}[,，]\s*\d{4})"
            r"|《[^》]{2,40}》|\[\d+\]|doi|et al",
            window))
        out.append({
            "text": txt,
            "context": window.replace("\n", " "),
            "anchored": has_anchor,
            "mention": mention_reason(t, s, e, spans),
        })
    return out


def detect_para_rhythm(paras):
    """段长与对称性。只统计正文段——标题段会污染中位数与对称判断。"""
    body_idx = [i for i, p in enumerate(paras) if not is_heading_para(p)]
    lens_all = [han_count(p) for p in paras]
    lens = [lens_all[i] for i in body_idx]
    if not lens:
        return {}
    # 连续段长度相近：偏差 <20%、长度 ≥50 汉字、连续 ≥3 段
    # 门槛 50 字是因为短段天然长度接近，不构成"刻意对称"的证据
    runs = []
    cur = 1
    for i in range(1, len(lens)):
        dev = abs(lens[i] - lens[i - 1]) / max(lens[i - 1], 1)
        if dev < 0.20 and lens[i] >= 50:
            cur += 1
        else:
            if cur >= 3:
                runs.append((i - cur, cur))
            cur = 1
    if cur >= 3:
        runs.append((len(lens) - cur, cur))

    run_detail = []
    for start, n in runs:
        run_detail.append({
            "start_para": body_idx[start] + 1,
            "count": n,
            "lengths": lens[start:start + n],
            "preview": paras[body_idx[start]][:40],
        })

    too_short = [body_idx[i] + 1 for i, x in enumerate(lens) if x < PARA_TOO_SHORT]
    too_long = [body_idx[i] + 1 for i, x in enumerate(lens) if x > PARA_TOO_LONG]

    return {
        "count": len(paras),
        "body_count": len(lens),
        "median": round(statistics.median(lens), 1),
        "mean": round(statistics.mean(lens), 1),
        "min": min(lens),
        "max": max(lens),
        "too_short": too_short,
        "too_long": too_long,
        "sym_runs": run_detail,
        "ideal_range": PARA_IDEAL,
    }


def detect_sentence_cv(t):
    sents = split_sentences(t)
    if len(sents) < 5:
        return {"count": len(sents), "cv": None}
    lens = [han_count(s) for s in sents]
    mean = statistics.mean(lens)
    cv = statistics.stdev(lens) / mean if mean else 0
    return {"count": len(sents), "cv": round(cv, 3),
            "mean_len": round(mean, 1), "max_len": max(lens), "min_len": min(lens)}


def detect_coverage(before, after, thresh=0.45, alarm=0.6):
    """段落覆盖率：原稿每个正文段，在改后稿里还找不找得到对应论证。

    防的是「改写时整段消失，但数字和来源一个没少」——这是守恒核验的最大盲区。
    0.45 是句级"视为保留"的相似度线；段落命中率低于 0.6 即报警。
    """
    pb = [p for p in split_paras(before) if not is_heading_para(p)]
    pa = [p for p in split_paras(after) if not is_heading_para(p)]
    a_sents = []
    for q in pa:
        a_sents += [s.strip() for s in re.split(r"[。；！？!?]", q) if len(s.strip()) > 6]
    missing = []
    for i, p in enumerate(pb):
        sents = [s.strip() for s in re.split(r"[。；！？!?]", p) if len(s.strip()) > 6]
        if not sents:
            continue
        hit, lost = 0, []
        for s in sents:
            best = 0.0
            for s2 in a_sents:
                r = SequenceMatcher(None, s, s2).ratio()
                if r > best:
                    best = r
            if best > thresh:
                hit += 1
            else:
                lost.append({"sent": s[:46], "best": round(best, 2)})
        rate = hit / len(sents)
        if rate < alarm:
            missing.append({
                "para": i + 1, "rate": round(rate, 2), "n_sent": len(sents),
                "lost": lost[:3], "preview": p[:48],
            })
    return {"body_before": len(pb), "body_after": len(pa), "missing": missing}


CONNECTORS = {
    "因果": "因为 因此 所以 由于 从而 以致 导致 使得 故而 由此可见",
    # 「但」单独用远比「但是」常见，只收「但是」会漏掉一半转折句。
    # 代价是「不但/非但」会被误吃，所以在 conn_kinds 里做负向排除。
    "转折": "但是 然而 不过 却 反而 相反 可是 恰恰 但",
    "递进": "而且 并且 不仅 甚至 乃至 进而 况且 不但 非但",
    "让步": "虽然 尽管 固然 即便 即使 诚然",
    "条件": "如果 若 只要 除非 一旦 唯有",
}

# 方向相反的连接词对：原稿是其中一类、改后稿同段变成另一类，最可能是意思被改反了
CONN_OPPOSE = [("因果", "转折"), ("因果", "让步"), ("递进", "转折"), ("条件", "转折")]

# 「但」在这些组合里不是转折：不但…而且、非但…反而、但凡
CONN_FALSE_FRIENDS = ("不但", "非但", "但凡")


def conn_kinds(s):
    """句/段里出现了哪几类逻辑连接词。

    两个虚词要多看一眼：
    - 「但」在「不但/非但/但凡」里不表转折
    - 「反而」在「不但…反而」里是递进，单用才是转折
    """
    kinds = set()
    has_budan = ("不但" in s) or ("非但" in s)
    for k, words in CONNECTORS.items():
        for w in words.split():
            pos = s.find(w)
            while pos >= 0:
                if w == "但":
                    prev = s[pos - 1] if pos > 0 else ""
                    if prev in ("不", "非") or s[pos:pos + 2] == "但凡":
                        pos = s.find(w, pos + 1)
                        continue
                if w == "反而" and has_budan:
                    pos = s.find(w, pos + 1)
                    continue
                kinds.add(k)
                break
    return kinds


def detect_connector_flip(before, after, thresh=0.30):
    """段落级逻辑方向翻转：句子都在、数字都对，但因果被改成了转折。

    为什么必须单列这项：`detect_coverage` 只保证"原稿的句子在改后稿里找得到对应"，
    它认不出「因为 A，所以 B」被改成「B，然而 A」——句子相似度照样很高，
    意思却反了。这是守恒核验最后一层够不着的地方，只能靠逻辑连接词兜。

    刻意只报**方向相反的替换**，不报"连接词变少/变多"：
    改写时删掉冗余连接词是正常操作，报了就是噪音。
    """
    pb = [p for p in split_paras(before) if not is_heading_para(p)]
    pa = [p for p in split_paras(after) if not is_heading_para(p)]
    flips = []
    for i, p in enumerate(pb):
        kb = conn_kinds(p)
        if not kb:
            continue
        best, bi = 0.0, -1
        for j, q in enumerate(pa):
            r = SequenceMatcher(None, p, q).ratio()
            if r > best:
                best, bi = r, j
        if bi < 0 or best < thresh:
            continue
        ka = conn_kinds(pa[bi])
        if not ka:
            continue
        for x, y in CONN_OPPOSE:
            # y not in kb 这个条件不能省：原稿本来就「因为 A，但 B」，
            # 改写只是删掉了"因为"，逻辑没变，不能报成翻转。
            if x in kb and y in ka and x not in ka and y not in kb:
                flips.append({
                    "para": i + 1, "from": x, "to": y, "sim": round(best, 2),
                    "before": p[:48], "after": pa[bi][:48],
                    "to_para": bi + 1,
                })
                break
    return flips


CITE_RE = re.compile(
    r"\[\d{1,2}\]"                                  # [1]
    r"|（[^）]{1,14}(?:19|20)\d{2}[^）]{0,8}）"      # （贝恩，2023）
    r"|\(\s*(?:19|20)\d{2}\s*\)"                    # (2023)
    r"|(?:据|根据|引用自)[^，。；]{2,18}(?:报告|研究|数据|统计|白皮书|年报)"
)
HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s+|[一二三四五六七八九十]{1,3}、|第[一二三四五六七八九十]{1,3}[章节]|\d+(?:\.\d+){0,2}\s+\S)",
    re.M,
)


def sniff_text_type(t):
    """判断这份稿子像不像学术论文。

    为什么需要：脚本对任何中文都能跑，但所有阈值（段长 120–350、序数骨架、
    破折号 0.80/千字）都是照学术文体定的。拿公众号推文、工作周报、产品文案
    来跑，会报出一堆"段长偏短""骨架 3 处"，全是噪音，而人看到报告会当真。

    不追求准确分类，只求在"明显不是论文"时把人拦一下。
    """
    han = han_count(t)
    paras = [p for p in split_paras(t) if not is_heading_para(p)]
    cites = len(CITE_RE.findall(t))
    heads = len(HEADING_RE.findall(t))
    srcs = extract_sources(t)[0]
    semis = len(re.findall(r"[；;]", t))

    score, reasons = 0, []
    if han >= 800:
        score += 2
        reasons.append("篇幅 %d 字" % han)
    elif han >= 400:
        score += 1
        reasons.append("篇幅 %d 字（偏短）" % han)
    else:
        reasons.append("篇幅仅 %d 字" % han)
    if len(paras) >= 5:
        score += 1
        reasons.append("%d 个正文段" % len(paras))
    else:
        reasons.append("正文段仅 %d 个" % len(paras))
    if cites >= 2:
        score += 2
        reasons.append("引用标记 %d 处" % cites)
    elif cites == 1:
        score += 1
        reasons.append("引用标记 1 处")
    else:
        reasons.append("无引用标记")
    if heads >= 2:
        score += 1
        reasons.append("章节标题 %d 个" % heads)
    if srcs:
        score += 1
        reasons.append("可识别来源 %d 个" % len(srcs))
    if semis >= 3:
        score += 1
        reasons.append("分号 %d 个" % semis)

    if score >= 5:
        verdict = "academic"
    elif score >= 3:
        verdict = "maybe"
    else:
        verdict = "unlikely"
    return {"score": score, "verdict": verdict, "reasons": reasons,
            "han": han, "paras": len(paras), "cites": cites, "heads": heads}


STRUCT_MARKS = "了着的地得是在把被使令让"


def detect_parallel(paras, dev_tol=0.30, minrun=3):
    """相邻段「首句长度相近 + 句数相同 + 共享结构标记词」→ 疑似新造的统一微模板。

    防的是坑 2：拆掉「第一第二第三」，又造出「主语+动词+了+宾语」三段同构。
    纯长度相近在学术文本里太常见，所以必须叠加"共享结构标记词"这个条件才报。
    """
    body = [(i, p) for i, p in enumerate(paras) if not is_heading_para(p)]
    info = []
    for i, p in body:
        sents = [s.strip() for s in re.split(r"[。；！？!?]", p) if len(s.strip()) > 3]
        if not sents:
            continue
        first = sents[0]
        info.append({"para": i + 1, "first": first, "flen": han_count(first),
                     "n_sent": len(sents),
                     "marks": set(ch for ch in first if ch in STRUCT_MARKS)})
    runs, cur = [], 1
    for k in range(1, len(info)):
        a, b = info[k - 1], info[k]
        dev = abs(b["flen"] - a["flen"]) / max(a["flen"], 1)
        if dev < dev_tol and a["n_sent"] == b["n_sent"] and (a["marks"] & b["marks"]):
            cur += 1
        else:
            if cur >= minrun:
                runs.append((k - cur, cur))
            cur = 1
    if cur >= minrun:
        runs.append((len(info) - cur, cur))
    out = []
    for start, n in runs:
        out.append({
            "start_para": info[start]["para"],
            "count": n,
            "first_lens": [info[start + t]["flen"] for t in range(n)],
            "firsts": [info[start + t]["first"][:26] for t in range(n)],
            "shared_marks": "".join(sorted(set.intersection(*[info[start + t]["marks"] for t in range(n)]))),
            "n_sent": info[start]["n_sent"],
        })
    return out


def detect_repeats(t, min_han=18, thresh=0.72, limit=8):
    """稿内重复论证：同一断言写了两遍。防的是坑 4。"""
    sents = []
    for m in re.finditer(r"[^。；！？!?\n]{0,300}[。；！？!?]", t):
        s = m.group(0).strip()
        if han_count(s) >= min_han:
            sents.append(s)
    pairs = []
    for i in range(len(sents)):
        for j in range(i + 1, len(sents)):
            r = SequenceMatcher(None, sents[i], sents[j]).ratio()
            if r >= thresh:
                pairs.append({"a": sents[i][:52], "b": sents[j][:52], "sim": round(r, 2)})
                if len(pairs) >= limit:
                    return pairs
    return pairs


def detect_term_variants(t):
    """同一术语的写法不一致。两类硬信号，几乎零误报：

      1. 拉丁词大小写变体：VALORANT / Valorant、Gartner / GARTNER
      2. 拉丁字母与汉字之间的空格不一致：Z 世代 / Z世代、B 站 / B站

    为什么值得单列：这是人工审稿最容易挑的硬伤之一，而且跟 AIGC 无关——
    纯粹是严谨度。上一轮是在报告里手写「Z 世代/00 后统一 1 处」发现的，
    靠人眼盯全文不可靠。

    只报计数不改：句首大写、专有名词大写、品牌官方写法都可能是刻意的。
    """
    # 1. 拉丁词大小写
    forms = {}
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9&.\-]*(?:[ \t]+[A-Za-z][A-Za-z0-9&.\-]*)*", t):
        raw = m.group(0).strip()
        if len(re.sub(r"[^A-Za-z0-9]", "", raw)) < 2:
            continue
        key = re.sub(r"\s+", "", raw).lower()
        forms.setdefault(key, Counter())[raw] += 1
    case = []
    for key, c in forms.items():
        if len(c) >= 2:
            case.append({"key": key, "forms": dict(c)})
    case.sort(key=lambda x: -sum(x["forms"].values()))

    # 2. 中英之间的空格
    sp = {}
    for m in re.finditer(r"[A-Za-z]+[ \t]*[\u4e00-\u9fff]", t):
        raw = m.group(0)
        key = re.sub(r"[ \t]+", "", raw).lower()
        sp.setdefault(key, Counter())[raw] += 1
    spacing = []
    for key, c in sp.items():
        if len(c) >= 2:
            spacing.append({"key": key, "forms": dict(c)})
    spacing.sort(key=lambda x: -sum(x["forms"].values()))

    return {"case": case, "spacing": spacing}


def extract_numbers(t):
    return Counter(NUM_RE.findall(t))


def extract_sources(t):
    """机构库命中 + 匿名候选（据 X / X 显示）。"""
    known = OrderedDict()
    for s in SOURCE_LIB:
        n = len(re.findall(re.escape(s), t))
        if n:
            known[s] = known.get(s, 0) + n
    # 合并同义（前瞻 / 前瞻产业研究院 取大者，避免重复计）
    merge_pairs = [("前瞻产业研究院", "前瞻"), ("Vogue Business", "Vogue"),
                   ("Louis Vuitton", "LV"), ("哔哩哔哩", "B站"),
                   ("无畏契约", "VALORANT"), ("巴黎世家", "Balenciaga")]
    for a, b in merge_pairs:
        if a in known and b in known:
            if known[b] >= known[a]:
                known.pop(a, None)
            else:
                known.pop(b, None)

    cands = []
    for m in re.finditer(r"(?:据|根据|依据|援引)\s*([^，。；：、\n《》]{2,20}?)(?:的)?"
                         r"(?:数据|报告|研究|统计|调查|白皮书|显示|指出|表明|称|预测|认为|分析)", t):
        cands.append(m.group(1).strip())
    for m in re.finditer(r"([^，。；：、\n《》]{2,20}?)(?:的)?(?:数据)?(?:显示|指出|表明|预测|援引|报告称)", t):
        c = m.group(1).strip()
        if len(c) >= 2:
            cands.append(c)
    uniq = []
    for c in cands:
        # 候选里已经含已知来源（如「Gartner 与 IDC」）就不是匿名的，跳过——
        # 否则报告会同时把它列进「已命中来源」和「匿名候选」，自相矛盾
        if any(s in c for s in known):
            continue
        if c not in uniq:
            uniq.append(c)
    return known, uniq


# ============================================================
# 前后对比 + 守恒核验
# ============================================================

def conservation(before, after):
    nb, na = extract_numbers(before), extract_numbers(after)
    lost = {k: nb[k] - na.get(k, 0) for k in nb if na.get(k, 0) < nb[k]}
    added = {k: na[k] - nb.get(k, 0) for k in na if nb.get(k, 0) < na[k]}
    sb, ub = extract_sources(before)
    sa, ua = extract_sources(after)
    src_lost = {k: v for k, v in sb.items() if sa.get(k, 0) < v}
    src_new = {k: v for k, v in sa.items() if sb.get(k, 0) < v}
    return {
        "num_before": sum(nb.values()), "num_after": sum(na.values()),
        "num_lost": lost, "num_added": added,
        "src_before": sb, "src_after": sa,
        "src_lost": src_lost, "src_new": src_new,
        "src_anon_before": ub, "src_anon_after": ua,
    }


def strength_shift(before, after):
    """检测论点强度回退：弱断言被改成强断言。按句级对齐定位。"""
    def score(txt):
        w = sum(txt.count(x) for x in WEAK_CLAIM)
        s = sum(txt.count(x) for x in STRONG_CLAIM)
        return w, s

    wb, sb = score(before)
    wa, sa = score(after)
    sb_sents = [s for s in split_sentences(before)]
    sa_sents = [s for s in split_sentences(after)]
    flagged = []
    for s in sb_sents:
        has_weak = [x for x in WEAK_CLAIM if x in s]
        if not has_weak:
            continue
        best, ratio = None, 0.0
        for t2 in sa_sents:
            r = SequenceMatcher(None, s, t2).ratio()
            if r > ratio:
                ratio, best = r, t2
        if best and ratio > 0.45:
            got_strong = [x for x in STRONG_CLAIM if x in best and x not in s]
            still_weak = [x for x in WEAK_CLAIM if x in best]
            if got_strong and not still_weak:
                flagged.append({
                    "before": s[:80],
                    "after": best[:80],
                    "weak_lost": has_weak,
                    "strong_gained": got_strong,
                    "similarity": round(ratio, 2),
                })
    return {
        "weak_before": wb, "weak_after": wa,
        "strong_before": sb, "strong_after": sa,
        "flagged": flagged,
    }


def rewrite_ratio(before, after):
    def norm(t):
        return re.sub(r"\s+", "", t)
    A, B = norm(after), norm(before)
    if not B:
        return {}
    sm = SequenceMatcher(None, B, A, autojunk=False)
    same = sum(bl.size for bl in sm.get_matching_blocks())
    char_ratio = (1 - same / len(B)) * 100
    sb = [x.strip() for x in re.split(r"[。；！？!?]", before) if len(x.strip()) > 1]
    sa = [x.strip() for x in re.split(r"[。；！？!?]", after) if len(x.strip()) > 1]
    kept = 0
    for s in sb:
        if any(SequenceMatcher(None, s, t).ratio() > 0.8 for t in sa):
            kept += 1
    sent_ratio = (1 - kept / max(len(sb), 1)) * 100
    return {
        "char_chars_before": len(B), "char_chars_after": len(A),
        "char_same": same,
        "char_ratio": round(char_ratio, 1),
        "sent_total": len(sb), "sent_kept": kept,
        "sent_ratio": round(sent_ratio, 1),
    }


# ============================================================
# 报告渲染
# ============================================================

def strategy_for(score):
    """按原稿分数给策略。

    这些区间是经验判断，不是实证换算——没有人做过"表层改写 X% 能降 Y 分"
    的对照实验，检测器的算法也不公开。所以这里给的是**方向**，不是承诺。
    """
    if score is None:
        return None
    if score < 15:
        return ("不要改",
                "已经低于多数学校的关注线。此时改写只有下行风险："
                "把你原本写的句子改坏，或引入新的不自然。"
                "更值得做的是留存写作过程证据（版本历史、笔记时间线、原始数据）。")
    if score < 25:
        return ("有机会压到 10%",
                "这是唯一有希望压到 10% 的区间。做法：拆模板骨架、拆枚举、"
                "重排段落顺序、把概括句换成具体事实。但仍不承诺结果。")
    if score < 35:
        return ("表层最多到 15–20%",
                "骨架和模板能拆干净，但要再往下必须换叙述顺序和切入角度——"
                "那属于改论证，需要你逐段确认。")
    if score < 50:
        return ("表层最多到 25–30%",
                "说明有相当篇幅是 AI 起草的。表层改写到不了 10%，"
                "必须把 AI 起草的段落用你自己的话重讲一遍，不是换词，是换脑子。")
    if score < 60:
        return ("大概率需要重写",
                "表层手段基本无效。先确认哪些段落是你自己的数据与判断，"
                "把那部分留住，其余重写。")
    return ("必须重写内容",
            "这个分数说明主体是 AI 生成的。任何表层改写都是在跟检测器玩猫鼠，"
            "而猫每月换一次版本。唯一的出路是按你自己的研究过程重写。")


def render_baseline_block():
    """前置检查区。给了 --baseline 就出策略，没给就出警告。"""
    L = []
    if BASELINE is None:
        L.append("## 〇、前置检查：**分数未提供，本报告不下策略结论**")
        L.append("")
        L.append("> ⚠️ **没有原稿真实分数，就不知道要不要改。**")
        L.append("> 如果原稿本来就是 8%（你自己写的却被误判），改写只有下行风险。")
        L.append("> 先去测一次（腾讯朱雀 `matrix.tencent.com/ai-detect/ai_gen_txt` 有免费额度），")
        L.append("> 再把分数传进来：`--baseline 23.5`")
        L.append("")
        L.append("另外两问仍需你回答：")
        L.append("")
        L.append("- [ ] **哪些部分是你自己写的**：________（决定「必须重写」还是「只是风格差异」）")
        L.append("- [ ] **学科与文体**：________（医学/工科/人文的人类基线不同，用 `--genre` 指定）")
        L.append("")
        L.append("参考区间：<15% 不要改｜15–25% 有机会到 10%｜25–35% 到 15–20%｜"
                 "35–50% 到 25–30%｜>50% 大概率重写")
        L.append("")
        return L

    action, detail = strategy_for(BASELINE)
    L.append("## 〇、前置检查：原稿 **%.1f%%** → 建议 **%s**" % (BASELINE, action))
    L.append("")
    L.append("> %s" % detail)
    L.append("")
    L.append("| 起点 | 这一档能到哪 |")
    L.append("|---|---|")
    L.append("| <15% | 不要改，留存写作过程证据 |")
    L.append("| 15–25% | 有机会压到 10%（骨架+段落重组+具体化） |")
    L.append("| 25–35% | 表层最多到 15–20%，再往下要改叙述顺序 |")
    L.append("| 35–50% | 表层最多到 25–30%，AI 起草段落必须重讲 |")
    L.append("| >50% | 表层无效，重写内容 |")
    L.append("")
    L.append("仍需你回答：**哪些部分是你自己写的**（决定「必须重写」还是「只是风格差异」）。")
    L.append("")
    return L


def verdict_ordinal(total, per_k, clusters):
    if total == 0:
        return "无骨架"
    if per_k >= 3 or len(clusters) >= 3:
        return "**重度模板化**"
    if per_k >= 1.5 or len(clusters) >= 1:
        return "**中度，需拆**"
    return "轻度，可保留论证必需的分条"


def render_single(path, t, raw):
    paras = split_paras(t)
    hits, clusters = detect_ordinals(t)
    punct = detect_punct(t, raw)
    strong, weak = detect_assertions(t)
    vague = detect_vague_authority(t)
    rhythm = detect_para_rhythm(paras)
    cv = detect_sentence_cv(t)
    nums = extract_numbers(t)
    src, anon = extract_sources(t)
    parallel = detect_parallel(paras)
    repeats = detect_repeats(t)
    terms = detect_term_variants(t)
    # use / mention 分层要在这里算好——摘要区就要用到，不能等到第三节
    real_vague = [v for v in vague if not v.get("mention")]
    mentioned = [v for v in vague if v.get("mention")]
    unanchored = [v for v in real_vague if not v["anchored"]]
    han = han_count(t)
    per_k = round(len(hits) / max(han, 1) * 1000, 2)

    L = []
    L.append("# 预扫描报告：%s" % os.path.basename(path))
    L.append("")
    L.append("汉字 %d ／ 字符 %d ／ 段 %d ／ 句 %d" % (han, len(t.replace("\n", "")), len(paras), cv["count"]))
    L.append("")
    L.append("文体口径：**%s**（段长 %d–%d 字，%s）。用 `--genre` 切换。"
             % (CURRENT_GENRE, PARA_IDEAL[0], PARA_IDEAL[1], GENRE_PROFILES[CURRENT_GENRE]["note"]))
    L.append("")
    if han < 50:
        L.append("> ⚠️ 汉字数只有 %d，本工具的中文口径（序数骨架、顿号、句长、断言词表）基本不适用。"
                 "若为非中文文本，请改用对应语种的判据，不要套用本报告结论。" % han)
        L.append("")

    # 文本体检：所有阈值都是照学术文体定的。不是论文就别拿这份报告当真。
    sniff = sniff_text_type(t)
    if sniff["verdict"] == "unlikely":
        L.append("> ⚠️ **这不像学术论文**（体检得分 %d/8：%s）。"
                 % (sniff["score"], "，".join(sniff["reasons"])))
        L.append("> 下面所有阈值（段长 %d–%d、破折号、序数骨架）都按学术文体设定，"
                 "套在非论文文本上会大面积虚报。" % (PARA_IDEAL[0], PARA_IDEAL[1]))
        L.append("> 确认要按论文处理再往下读；否则这份报告只能当格式参考。")
        L.append("")
    elif sniff["verdict"] == "maybe":
        L.append("> 体检得分 %d/8（%s）——像论文但特征不全，"
                 "例如缺引用标记或篇幅偏短。指标可用，但段长结论要看文体是否匹配。"
                 % (sniff["score"], "，".join(sniff["reasons"])))
        L.append("")

    # 〇、前置检查：脚本管不了"要不要改"，但必须先把这个问题摆到桌面上，
    # 并且把"有没有问过分数"变成可检查的状态，而不是靠自觉
    L.extend(render_baseline_block())

    # 摘要：报告越写越长，必须让人在第一屏就知道该干什么。
    # 按可验证的严重度排序，只取前三条，避免又变成一份"什么都有"的清单。
    issues = []
    if len(hits) >= 6:
        issues.append((100, 1, "序数骨架 **%d 处**（%s）——结构模板，本文最重的痕迹"
                        % (len(hits), verdict_ordinal(len(hits), per_k, clusters).strip("*"))))
    elif len(hits) > 0:
        issues.append((40, 1, "序数骨架 %d 处，轻度" % len(hits)))
    if unanchored:
        issues.append((90, 1, "借来的权威 **%d 处**无具体归属——答辩会被直接追问" % len(unanchored)))
    if parallel:
        issues.append((85, 1, "首句同构 **%d 组**——拆了旧模板又造了新模板" % len(parallel)))
    if repeats:
        issues.append((80, 1, "稿内重复论证 **%d 对**——同一断言写了两遍" % len(repeats)))
    if strong:
        issues.append((60, 1, "强信号断言词 **%d 处**——把判断说成了事实"
                        % sum(n for _, n in strong)))
    if terms["case"] or terms["spacing"]:
        issues.append((30, 3, "术语写法不一致 **%d 组**——与 AIGC 无关，但审稿最容易看见"
                        % (len(terms["case"]) + len(terms["spacing"]))))
    if rhythm and rhythm["median"] < PARA_IDEAL[0]:
        issues.append((50, 2, "段长中位 **%s 字**，低于 %s 口径 %d–%d"
                        % (rhythm["median"], CURRENT_GENRE, PARA_IDEAL[0], PARA_IDEAL[1])))
    if punct["破折号（——）"]["per_k"] >= 1.5:
        issues.append((45, 2, "破折号 **%.2f/千字**（人类 0.80）" % punct["破折号（——）"]["per_k"]))
    if punct["冒号-正文"]["per_k"] >= 2:
        issues.append((44, 2, "正文冒号 **%.2f/千字**，疑似「标签：注脚」" % punct["冒号-正文"]["per_k"]))
    issues.sort(key=lambda x: (-x[0]))

    # 摘要分区：并列十项等于没有优先级。按"这处痕迹有多确定"分三级——
    # T1 是检测器和人眼都认的结构模板，T2 要结合文体判断，T3 跟 AIGC 根本无关。
    if issues:
        L.append("## 摘要（先看这里）")
        L.append("")
        tier_meta = {
            1: ("T1 结构模板", "**先改这一级**，检测器和人眼都认，改完再往下看"),
            2: ("T2 节奏与标点", "改完 T1 再评估，不要和 T1 同时动手——节奏改坏了不容易看出来"),
            3: ("T3 仅严谨度", "与 AIGC 分数无关，但导师和审稿人最容易看见"),
        }
        for tier in (1, 2, 3):
            group = [t for _, ti, t in issues if ti == tier]
            if not group:
                continue
            name, note = tier_meta[tier]
            L.append("**%s**（%d 项）— %s" % (name, len(group), note))
            L.append("")
            for txt in group:
                L.append("- %s" % txt)
            L.append("")
        L.append("完整数据见下文；改法与不可改项见「判读提示」。")
        L.append("")
    else:
        L.append("## 摘要")
        L.append("")
        L.append("未发现结构性痕迹。**注意不要为改而改**——本脚本报不出来，"
                 "不代表检测器会给低分（它算的是困惑度，不是这些形态）。")
        L.append("")

    # 一、序数骨架
    L.append("## 一、中文序数骨架（主流中文检测工具的共同盲区）")
    L.append("")
    L.append("命中 **%d 处**，密度 %.2f /千字，判定：%s" % (len(hits), per_k, verdict_ordinal(len(hits), per_k, clusters)))
    L.append("")
    if hits:
        L.append("| # | 类型 | 命中 | 上下文 |")
        L.append("|---|---|---|---|")
        for i, h in enumerate(hits[:40], 1):
            L.append("| %d | %s | `%s` | %s |" % (i, h["kind"], h["text"], h["context"][:56].replace("|", "/")))
        if len(hits) > 40:
            L.append("| … | 其余 %d 处略 | | |" % (len(hits) - 40))
        L.append("")
    if clusters:
        L.append("**同段多连（模板骨架的强证据）%d 段：**" % len(clusters))
        L.append("")
        for c in clusters:
            L.append("- 第 %d 段，%d 处（%s）：%s…" % (c["para_index"], c["count"], "、".join(c["kinds"]), c["preview"]))
        L.append("")

    # 二、标点
    L.append("## 二、标点密度")
    L.append("")
    L.append("| 指标 | 本篇 | 每千字 | 参照 |")
    L.append("|---|---|---|---|")
    for k, v in punct.items():
        pk = "—" if not v["per_k"] else "%.3f" % v["per_k"]
        L.append("| %s | %d | %s | %s |" % (k, v["count"], pk, v["ref"]))
    L.append("")

    # 三、断言与归因
    L.append("## 三、断言与归因")
    L.append("")
    if strong:
        L.append("**强信号断言词（学术文本中一律可疑，需逐条看是否把判断说成了事实）：**")
        L.append("")
        L.append("　" + "　".join("%s×%d" % (w, n) for w, n in sorted(strong, key=lambda x: -x[1])))
        L.append("")
    else:
        L.append("**强信号断言词：0 处。**")
        L.append("")
    if weak:
        L.append("弱信号（需看上下文，不构成必改）：")
        L.append("")
        L.append("　" + "　".join("%s×%d" % (w, n) for w, n in sorted(weak, key=lambda x: -x[1])))
        L.append("")
    if real_vague:
        L.append("**借来的权威（真正在使用：%d 处，其中 %d 处无具体归属）：**" % (len(real_vague), len(unanchored)))
        L.append("")
        for v in real_vague:
            mark = "❌ 无归属" if not v["anchored"] else "✅ 有归属"
            L.append("- %s `%s` — %s" % (mark, v["text"], v["context"][:70]))
        L.append("")
    if mentioned:
        L.append("**仅在提及／引用（不计风险，共 %d 处）：**" % len(mentioned))
        L.append("")
        for v in mentioned:
            L.append("- `%s` —%s" % (v["text"], v["mention"]))
        L.append("")
    if not real_vague and not mentioned:
        L.append("**借来的权威：0 处。**")
        L.append("")

    # 四、段落与节奏
    L.append("## 四、段落与节奏")
    L.append("")
    if rhythm:
        L.append("| 项 | 值 |")
        L.append("|---|---|")
        L.append("| 段数 | %d（正文 %d ＋ 标题 %d） |" % (rhythm["count"], rhythm["body_count"], rhythm["count"] - rhythm["body_count"]))
        L.append("| 段长中位 | %s 字（%s 口径 %d–%d） |" % (rhythm["median"], CURRENT_GENRE, PARA_IDEAL[0], PARA_IDEAL[1]))
        L.append("| 段长均值 / 极值 | %s / %s–%s |" % (rhythm["mean"], rhythm["min"], rhythm["max"]))
        L.append("| < %d 字的段 | %s |" % (PARA_TOO_SHORT, ("%d 段（第 %s 段）" % (len(rhythm["too_short"]), "、".join(map(str, rhythm["too_short"][:12])))) if rhythm["too_short"] else "无"))
        L.append("| > %d 字的段 | %s |" % (PARA_TOO_LONG, ("%d 段（第 %s 段）" % (len(rhythm["too_long"]), "、".join(map(str, rhythm["too_long"][:12])))) if rhythm["too_long"] else "无"))
        L.append("| 句长 CV | %s（人类约 0.54 / AI 约 0.46，仅同工具前后可比） |" % (cv["cv"] if cv["cv"] is not None else "样本过少"))
        L.append("")
        if rhythm["sym_runs"]:
            L.append("**连续段长度相近（偏差 <20%%，≥3 段）：%d 组**" % len(rhythm["sym_runs"]))
            L.append("")
            for r in rhythm["sym_runs"]:
                L.append("- 从第 %d 段起 %d 段，长度 %s：%s…" % (r["start_para"], r["count"], "/".join(map(str, r["lengths"])), r["preview"]))
            L.append("")
        else:
            L.append("连续段长度相近：**0 组**")
            L.append("")

    # 五、同构与重复（改写最容易新引入的两类痕迹）+ 术语一致性
    L.append("## 五、同构、重复与术语一致性")
    L.append("")
    if parallel:
        L.append("**⚠️ 首句同构：连续 %d 组段落的首句长度相近、句数相同。** 这通常是拆掉旧编号模板后又造出的新模板。" % len(parallel))
        L.append("")
        for r in parallel:
            L.append("- 从第 %d 段起 %d 段，首句 %s 字、每段 %d 句：" % (r["start_para"], r["count"], "/".join(map(str, r["first_lens"])), r["n_sent"]))
            for f in r["firsts"]:
                L.append("  - %s…" % f)
        L.append("")
    else:
        L.append("首句同构：**0 组**")
        L.append("")
    if repeats:
        L.append("**⚠️ 稿内重复论证：%d 对句子相似度 ≥0.72。** 同一断言写了两遍，读者会觉得在凑字数。" % len(repeats))
        L.append("")
        for p in repeats:
            L.append("- （相似度 %.2f）" % p["sim"])
            L.append("  - A：%s" % p["a"])
            L.append("  - B：%s" % p["b"])
        L.append("")
    else:
        L.append("稿内重复论证：**0 对**")
        L.append("")

    # 术语一致性：同一术语的写法必须统一，这是纯严谨度问题，与 AIGC 无关
    if terms["case"] or terms["spacing"]:
        if terms["case"]:
            L.append("**⚠️ 同一术语大小写不一致：%d 组。**" % len(terms["case"]))
            L.append("")
            for g in terms["case"][:6]:
                L.append("- " + " ／ ".join("`%s` ×%d" % (k, v) for k, v in g["forms"].items()))
            L.append("")
        if terms["spacing"]:
            L.append("**⚠️ 中英文之间空格不一致：%d 组。**" % len(terms["spacing"]))
            L.append("")
            for g in terms["spacing"][:6]:
                L.append("- " + " ／ ".join("`%s` ×%d" % (k, v) for k, v in g["forms"].items()))
            L.append("")
    else:
        L.append("术语写法一致性：**无不一致**")
        L.append("")

    # 六、守恒基准
    L.append("## 六、守恒基准（改写后必须逐项核对）")
    L.append("")
    L.append("**数字 token：%d 个**" % sum(nums.values()))
    L.append("")
    L.append("　" + "　".join("%s×%d" % (k, v) if v > 1 else k for k, v in sorted(nums.items(), key=lambda x: -x[1])[:40]))
    if len(nums) > 40:
        L.append("　（另有 %d 种未列出）" % (len(nums) - 40))
    L.append("")
    L.append("**来源清单：%d 个**" % len(src))
    L.append("")
    L.append("　" + "　".join("%s×%d" % (k, v) for k, v in src.items()))
    L.append("")
    if anon:
        L.append("匿名候选（脚本抽出的“据 X / X 显示”，需人工确认是否为独立来源）：")
        L.append("")
        L.append("　" + " / ".join(anon[:20]))
        L.append("")

    # 七、判读提示
    L.append("## 七、判读提示（脚本不下结论，以下为需人工确认项）")
    L.append("")
    tips = []
    if len(hits) >= 6:
        tips.append("序数骨架 %d 处是本文最重的痕迹。**保留分条逻辑、只拆条内的统一微模板**——学术写作需要可检索结构，不要为了避骨架把论点拆散。" % len(hits))
    if strong:
        tips.append("强信号断言词需逐条核：改成陈述事实或补依据，不要只是换个词。")
    if unanchored:
        tips.append("%d 处借来的权威无具体归属，答辩会被直接追问，必须补来源或改为自己的推断。" % len(unanchored))
    if parallel:
        tips.append("首句同构 %d 组——把其中一段的句式彻底换掉（名词短语开头／把论断后置／改成让步句），只改长度没用。" % len(parallel))
    if repeats:
        tips.append("重复论证 %d 对——同一断言只保留一处完整论证，另一处改为回指，不要重讲一遍。" % len(repeats))
    if terms["case"] or terms["spacing"]:
        tips.append("术语写法不一致 %d 组——定一个全篇统一写法（一般取首次出现的形式，"
                    "或品牌官方写法），其余全部替换。这一项与 AIGC 无关，是纯严谨度问题，"
                    "但人工审稿最容易看见。" % (len(terms["case"]) + len(terms["spacing"])))
    if rhythm and rhythm["median"] < PARA_IDEAL[0]:
        tips.append("段长中位 %s 字，低于 %s 口径下限。%s，建议合到 %d–%d 字。"
                    % (rhythm["median"], CURRENT_GENRE, GENRE_PROFILES[CURRENT_GENRE]["note"], PARA_IDEAL[0], PARA_IDEAL[1]))
    if rhythm and rhythm["median"] > PARA_IDEAL[1]:
        tips.append("段长中位 %s 字，超过 %s 口径上限，建议按论点切分。" % (rhythm["median"], CURRENT_GENRE))
    if punct["冒号-正文"]["per_k"] >= 2:
        tips.append("正文冒号密集，检查是否为「标签：注脚」型——把标签还原成句子主语。")
    if punct["破折号（——）"]["per_k"] >= 1.5:
        tips.append("破折号 %.2f/千字，高于人类 0.80。若功能一致（都是「论断——数据」），改写成独立句。" % punct["破折号（——）"]["per_k"])
    if not tips:
        tips.append("本次未发现结构性痕迹，注意不要为改而改。")
    for x in tips:
        L.append("- " + x)
    L.append("")

    return "\n".join(L)


EMPTY_RHYTHM = {"count": 0, "body_count": 0, "median": 0, "mean": 0, "min": 0,
                "max": 0, "too_short": [], "too_long": [], "sym_runs": [], "ideal_range": (0, 0)}


def render_diff(before_path, before, after_path, after, base_path=None, base=None):
    tb, ta = strip_markdown(before), strip_markdown(after)
    hb, cb = detect_ordinals(tb)
    ha, ca = detect_ordinals(ta)
    pb, pa = detect_punct(tb, before), detect_punct(ta, after)
    sb, wkb = detect_assertions(tb)
    sa, wka = detect_assertions(ta)
    vb, va = detect_vague_authority(tb), detect_vague_authority(ta)
    rb = detect_para_rhythm(split_paras(tb)) or dict(EMPTY_RHYTHM)
    ra = detect_para_rhythm(split_paras(ta)) or dict(EMPTY_RHYTHM)
    cvb, cva = detect_sentence_cv(tb), detect_sentence_cv(ta)
    cons = conservation(tb, ta)
    strength = strength_shift(tb, ta)
    ratio = rewrite_ratio(tb, ta)
    cover = detect_coverage(tb, ta)
    par_a = detect_parallel(split_paras(ta))
    rep_a = detect_repeats(ta)
    flips = detect_connector_flip(tb, ta)
    tb_terms = detect_term_variants(tb)
    ta_terms = detect_term_variants(ta)
    vb_real = [v for v in vb if not v.get("mention")]
    va_real = [v for v in va if not v.get("mention")]
    hanb, hana = han_count(tb), han_count(ta)

    L = []
    L.append("# 前后对比与守恒核验")
    L.append("")
    L.append("原稿 `%s` → 改后 `%s`" % (os.path.basename(before_path), os.path.basename(after_path)))
    L.append("")
    if base is not None:
        L.append("本轮起点 `%s`（`--base`）。下表「本轮」= 起点→改后，「累计」= 最初原稿→改后。"
                 % os.path.basename(base_path))
        L.append("")
        L.append("> 为什么分两条：第二轮返修时，只看累计会重复报第一轮已解决的问题，"
                 "看不到这一轮改了什么；只看本轮，又发现不了「改了三十处、净收益为零」。")
        L.append("")
    L.extend(render_baseline_block())

    # 摘要：diff 报告的核心只有一件事——有没有改坏。先给结论，再给数据。
    num_ok = not cons["num_lost"] and not cons["num_added"]
    src_ok = not cons["src_lost"] and not cons["src_new"]
    num_msg = "零丢失、零新增" if num_ok else "；".join(
        part for part in (
            "丢失 %s" % cons["num_lost"] if cons["num_lost"] else "",
            "新增 %s" % cons["num_added"] if cons["num_added"] else "",
        ) if part)
    src_msg = "零丢失、零新增" if src_ok else "；".join(
        part for part in (
            "丢失 %s" % cons["src_lost"] if cons["src_lost"] else "",
            "新增 %s" % cons["src_new"] if cons["src_new"] else "",
        ) if part)
    checks = [
        ("数字守恒", num_ok, num_msg),
        ("来源守恒", src_ok, src_msg),
        ("论证覆盖", not cover["missing"],
         "%d 段全部有对应" % cover["body_before"] if not cover["missing"]
         else "%d 段找不到对应" % len(cover["missing"])),
        ("论点强度", not strength["flagged"],
         "未检出回退" if not strength["flagged"]
         else "%d 处被改强" % len(strength["flagged"])),
        ("逻辑方向", not flips,
         "因果/转折关系未被改反" if not flips
         else "%d 段逻辑词方向翻转" % len(flips)),
    ]
    L.append("## 摘要")
    L.append("")
    for name, ok, msg in checks:
        L.append("- %s **%s**：%s" % ("✅" if ok else "❌", name, msg))
    L.append("")
    bad = [c for c in checks if not c[1]]
    if bad:
        L.append("**%d 项未通过，先看第二节再决定是否采用这版改后稿。**" % len(bad))
    else:
        L.append("**五项守恒全部通过。** 但守恒只保证「没丢东西」，"
                 "不保证「改得更好」——表达质量和检测分数都不在这一页的检查范围内。")
    L.append("")

    if base is not None:
        # 本轮 vs 累计：两个基准各回答一个问题。
        # 本轮答「这一轮改对了吗」，累计答「整条链路有没有跑偏」。
        tbase = strip_markdown(base)
        hb2, cb2 = detect_ordinals(tbase)
        pb2 = detect_punct(tbase, base)
        rb2 = detect_para_rhythm(split_paras(tbase)) or dict(EMPTY_RHYTHM)
        sb2, _ = detect_assertions(tbase)
        L.append("## 本轮变化（%s → 改后）" % os.path.basename(base_path))
        L.append("")
        L.append("| 指标 | 本轮起点 | 改后 | 本轮变化 | 累计变化 |")
        L.append("|---|---|---|---|---|")
        rows = [
            ("汉字", han_count(tbase), hana, "%+d"),
            ("段数", rb2["count"], ra["count"], "%+d"),
            ("段长中位", rb2["median"], ra["median"], "%+g"),
            ("序数骨架", len(hb2), len(ha), "%+d"),
            ("破折号/千字", pb2["破折号（——）"]["per_k"], pa["破折号（——）"]["per_k"], "%+.2f"),
            ("强断言词", sum(n for _, n in sb2), sum(n for _, n in sa), "%+d"),
        ]
        for name, v0, v1, fmt in rows:
            delta = v1 - v0
            cumulative = v1 - ({"汉字": hanb, "段数": rb["count"], "段长中位": rb["median"],
                                "序数骨架": len(hb), "破折号/千字": pb["破折号（——）"]["per_k"],
                                "强断言词": sum(n for _, n in sb)}[name])
            L.append("| %s | %s | %s | %s | %s |"
                     % (name, v0, v1, (fmt % delta) if delta else "0",
                        (fmt % cumulative) if cumulative else "0"))
        L.append("")
        # 本轮净收益就是「本轮起点骨架数 - 改后骨架数」。
        # 旧公式又减去累计改善量，会把本轮真实的减少误报成负数。
        round_gain = len(hb2) - len(ha)
        if tbase != ta:
            if round_gain > 0:
                L.append("> **本轮净收益为 +%d 处**：序数骨架从 %d 处降至 %d 处。"
                         "这只是结构指标的变化，不代表检测分数会同幅变化。"
                         % (round_gain, len(hb2), len(ha)))
            elif round_gain < 0:
                L.append("> **本轮净收益为 %d 处**：序数骨架从 %d 处增至 %d 处，出现回退，"
                         "应先检查本轮新增的骨架，而不是继续改措辞。"
                         % (round_gain, len(hb2), len(ha)))
            else:
                L.append("> **本轮净收益为 0 处**：文本变了，但序数骨架没有减少。"
                         "说明本轮改的是节奏或措辞，不是结构模板——别指望这一项带动分数变化。")
            L.append("")

    L.append("## 一、指标变化")
    L.append("")
    L.append("| 指标 | 原稿 | 改后 | 变化 |")
    L.append("|---|---|---|---|")
    L.append("| 汉字 | %d | %d | %+d |" % (hanb, hana, hana - hanb))
    L.append("| 段数 | %d | %d | %+d |" % (rb["count"], ra["count"], ra["count"] - rb["count"]))
    L.append("| 段长中位 | %s | %s | %+g |" % (rb["median"], ra["median"], ra["median"] - rb["median"]))
    L.append("| 序数骨架 | %d | %d | %+d |" % (len(hb), len(ha), len(ha) - len(hb)))
    L.append("| 破折号/千字 | %.2f | %.2f | %+.2f |" % (pb["破折号（——）"]["per_k"], pa["破折号（——）"]["per_k"], pa["破折号（——）"]["per_k"] - pb["破折号（——）"]["per_k"]))
    L.append("| 正文冒号 | %d | %d | %+d |" % (pb["冒号-正文"]["count"], pa["冒号-正文"]["count"], pa["冒号-正文"]["count"] - pb["冒号-正文"]["count"]))
    L.append("| 顿号/千字 | %.2f | %.2f | %+.2f |" % (pb["顿号（、）"]["per_k"], pa["顿号（、）"]["per_k"], pa["顿号（、）"]["per_k"] - pb["顿号（、）"]["per_k"]))
    L.append("| 分号/千字 | %.2f | %.2f | %+.2f |" % (pb["分号（；）"]["per_k"], pa["分号（；）"]["per_k"], pa["分号（；）"]["per_k"] - pb["分号（；）"]["per_k"]))
    L.append("| 强断言词 | %d | %d | %+d |" % (sum(n for _, n in sb), sum(n for _, n in sa), sum(n for _, n in sa) - sum(n for _, n in sb)))
    L.append("| 借来的权威 | %d | %d | %+d |" % (len(vb_real), len(va_real), len(va_real) - len(vb_real)))
    L.append("| 句长 CV | %s | %s | |" % (cvb["cv"], cva["cv"]))
    L.append("| 连续段对称组 | %d | %d | %+d |" % (len(rb["sym_runs"]), len(ra["sym_runs"]), len(ra["sym_runs"]) - len(rb["sym_runs"])))
    L.append("")
    L.append("**改写率**：字符级 %.1f%%（%d→%d 字，%d 个字符未动）／句级 %.1f%%（%d 句中 %d 句基本原样）"
             % (ratio.get("char_ratio", 0), ratio.get("char_chars_before", 0), ratio.get("char_chars_after", 0),
                ratio.get("char_same", 0), ratio.get("sent_ratio", 0), ratio.get("sent_total", 0), ratio.get("sent_kept", 0)))
    L.append("")

    L.append("## 二、守恒核验")
    L.append("")
    L.append("### 数字 token")
    L.append("")
    ok_num = not cons["num_lost"] and not cons["num_added"]
    L.append("原稿 %d 个 → 改后 %d 个。%s" % (cons["num_before"], cons["num_after"], "**✅ 零丢失、零新增**" if ok_num else "**⚠️ 有差异**"))
    L.append("")
    if cons["num_lost"]:
        L.append("**丢失/缺额：** " + "；".join("%s（少 %d）" % (k, v) for k, v in cons["num_lost"].items()))
        L.append("")
    if cons["num_added"]:
        L.append("**新增：** " + "；".join("%s（多 %d）" % (k, v) for k, v in cons["num_added"].items()))
        L.append("")

    L.append("### 来源清单")
    L.append("")
    ok_src = not cons["src_lost"] and not cons["src_new"]
    L.append("原稿 %d 个 → 改后 %d 个。%s" % (len(cons["src_before"]), len(cons["src_after"]), "**✅ 全部在位**" if ok_src else "**⚠️ 有差异**"))
    L.append("")
    if cons["src_lost"]:
        L.append("**丢失：** " + "；".join("%s（%d→%d）" % (k, v, cons["src_after"].get(k, 0)) for k, v in cons["src_lost"].items()))
        L.append("")
    if cons["src_new"]:
        L.append("**新增：** " + "；".join("%s（%d→%d）" % (k, cons["src_before"].get(k, 0), v) for k, v in cons["src_new"].items()))
        L.append("")

    L.append("### 论证覆盖（防整段悄悄消失）")
    L.append("")
    if cover["missing"]:
        L.append("**⚠️ 原稿有 %d 个正文段在改后稿中找不到足够对应（命中率 <60%%），可能有论证被整段删掉：**" % len(cover["missing"]))
        L.append("")
        for m in cover["missing"]:
            L.append("- 原稿第 %d 段（%d 句，命中率 %.0f%%）：%s…" % (m["para"], m["n_sent"], m["rate"] * 100, m["preview"]))
            for l in m["lost"]:
                L.append("  - 未找到对应（最相似仅 %.2f）：%s" % (l["best"], l["sent"]))
        L.append("")
    else:
        L.append("**✅ 原稿 %d 个正文段全部在改后稿中找到对应。** 数字与来源没变、论证也没丢。" % cover["body_before"])
        L.append("")

    L.append("### 论点强度")
    L.append("")
    L.append("弱断言 %d→%d，强断言 %d→%d。" % (strength["weak_before"], strength["weak_after"], strength["strong_before"], strength["strong_after"]))
    L.append("")
    if strength["flagged"]:
        L.append("**⚠️ 疑似把弱断言改成了强断言（%d 处）：**" % len(strength["flagged"]))
        L.append("")
        for f in strength["flagged"]:
            L.append("- 原：%s" % f["before"])
            L.append("  改：%s" % f["after"])
            L.append("  （失去 %s，获得 %s，相似度 %.2f）" % ("、".join(f["weak_lost"]), "、".join(f["strong_gained"]), f["similarity"]))
            L.append("")
    else:
        L.append("**✅ 未检出论点强度回退。**")
        L.append("")

    L.append("### 逻辑方向（防意思被改反）")
    L.append("")
    if flips:
        L.append("**⚠️ %d 段出现逻辑连接词方向翻转**——句子还在、数字也对，但因果/转折关系可能反了：" % len(flips))
        L.append("")
        for f in flips:
            L.append("- 原稿第 %d 段是**%s**关系，改后第 %d 段变成**%s**（段落相似度 %.2f）" % (f["para"], f["from"], f["to_para"], f["to"], f["sim"]))
            L.append("  - 原：%s…" % f["before"])
            L.append("  - 改：%s…" % f["after"])
        L.append("")
        L.append("> 这一项**必须人工确认**。有时是把「因为 A 所以 B」改成「B 的原因在 A」，"
                 "属于改写而非反转，可以放行；但如果原来是否定、改后是肯定（或反之），必须回退。")
        L.append("")
    else:
        L.append("**✅ 未检出逻辑方向翻转。** 原稿的因果、转折、递进关系在改后稿中保持同向。")
        L.append("")
        L.append("> 说明边界：本项只比对连接词类型（因果/转折/递进/让步/条件），"
                 "**认不出不带连接词的语义反转**。后者只能靠通读，脚本管不了。")
        L.append("")

    L.append("## 三、回退检查（改写最容易引入的新问题）")
    L.append("")
    regress = []
    if len(ra["sym_runs"]) > len(rb["sym_runs"]):
        for r in ra["sym_runs"]:
            regress.append("连续 %d 段长度相近，从第 %d 段起（长度 %s）：%s…" % (r["count"], r["start_para"], "/".join(map(str, r["lengths"])), r["preview"]))
    if ra["median"] < PARA_IDEAL[0] and ra["median"] < rb["median"]:
        regress.append("段长中位从 %s 降到 %s 字，低于 %s 口径下限 %d–%d"
                       % (rb["median"], ra["median"], CURRENT_GENRE, PARA_IDEAL[0], PARA_IDEAL[1]))
    if cvb["cv"] and cva["cv"] and cva["cv"] < cvb["cv"] - 0.04:
        regress.append("句长 CV 从 %s 降到 %s，节奏趋匀" % (cvb["cv"], cva["cv"]))
    if pa["顿号（、）"]["per_k"] > pb["顿号（、）"]["per_k"] * 1.8:
        regress.append("顿号密度从 %.2f 升至 %.2f（多半是斜杠改顿号所致，需判断是否排版更正确）" % (pb["顿号（、）"]["per_k"], pa["顿号（、）"]["per_k"]))
    if pa["分号（；）"]["per_k"] > pb["分号（；）"]["per_k"] and pa["分号（；）"]["per_k"] > 1.2:
        regress.append("分号密度升至 %.2f/千字（分号是分模型指纹指标，能换成句号就换）" % pa["分号（；）"]["per_k"])
    if par_a:
        for r in par_a:
            regress.append("首句同构：第 %d 段起 %d 段，首句 %s 字、每段 %d 句 —— 可能拆掉旧模板又造了新模板"
                           % (r["start_para"], r["count"], "/".join(map(str, r["first_lens"])), r["n_sent"]))
    if rep_a:
        regress.append("稿内重复论证 %d 对（相似度 ≥0.72），同一断言写了两遍" % len(rep_a))
    n_tb = len(tb_terms["case"]) + len(tb_terms["spacing"])
    n_ta = len(ta_terms["case"]) + len(ta_terms["spacing"])
    if n_ta > n_tb:
        regress.append("术语写法不一致从 %d 组增至 %d 组——改写时新引入了写法不统一"
                       % (n_tb, n_ta))
    if regress:
        for x in regress:
            L.append("- ⚠️ " + x)
    else:
        L.append("- ✅ 未检出回退。")
    L.append("")

    L.append("## 四、仍需人工判断的")
    L.append("")
    todo = []
    if ha:
        L.append("**改后仍有 %d 处序数骨架：**" % len(ha))
        for h in ha[:10]:
            L.append("- `%s`（%s）：%s…" % (h["text"], h["kind"], h["context"][:50]))
        L.append("")
    un = [v for v in va_real if not v["anchored"]]
    if un:
        todo.append("%d 处借来的权威仍无具体归属（脚本不代编，必须本人补）" % len(un))
    if cons["src_anon_after"]:
        todo.append("匿名来源候选 %d 条待确认" % len(cons["src_anon_after"]))
    if ratio.get("char_ratio", 0) < 25:
        todo.append("字符级改写率仅 %.1f%%，仍处浅层；再往下需要动论证顺序，须作者确认" % ratio.get("char_ratio", 0))
    todo.append("**真实 AIGC 分数无法由本流程给出**，需上传检测平台自测（见 reference）")
    for x in todo:
        L.append("- " + x)
    L.append("")

    return "\n".join(L)


# ============================================================
# 入口
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="中文论文文本自检：预扫描与守恒核验")
    ap.add_argument("file", nargs="?", help="待检文件（md/txt）")
    ap.add_argument("--diff", metavar="AFTER", help="改后稿，做前后对比与守恒核验")
    ap.add_argument("--base", metavar="PREV",
                    help="本轮起点（上一版改后稿）。需与 --diff 同用，"
                         "同时报「本轮变化」与「累计变化」")
    ap.add_argument("--json", metavar="OUT", help="把指标存档为 JSON")
    ap.add_argument("--txt", metavar="OUT", help="导出可直接上传检测平台的纯文本")
    ap.add_argument("--out", metavar="OUT", help="把 markdown 报告写入文件")
    ap.add_argument("--genre", default="academic", choices=list(GENRE_PROFILES.keys()),
                    help="文体，决定段落合理长度区间（默认 academic）")
    ap.add_argument("--baseline", type=float, metavar="PCT",
                    help="原稿真实 AIGC 分数（百分数，如 23.5）。不传则报告只出警告、不给策略")
    ap.add_argument("--sources", action="append", metavar="FILE",
                    help="只加载指定的来源词表文件（可多次）。默认加载 data/sources.*.txt 全部")
    ap.add_argument("--list-genres", action="store_true", help="列出支持的文体")
    ap.add_argument("--list-sources", action="store_true", help="列出当前加载的来源词表")
    ap.add_argument("--version", action="store_true", help="显示版本号")
    args = ap.parse_args()

    if args.version:
        print("zh-paper-audit %s（schema %d）" % (__version__, SCHEMA_VERSION))
        return

    if args.list_genres:
        for k, v in GENRE_PROFILES.items():
            print("%-10s 段长 %d–%d 字   %s" % (k, v["para"][0], v["para"][1], v["note"]))
        return

    if args.list_sources:
        if args.sources:
            try:
                for p in args.sources:
                    if not os.path.isfile(p):
                        print("词表不存在：%s" % p, file=sys.stderr)
                        sys.exit(1)
                load_source_lib(only=args.sources)
            except OSError as e:
                print("词表读取失败：%s" % e, file=sys.stderr)
                sys.exit(1)
        print("当前来源词表共 %d 条（内置 %d + 外部文件）："
              % (len(SOURCE_LIB), len(BUILTIN_SOURCES)))
        print("  " + "、".join(SOURCE_LIB[:40]) + ("…" if len(SOURCE_LIB) > 40 else ""))
        return

    if args.base and not args.diff:
        ap.error("--base 是「本轮起点」，必须与 --diff 改后稿成对使用；不带 --diff 时请删掉 --base")
    if not args.file:
        ap.error("缺少待检文件（或在 --list-genres/--list-sources/--version 里选一个模式）")

    global PARA_IDEAL, PARA_TOO_SHORT, PARA_TOO_LONG, CURRENT_GENRE, BASELINE
    g = GENRE_PROFILES[args.genre]["para"]
    PARA_IDEAL[:] = list(g)
    PARA_TOO_SHORT = int(g[0] * 0.67)
    PARA_TOO_LONG = int(g[1] * 1.43)
    CURRENT_GENRE = args.genre
    BASELINE = args.baseline
    if BASELINE is not None and not (0 <= BASELINE <= 100):
        print("--baseline 应为 0–100 的百分数，收到 %s" % BASELINE, file=sys.stderr)
        sys.exit(2)

    for label, path in (("输入稿", args.file), ("改后稿", args.diff), ("本轮起点", args.base)):
        if path and not os.path.isfile(path):
            print("%s不存在：%s" % (label, path), file=sys.stderr)
            sys.exit(1)

    if args.diff and same_path(args.file, args.diff):
        print("改后稿与输入稿是同一个文件：%s（用单稿模式即可）" % args.diff, file=sys.stderr)
        sys.exit(2)
    if args.diff and args.base and same_path(args.base, args.diff):
        print("--base（本轮起点）与 --diff（改后稿）是同一个文件，本轮没有变化" % args.base,
              file=sys.stderr)
        sys.exit(2)
    if args.base and same_path(args.base, args.file):
        print("--base 与最初原稿是同一个文件，删掉 --base 即可（本轮=累计）", file=sys.stderr)
        sys.exit(2)

    try:
        check_path_collision(args)
    except ValueError as e:
        print("输出路径冲突：%s" % e, file=sys.stderr)
        sys.exit(2)

    if args.sources:
        try:
            for p in args.sources:
                if not os.path.isfile(p):
                    print("词表不存在：%s" % p, file=sys.stderr)
                    sys.exit(1)
            load_source_lib(only=args.sources)
        except OSError as e:
            print("词表读取失败：%s" % e, file=sys.stderr)
            sys.exit(1)

    try:
        raw = read_text(args.file)
    except UnicodeDecodeError as e:
        print("文件无法解码（UTF-8/UTF-16/GB18030 都失败）：%s" % e, file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print("文件读取失败：%s" % e, file=sys.stderr)
        sys.exit(1)
    body = strip_markdown(raw)

    if args.txt:
        try:
            write_output(args.txt, body)
        except OSError as e:
            print("纯文本导出失败：%s" % e, file=sys.stderr)
            sys.exit(1)
        print("已导出检测用纯文本：%s（%d 字符 / %d 段）"
              % (args.txt, len(body), len(split_paras(body))))

    if args.diff:
        try:
            raw2 = read_text(args.diff)
        except (OSError, UnicodeDecodeError) as e:
            print("改后稿读取失败：%s" % e, file=sys.stderr)
            sys.exit(1)
        base_raw, base_path = None, None
        if args.base:
            try:
                base_raw = read_text(args.base)
            except (OSError, UnicodeDecodeError) as e:
                print("本轮起点读取失败：%s" % e, file=sys.stderr)
                sys.exit(1)
            base_path = args.base
        report = render_diff(args.file, raw, args.diff, raw2,
                             base_path=base_path, base=base_raw)
    else:
        report = render_single(args.file, body, raw)

    if args.json:
        data = {
            "tool_version": __version__,
            "schema_version": SCHEMA_VERSION,
            "file": os.path.abspath(args.file),
            "genre": CURRENT_GENRE,
            "thresholds": {
                "para_ideal": list(PARA_IDEAL),
                "coverage_keep": 0.45,
                "coverage_alarm": 0.60,
                "repeat_sim": 0.72,
            },
            "source_lib_size": len(SOURCE_LIB),
            "han": han_count(body),
            "chars": len(body.replace("\n", "")),
            "paras": len(split_paras(body)),
            "ordinals": len(detect_ordinals(body)[0]),
            "punct": detect_punct(body, raw),
            "strong_assert": detect_assertions(body)[0],
            "vague_authority": detect_vague_authority(body),
            "rhythm": detect_para_rhythm(split_paras(body)),
            "sentence_cv": detect_sentence_cv(body),
            "parallel": detect_parallel(split_paras(body)),
            "repeats": detect_repeats(body),
            "term_variants": detect_term_variants(body),
            "sniff": sniff_text_type(body),
            "numbers": dict(extract_numbers(body)),
            "sources": extract_sources(body)[0],
            "baseline": BASELINE,
            "strategy": (strategy_for(BASELINE) or (None, None))[0],
        }
        if args.diff:
            b2 = strip_markdown(raw2)
            data["conservation"] = conservation(body, b2)
            data["strength_shift"] = strength_shift(body, b2)
            data["rewrite_ratio"] = rewrite_ratio(body, b2)
            data["coverage"] = detect_coverage(body, b2)
            data["connector_flip"] = detect_connector_flip(body, b2)
            if base_raw is not None:
                b3 = strip_markdown(base_raw)
                data["base"] = os.path.abspath(base_path)
                data["round_delta"] = {
                    "ordinals": len(detect_ordinals(b3)[0]) - len(detect_ordinals(b2)[0]),
                    "han": han_count(b3) - han_count(b2),
                }
        try:
            write_output(args.json, json.dumps(data, ensure_ascii=False, indent=2))
        except OSError as e:
            print("指标存档失败：%s" % e, file=sys.stderr)
            sys.exit(1)
        print("指标已存档：%s" % args.json)

    if args.out:
        try:
            write_output(args.out, report)
        except OSError as e:
            print("报告写入失败：%s" % e, file=sys.stderr)
            sys.exit(1)
        print("报告已写入：%s" % args.out)
    else:
        print(report)


if __name__ == "__main__":
    main()
