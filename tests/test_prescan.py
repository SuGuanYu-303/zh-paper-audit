#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prescan.py 回归测试套件

为什么需要这个文件
------------------
上一轮"五问自查"里做了四组对抗测试，发现了四个真漏洞并修掉。
但那些测试是**手工敲命令跑的**，致命问题在于：下次改一行正则，
把之前修好的行为弄坏，没有人会知道。

这个文件把那些测试固化下来。改完 prescan.py，先跑：

    cd <skill_dir> && python3 tests/test_prescan.py

全绿才算没有破坏已有行为。

12 项断言，每项对应一次真实踩过的坑
-----------------------------------
 1  破折号不重复计数            坑：4 处算成 12 处
 2  序数骨架 8 类全识别          盲区：lieflat 与 aigc-reduce 同时失明
 3  标题段不污染段长统计         坑：中位数从 148 被拉低到 19
 4  数字守恒 + 日期不吃 token    坑：DATE_PROTECT 吞掉 3 个 token
 5  use / mention 区分           坑：4 处「有学者认为」全是误报
 6  论证覆盖率                   坑：删掉整段 44 字论证，守恒核验全对
 7  首句同构                     坑：拆掉「第一第二」造出三段同构
 8  稿内重复论证                 坑：同一论点写了两遍
 9  文体口径切换                 坑：工科 44 字/段被判「偏碎」
10  边界：空 / 极短 / 纯英文     要求不崩
11  幂等：两次运行结果完全相同    输出确定性的底线
12  txt 导出无 markdown 残留

设计约束
--------
- 纯标准库 unittest，不引第三方依赖（与 prescan.py 保持一致）
- 每个用例用小段合成文本，自包含，不依赖工作区里的论文
- 断言具体数值而非"大于零"，这样回归才能被真的抓到
"""

import os
import sys
import json
import tempfile
import unittest
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# 同一份测试要能跑两种布局：仓库根（prescan.py 在根目录）和技能目录（scripts/ 布局）
SCRIPT_DIR = ROOT if os.path.isfile(os.path.join(ROOT, "prescan.py")) \
    else os.path.join(ROOT, "scripts")
sys.path.insert(0, SCRIPT_DIR)

import prescan as ps  # noqa: E402


# ============================================================
# 辅助
# ============================================================

def set_genre(g):
    """切换文体口径。main() 里做的事，测试要自己复现。"""
    prof = ps.GENRE_PROFILES[g]["para"]
    ps.PARA_IDEAL[:] = list(prof)
    ps.PARA_TOO_SHORT = int(prof[0] * 0.67)
    ps.PARA_TOO_LONG = int(prof[1] * 1.43)
    ps.CURRENT_GENRE = g


def para(n):
    """生成一段约 n 个汉字的正文（含句读，避免被当成标题段）。"""
    unit = "这是用于测试的正文内容。"
    s = unit * (n // len(unit) + 1)
    return s[:n - 1] + "。"


class Base(unittest.TestCase):
    def setUp(self):
        set_genre("academic")

    def tearDown(self):
        set_genre("academic")


# ============================================================
# 1 · 破折号不重复计数
# ============================================================

class TestDashCount(Base):
    """坑：曾经用 t.count("——") + t.count("—") 把每处破折号算了 3 次，
    原稿 4 处报成 12 处，密度虚高 3 倍。"""

    def test_four_dashes_counted_as_four(self):
        t = "甲句——带破折号。乙句——又一处。丙句——第三处。丁句——第四处。"
        p = ps.detect_punct(t, t)
        self.assertEqual(p["破折号（——）"]["count"], 4)

    def test_single_em_dash_also_counted(self):
        t = "甲句—单破折号。"
        p = ps.detect_punct(t, t)
        self.assertEqual(p["破折号（——）"]["count"], 1)

    def test_no_dash(self):
        t = "完全没有破折号的句子。"
        p = ps.detect_punct(t, t)
        self.assertEqual(p["破折号（——）"]["count"], 0)


# ============================================================
# 2 · 序数骨架 8 类全识别
# ============================================================

class TestOrdinals(Base):
    """盲区：lieflat 主动放弃该规则；aigc-reduce 正则只认阿拉伯数字括号。
    带 11 处「其一/第一」的论文，两个工具都报 0 命中。"""

    CASES = [
        ("其一，这是甲。", "其一/其二"),
        ("其二，这是乙。", "其一/其二"),
        ("第一，这是甲。", "第N，(顿逗)"),
        ("第三，这是丙。", "第N，(顿逗)"),
        ("首先，要做这件事。", "首先/其次/再次/最后"),
        ("最后，总结一下。", "首先/其次/再次/最后"),
        ("一是要快。", "一是/二是"),
        ("一方面，要控制成本。", "一方面/另一方面"),
        ("（1）这是第一步。", "(N) 括号编号"),
        ("① 这是第一步。", "①②③ 圈号"),
        ("A. 这是第一条。", "行首字母序号"),
    ]

    def test_each_kind_detected(self):
        for text, kind in self.CASES:
            with self.subTest(text=text, kind=kind):
                hits, _ = ps.detect_ordinals(text)
                kinds = {h["kind"] for h in hits}
                self.assertIn(kind, kinds, "未识别：%s（期望 %s，实得 %s）" % (text, kind, kinds))

    def test_chart_caption_not_counted(self):
        """「图(1)」这类图表编号不是议论骨架，不该算。"""
        hits, _ = ps.detect_ordinals("详见图(1)所示。")
        self.assertEqual(len(hits), 0)

    def test_same_para_two_ordinals_is_cluster(self):
        """同段内 ≥2 个序数 → 模板骨架的强证据。"""
        t = "其一，甲。其二，乙。"
        _, clusters = ps.detect_ordinals(t)
        self.assertGreaterEqual(len(clusters), 1)

    def test_eleven_ordinals_in_real_paper(self):
        """真实论文基准：原稿 11 处。文件不存在则跳过（fixture 是外部文件）。"""
        fixture = os.path.join(HERE, "fixtures", "ordinal-11.md")
        if not os.path.isfile(fixture):
            self.skipTest("缺少 fixture：ordinal-11.md")
        t = ps.strip_markdown(ps.read_text(fixture))
        hits, _ = ps.detect_ordinals(t)
        self.assertEqual(len(hits), 11)


# ============================================================
# 3 · 标题段不污染段长统计
# ============================================================

class TestHeadingNotPollute(Base):
    """坑：小节标题被当成正文段（5 汉字），段长中位数从 148 被拉到 19，
    报告因此说「段落偏碎」——完全是假的。"""

    def test_heading_excluded_from_body_stats(self):
        t = "一、核心模式\n\n" + para(200)
        paras = ps.split_paras(t)
        self.assertEqual(len(paras), 2)
        r = ps.detect_para_rhythm(paras)
        self.assertEqual(r["body_count"], 1, "标题段不该计入正文段")
        expect = ps.han_count(para(200))
        self.assertEqual(r["median"], expect, "中位数应等于正文段长度，不含标题段")

    def test_short_para_with_punctuation_is_body(self):
        """有句读的短段是正文，不是标题——否则真实短段会被误排除。"""
        self.assertFalse(ps.is_heading_para("短，但有句读。"))

    def test_long_para_without_punctuation_is_heading(self):
        """无句读的长串（目录行/标题）仍判标题。"""
        self.assertTrue(ps.is_heading_para("（一）产品：功能性联名与角色资产化"))


# ============================================================
# 4 · 数字守恒
# ============================================================

class TestNumbers(Base):
    """坑：日期保护用 sub 替换成空格，把日期里的数字 token 一起吃掉了，
    前后稿同样处理所以"看起来守恒"，但绝对值少 3 个。"""

    def test_percent_and_amount_extracted(self):
        n = ps.extract_numbers("增长 44.6%，规模 170 亿欧元，份额 60%–66%。")
        self.assertIn("44.6%", n)
        self.assertIn("170", n)
        self.assertIn("60%", n, "百分号与数字属同一 token")
        self.assertIn("66%", n)

    def test_date_not_swallowed(self):
        n = ps.extract_numbers("2026-09-24 发布。")
        self.assertGreater(len(n), 0, "日期不该被整体吞掉")

    def test_conservation_detects_loss(self):
        before = "增长 44.6%，规模 170 亿。"
        after = "增长 44.6%。"
        c = ps.conservation(before, after)
        self.assertEqual(c["num_lost"], {"170": 1}, "少了 170 应被报出：%r" % (c,))

    def test_conservation_clean_when_identical(self):
        t = "增长 44.6%，规模 170 亿。"
        c = ps.conservation(t, t)
        self.assertEqual(c["num_lost"], {}, "同一文本不该报丢失")
        self.assertEqual(c["num_added"], {}, "同一文本不该报新增")


# ============================================================
# 5 · use / mention 区分
# ============================================================

class TestUseMention(Base):
    """坑：拿两份分析报告试跑，4 处「有学者认为」全被判风险。
    实际那两句是在**讨论**这个说法，不是在使用它。"""

    def test_quoting_is_mention(self):
        t = "报告里那句「有学者认为」属于没有指名的归因。"
        hits = ps.detect_vague_authority(t)
        self.assertEqual(len(hits), 1)
        self.assertIsNotNone(hits[0]["mention"], "引号内应判为提及")

    def test_meta_word_is_mention(self):
        t = "这句话的表述是：有学者认为这种符号有用。"
        hits = ps.detect_vague_authority(t)
        self.assertEqual(len(hits), 1)
        self.assertIsNotNone(hits[0]["mention"], "含元语言词应判为提及")

    def test_table_row_is_mention(self):
        t = "| 有学者认为 | 未指名 | 待补 |"
        hits = ps.detect_vague_authority(t)
        self.assertEqual(len(hits), 1)
        self.assertIsNotNone(hits[0]["mention"], "表格内应判为提及")

    def test_real_use_is_risk(self):
        t = "有学者认为这种统一符号有助于降低跨文化折扣。"
        hits = ps.detect_vague_authority(t)
        self.assertEqual(len(hits), 1)
        self.assertIsNone(hits[0]["mention"], "正文使用不该判为提及")
        self.assertFalse(hits[0]["anchored"], "无具体归属应判未锚定")

    def test_anchored_when_citation_present(self):
        t = "有学者认为这一点成立（张三，2024）。"
        hits = ps.detect_vague_authority(t)
        self.assertTrue(hits[0]["anchored"], "带年份作者应判已锚定")


# ============================================================
# 6 · 论证覆盖率
# ============================================================

class TestCoverage(Base):
    """坑：删掉改后稿一段 44 字论证，数字 56→56、标点全好，
    守恒核验毫无反应——因为那段恰好不含机构名才被来源清单抓到。
    删不含机构名的段落则完全静默。"""

    def test_deleted_para_detected(self):
        before = ("第一段论证讲的是甲方面的问题，内容比较充实。"
                  "这里继续展开，说明为什么甲方面重要。\n\n"
                  "第二段转向乙方面的讨论，用词与上一段完全不重叠。"
                  "这一段要被删掉，用来验证覆盖率能不能发现。")
        after = ("第一段论证讲的是甲方面的问题，内容比较充实。"
                 "这里继续展开，说明为什么甲方面重要。")
        cov = ps.detect_coverage(before, after)
        self.assertEqual(len(cov["missing"]), 1, "删掉一段应被报出：%r" % (cov,))
        self.assertEqual(cov["missing"][0]["para"], 2)
        self.assertLess(cov["missing"][0]["rate"], 0.6)

    def test_paraphrase_not_flagged(self):
        """正常改写（同义替换）不该误报——阈值 0.45 是实测标定的。"""
        before = "功能性眼镜加数字品牌挚友的组合，绕开了已经用旧的数字皮肤模式。"
        after = "功能性眼镜与数字品牌挚友这一组合，避开了此前被反复使用的皮肤授权套路。"
        cov = ps.detect_coverage(before, after)
        self.assertEqual(len(cov["missing"]), 0, "同义改写不该报丢失：%r" % (cov,))


# ============================================================
# 7 · 首句同构
# ============================================================

class TestParallel(Base):
    """坑：拆掉「第一、第二、第三」后，造出三段都是
    「主语+动词+了+宾语」的新模板，长度 60/55/52。"""

    def test_three_isomorphic_paras_detected(self):
        t = ("统一符号降低了跨文化折扣。后面跟一句展开说明。\n\n"
             "体验营销拆掉了奢侈品的距离感。后面也跟一句展开说明。\n\n"
             "差异化避开了同质化竞争。后面又跟一句展开说明。")
        runs = ps.detect_parallel(ps.split_paras(t))
        self.assertEqual(len(runs), 1, "三段同构应报 1 组：%r" % (runs,))
        self.assertEqual(runs[0]["count"], 3)

    def test_length_similar_without_shared_mark_not_flagged(self):
        """纯长度相近、无共享结构标记词 → 不报。学术文本这种太常见。"""
        t = ("市场规模在持续扩大。数据显示增长明显。\n\n"
             "渠道结构发生变化。线下门店占比上升。\n\n"
             "竞争格局趋于集中。头部品牌优势明显。")
        runs = ps.detect_parallel(ps.split_paras(t))
        self.assertEqual(len(runs), 0, "无共享标记词不该报：%r" % (runs,))

    def test_two_paras_not_enough(self):
        """只有两段不算模板，至少三段。"""
        t = ("符号降低了距离。展开一句。\n\n"
             "体验拆掉了门槛。展开一句。")
        runs = ps.detect_parallel(ps.split_paras(t))
        self.assertEqual(len(runs), 0)


# ============================================================
# 8 · 稿内重复论证
# ============================================================

class TestRepeats(Base):
    """坑：改后稿第 43 行与第 55 行都在讲「统一符号降低跨文化折扣」，
    两处都写着「有学者认为」——同一论点写了两遍。"""

    def test_duplicate_argument_detected(self):
        t = ("有学者认为这种统一符号有助于降低跨文化折扣，但区域差异仍需考虑。\n\n"
             "这里插入一句完全无关的过渡，用来打乱邻接关系，长度也刻意写得不一样。\n\n"
             "有学者认为，这种统一符号有助于降低跨文化折扣，不过区域差异也要考虑。")
        pairs = ps.detect_repeats(t)
        self.assertGreaterEqual(len(pairs), 1, "重复论证应被报出")
        self.assertGreaterEqual(pairs[0]["sim"], 0.72)

    def test_distinct_arguments_not_flagged(self):
        t = ("市场规模在扩大，增速保持在两位数区间。\n\n"
             "渠道结构 meanwhile 出现分化，线上占比首次超过线下。\n\n"
             "竞争格局方面，头部三家的合计份额已接近六成。")
        pairs = ps.detect_repeats(t)
        self.assertEqual(len(pairs), 0, "不同论点不该报重复：%r" % (pairs,))


# ============================================================
# 9 · 文体口径
# ============================================================

class TestGenre(Base):
    """坑：工科实验报告段长中位 44 字，被 academic 口径判「偏碎，
    建议合到 120–350 字」——这个建议对实验报告是错的。"""

    # 三个正文段各 65–80 汉字：落在 lab 口径（50–220）内，
    # 同时低于 academic 下限（120）——这才是有效的对照，
    # 让同一文本在两种口径下得到相反判定。
    LAB_TEXT = (
        "三、实验方法与结果\n\n"
        "实验采用有限元方法对试件建模。网格划分采用四面体单元，单元尺寸设为 2 mm，"
        "过渡区局部加密。边界条件按实际工况施加，约束底部节点的全部自由度。"
        "材料参数取自 GB/T 700-2006，弹性模量取 206 GPa，泊松比取 0.3。\n\n"
        "计算结果表明，最大应力出现在圆角过渡区，峰值为 235 MPa，低于材料的屈服强度"
        " 345 MPa，安全系数约为 1.47。应力集中系数随圆角半径增大而下降，"
        "与理论解的趋势一致，偏差在 5% 以内。\n\n"
        "为进一步验证模型可靠性，进行了三组重复试验。三组实测应变的变异系数分别为"
        " 3.2%、2.8% 和 3.5%，重复性良好。试验装置与加载方式见图 3，"
        "测点布置与有限元模型的取样位置一一对应。"
    )

    def test_lab_genre_does_not_flag_fragmented(self):
        set_genre("lab")
        rep = ps.render_single("lab.md", ps.strip_markdown(self.LAB_TEXT), self.LAB_TEXT)
        self.assertNotIn("低于 lab 口径下限", rep, "lab 口径不该判段长不足")

    def test_academic_genre_flags_fragmented(self):
        set_genre("academic")
        rep = ps.render_single("lab.md", ps.strip_markdown(self.LAB_TEXT), self.LAB_TEXT)
        self.assertIn("低于 academic 口径下限", rep,
                      "同一文本在 academic 口径下应判段长不足（说明口径确实生效）")

    def test_genre_profiles_have_expected_keys(self):
        for g in ("academic", "lab", "essay"):
            self.assertIn(g, ps.GENRE_PROFILES)
            lo, hi = ps.GENRE_PROFILES[g]["para"]
            self.assertLess(lo, hi)


# ============================================================
# 10 · 边界情况
# ============================================================

class TestEdgeCases(Base):
    """要求：不崩、不抛异常。报告内容可以为空或极少，但进程必须正常结束。"""

    def test_empty(self):
        rep = ps.render_single("empty.md", "", "")
        self.assertIsInstance(rep, str)

    def test_tiny(self):
        t = "一句话。"
        rep = ps.render_single("tiny.md", t, t)
        self.assertIsInstance(rep, str)

    def test_english_only_warns(self):
        t = "This is an English paper about AI writing.\n\nIt has two paragraphs only."
        rep = ps.render_single("en.md", t, t)
        self.assertIn("中文口径", rep, "非中文文本应打口径不适用警告")

    def test_empty_diff(self):
        rep = ps.render_diff("a.md", "", "b.md", "")
        self.assertIsInstance(rep, str)

    def test_heading_only(self):
        t = "一、标题\n\n二、标题"
        rep = ps.render_single("h.md", t, t)
        self.assertIsInstance(rep, str)


# ============================================================
# 11 · 幂等
# ============================================================

class TestIdempotent(Base):
    """输出确定性是这工具相对 LLM 改写工具的核心优势之一：
    同一文件跑两次，报告必须逐字相同。"""

    SAMPLE = (
        "一、市场分析\n\n"
        "其一，产品层面采用三层架构。其二，价格层面存在张力。其三，渠道层面全球标准化。\n\n"
        "有学者认为这种统一符号有助于降低跨文化折扣。据贝恩报告，2024 年增长 3%–5%。\n\n"
        "值得注意的是，电竞观看中有 44.6% 来自创作者协同直播——这个数字很关键。"
    )

    def test_render_single_twice_identical(self):
        body = ps.strip_markdown(self.SAMPLE)
        a = ps.render_single("s.md", body, self.SAMPLE)
        b = ps.render_single("s.md", body, self.SAMPLE)
        self.assertEqual(a, b)

    def test_detect_functions_twice_identical(self):
        body = ps.strip_markdown(self.SAMPLE)
        self.assertEqual(ps.detect_ordinals(body), ps.detect_ordinals(body))
        self.assertEqual(ps.detect_para_rhythm(ps.split_paras(body)),
                         ps.detect_para_rhythm(ps.split_paras(body)))
        self.assertEqual(ps.detect_vague_authority(body), ps.detect_vague_authority(body))


# ============================================================
# 13 · 前置判断：没有原稿分数就不给策略
# ============================================================

class TestBaseline(Base):
    """v1 最大的失误是不问起点就开改。把"有没有问过分数"
    从依赖自觉变成可检查的状态。"""

    SAMPLE = "一、小节\n\n" + para(120) + "\n\n" + para(130)

    def tearDown(self):
        ps.BASELINE = None
        super().tearDown()

    def test_no_baseline_warns_and_gives_no_strategy(self):
        ps.BASELINE = None
        rep = ps.render_single("s.md", ps.strip_markdown(self.SAMPLE), self.SAMPLE)
        self.assertIn("分数未提供", rep)
        self.assertIn("不下策略结论", rep)

    def test_low_baseline_advises_not_to_touch(self):
        ps.BASELINE = 8.0
        rep = ps.render_single("s.md", ps.strip_markdown(self.SAMPLE), self.SAMPLE)
        self.assertIn("不要改", rep)

    def test_mid_baseline_advises_rewrite_content(self):
        ps.BASELINE = 55.0
        rep = ps.render_single("s.md", ps.strip_markdown(self.SAMPLE), self.SAMPLE)
        self.assertIn("重写", rep)

    def test_strategy_boundaries(self):
        """区间边界必须闭合，不允许出现 None 导致渲染崩。"""
        for s in (0, 8, 14.9, 15, 24.9, 25, 34.9, 35, 49.9, 50, 59.9, 60, 100):
            with self.subTest(score=s):
                r = ps.strategy_for(s)
                self.assertIsNotNone(r, "分数 %s 无对应策略" % s)
                self.assertEqual(len(r), 2)

    def test_strategy_none_for_missing(self):
        self.assertIsNone(ps.strategy_for(None))


# ============================================================
# 14 · 来源词表外部化
# ============================================================

class TestSources(unittest.TestCase):
    """v2.1 实测发现：机构库硬编码了商科那一套，换工科全部不命中。
    改成「内置保底 + data/sources.*.txt」，换领域只加文件。"""

    def test_external_tables_loaded(self):
        self.assertGreater(len(ps.SOURCE_LIB), len(ps.BUILTIN_SOURCES),
                           "data/ 下的词表应被加载进来")

    def test_tech_domain_hits(self):
        t = "据 Gartner 与 IDC 数据，台积电在 SEMI 统计口径下份额上升。"
        known, _ = ps.extract_sources(t)
        self.assertIn("Gartner", known)
        self.assertIn("IDC", known)
        self.assertIn("台积电", known)

    def test_business_domain_still_hits(self):
        t = "据贝恩报告，2024 年眼镜品类达 170 亿欧元。"
        known, _ = ps.extract_sources(t)
        self.assertIn("贝恩", known)

    def test_anonymous_candidate_excludes_known(self):
        """已知来源不该同时被列为匿名候选——否则报告自相矛盾。"""
        t = "据 Gartner 与 IDC 数据显示，份额上升。"
        known, anon = ps.extract_sources(t)
        for c in anon:
            self.assertFalse(any(s in c for s in known),
                             "「%s」已含已知来源，不该进匿名候选" % c)

    def test_unknown_source_is_anonymous(self):
        t = "据某某咨询数据显示，份额上升。"
        known, anon = ps.extract_sources(t)
        self.assertEqual(known, {})
        self.assertTrue(anon, "完全陌生的来源应被列为匿名候选")

    def test_missing_data_dir_does_not_crash(self):
        """把 DATA_DIR 指到不存在的位置，应静默退回内置表而不是崩。"""
        old = ps.DATA_DIR
        try:
            ps.DATA_DIR = os.path.join(HERE, "__no_such_dir__")
            ps.load_source_lib()
            self.assertEqual(len(ps.SOURCE_LIB), len(ps.BUILTIN_SOURCES))
        finally:
            ps.DATA_DIR = old
            ps.load_source_lib()


# ============================================================
# 15 · 术语写法一致性
# ============================================================

class TestTermVariants(Base):
    """同一术语的写法必须统一。上一轮是靠人眼在报告里手写发现的，
    靠人眼盯全文不可靠，所以做成检测项。"""

    def test_case_variant_detected(self):
        r = ps.detect_term_variants("VALORANT 与 Valorant 是同一款游戏。")
        self.assertEqual(len(r["case"]), 1)
        self.assertIn("VALORANT", r["case"][0]["forms"])
        self.assertIn("Valorant", r["case"][0]["forms"])

    def test_spacing_variant_detected(self):
        r = ps.detect_term_variants("Z 世代玩家与Z世代观众高度重叠。")
        self.assertEqual(len(r["spacing"]), 1)

    def test_consistent_usage_not_flagged(self):
        r = ps.detect_term_variants("VALORANT 很好玩。VALORANT 是射击游戏。")
        self.assertEqual(r["case"], [], "写法一致不该报")

    def test_single_letter_not_flagged(self):
        """单字母 token（A/B/C 选项）不算术语大小写变体。"""
        r = ps.detect_term_variants("选项 A 与选项 a 不同。")
        self.assertEqual(r["case"], [])

    def test_multiword_phrase_variant(self):
        r = ps.detect_term_variants("Vogue Business 报道过。Vogue business 也提过。")
        self.assertEqual(len(r["case"]), 1)


# ============================================================
# 16 · 摘要（第一屏就要知道该干什么）
# ============================================================

class TestSummary(Base):
    """报告从 6 节涨到 8 节，不给人一屏结论就等于让人自己找重点。"""

    def test_bad_text_shows_top_issues(self):
        t = ("其一，甲。其二，乙。其三，丙。其四，丁。\n\n"
             "有学者认为这种做法很有效，但没有人给出具体出处。\n\n"
             + para(130) + "\n\n" + para(140))
        rep = ps.render_single("s.md", ps.strip_markdown(t), t)
        self.assertIn("摘要", rep)
        self.assertIn("序数骨架", rep)

    def test_clean_text_says_do_not_change_for_sake_of_it(self):
        t = para(200) + "\n\n" + para(210)
        rep = ps.render_single("s.md", ps.strip_markdown(t), t)
        self.assertIn("未发现结构性痕迹", rep)
        self.assertIn("不要为改而改", rep)

    def test_at_most_three_items(self):
        """摘要只列前三，不许变成又一份清单。"""
        t = ("其一，甲。其二，乙。其三，丙。其四，丁。其五，戊。其六，己。\n\n"
             "有学者认为成立。一般认为也成立。研究表明同样成立。\n\n"
             + para(60) + "\n\n" + para(62) + "\n\n" + para(64))
        rep = ps.render_single("s.md", ps.strip_markdown(t), t)
        blk = rep.split("## 摘要")[1].split("## 一、")[0]
        numbered = [ln for ln in blk.splitlines() if ln[:2] in ("1.", "2.", "3.", "4.")]
        self.assertLessEqual(len(numbered), 3, "摘要最多三条：%r" % (numbered,))


class TestDiffSummary(Base):
    """diff 报告的核心只有一件事：有没有改坏。五项守恒必须在第一屏。"""

    BEFORE = ("第一段论证给出一个明确判断。据贝恩报告，2024 年规模达 170 亿欧元。\n\n"
              "第二段转向另一个角度，讨论渠道结构的变化及其对定价的影响。\n\n"
              "第三段收束，指出这套模式在中国的落地条件与限制。")

    def tearDown(self):
        ps.BASELINE = None
        super().tearDown()

    def test_all_pass_on_identical(self):
        rep = ps.render_diff("a.md", self.BEFORE, "b.md", self.BEFORE)
        blk = rep.split("## 摘要")[1].split("## 一、")[0]
        self.assertEqual(blk.count("✅"), 5, "五项守恒应全绿：%s" % blk)
        self.assertIn("五项守恒全部通过", rep)

    def test_deleted_para_breaks_summary(self):
        after = ("第一段论证给出一个明确判断。据贝恩报告，2024 年规模达 170 亿欧元。\n\n"
                 "第二段转向另一个角度，讨论渠道结构的变化及其对定价的影响。")
        rep = ps.render_diff("a.md", self.BEFORE, "b.md", after)
        blk = rep.split("## 摘要")[1].split("## 一、")[0]
        self.assertIn("❌", blk, "删段应在摘要里报红")
        self.assertIn("未通过", rep)

    def test_number_loss_breaks_summary(self):
        after = self.BEFORE.replace("170 亿欧元", "较大规模")
        rep = ps.render_diff("a.md", self.BEFORE, "b.md", after)
        self.assertIn("❌", rep.split("## 摘要")[1].split("## 一、")[0])

    def test_number_addition_breaks_summary(self):
        """新增数字也可能是编造数据，不能只检查丢失。"""
        after = self.BEFORE + "\n\n第四段新增一个原稿没有的数字 99%。"
        rep = ps.render_diff("a.md", self.BEFORE, "b.md", after)
        blk = rep.split("## 摘要")[1].split("## 一、")[0]
        self.assertIn("❌ **数字守恒**", blk)
        self.assertIn("新增", blk)
        self.assertNotIn("五项守恒全部通过", blk)

    def test_source_addition_breaks_summary(self):
        """来源清单要求前后一致，新增来源必须显式进入人工核验。"""
        after = self.BEFORE + "\n\n第四段补充据 Gartner 报告得出的判断。"
        rep = ps.render_diff("a.md", self.BEFORE, "b.md", after)
        blk = rep.split("## 摘要")[1].split("## 一、")[0]
        self.assertIn("❌ **来源守恒**", blk)
        self.assertIn("新增", blk)
        self.assertNotIn("五项守恒全部通过", blk)


class TestReportCounting(Base):
    def test_strong_assertion_summary_counts_occurrences(self):
        """摘要写「处」就必须按出现次数算，不能按词种数算。"""
        text = "这一模式赋能渠道，也赋能品牌，并进一步赋能用户。"
        rep = ps.render_single("x.md", text, text)
        self.assertIn("强信号断言词 **3 处**", rep)


# ============================================================
# 12 · 纯文本导出
# ============================================================
class TestStripMarkdown(Base):
    """导出的纯文本要能直接贴进检测平台，不能带 markdown 标记。"""

    def test_heading_marks_removed(self):
        out = ps.strip_markdown("## 二级标题\n\n正文内容。")
        self.assertNotIn("##", out)
        self.assertIn("二级标题", out)

    def test_bold_removed(self):
        out = ps.strip_markdown("这是**加粗**内容。")
        self.assertNotIn("**", out)
        self.assertIn("加粗", out)

    def test_bullet_removed(self):
        out = ps.strip_markdown("- 第一条\n- 第二条")
        self.assertNotIn("- ", out)
        self.assertIn("第一条", out)

    def test_blank_lines_collapsed(self):
        out = ps.strip_markdown("甲段。\n\n\n\n乙段。")
        self.assertNotIn("\n\n\n", out)

    def test_zero_width_removed(self):
        out = ps.strip_markdown("正文\u200b含零宽\u200c字符。")
        self.assertNotIn("\u200b", out)
        self.assertNotIn("\u200c", out)


class TestConnectorFlip(Base):
    """逻辑方向翻转：句子都在、数字都对，但因果被改成转折。

    这是覆盖率的盲区——句子相似度照样很高，意思却反了。
    """

    def test_causal_to_adversative_flagged(self):
        before = "因为渠道成本高企，所以品牌选择直营。这条路径在 2023 年被验证过。"
        after = "渠道成本高企，然而品牌最终仍选择了直营，这条路径在 2023 年被验证过。"
        flips = ps.detect_connector_flip(before, after)
        self.assertEqual(len(flips), 1, "因果→转折应报 1 处：%r" % (flips,))
        self.assertEqual(flips[0]["from"], "因果")
        self.assertEqual(flips[0]["to"], "转折")

    def test_same_direction_not_flagged(self):
        """因果改成因果（换了同义词），不该报。"""
        before = "因为渠道成本高企，所以品牌选择直营。这条路径在 2023 年被验证过。"
        after = "由于渠道成本高企，因此品牌选择直营。这条路径在 2023 年被验证过。"
        self.assertEqual(ps.detect_connector_flip(before, after), [],
                         "同向替换是正常改写，不能报")

    def test_connector_removed_not_flagged(self):
        """改写时删掉冗余连接词是正常操作，不能报。"""
        before = "因为渠道成本高企，所以品牌选择直营。这条路径在 2023 年被验证过。"
        after = "渠道成本高企。品牌选择直营，这条路径在 2023 年被验证过。"
        self.assertEqual(ps.detect_connector_flip(before, after), [],
                         "连接词消失不算翻转，报了就是噪音")

    def test_unrelated_para_not_flagged(self):
        """段落相似度低于阈值时不必配对——否则会拿两段无关的话乱比。"""
        before = "因为渠道成本高企，所以品牌选择直营。这条路径在 2023 年被验证过。"
        after = "这次调研覆盖了十二个城市的门店。受访者普遍反映陈列密度过高。"
        self.assertEqual(ps.detect_connector_flip(before, after), [])

    def test_single_but_recognized(self):
        """单字「但」必须算转折——中文里它比「但是」常见得多。"""
        before = "因为渠道成本高企，所以品牌选择直营。这条路径在 2023 年被验证过。"
        after = "渠道成本高企，但品牌最终仍选择直营。这条路径在 2023 年被验证过。"
        flips = ps.detect_connector_flip(before, after)
        self.assertEqual(len(flips), 1, "单字「但」应被识别为转折：%r" % (flips,))

    def test_budan_is_not_adversative(self):
        """「不但…而且」是递进，不是转折。"""
        self.assertNotIn("转折", ps.conn_kinds("品牌不但没有退出，反而加快了开店。"))
        self.assertIn("递进", ps.conn_kinds("品牌不但没有退出，反而加快了开店。"))

    def test_existing_turn_not_flagged(self):
        """原稿本来就有转折，改写只是删掉「因为」——逻辑没变，不能报。"""
        before = "因为渠道成本高企，但品牌仍然选择直营。这条路径在 2023 年被验证过。"
        after = "渠道成本高企，但品牌仍然选择直营。这条路径在 2023 年被验证过。"
        self.assertEqual(ps.detect_connector_flip(before, after), [],
                         "原本就有转折，不该报成新引入")

    def test_flip_appears_in_diff_summary(self):
        before = "因为渠道成本高企，所以品牌选择直营。这条路径在 2023 年被验证过。"
        after = "渠道成本高企，然而品牌最终仍选择了直营，这条路径在 2023 年被验证过。"
        rep = ps.render_diff("a.md", before, "b.md", after)
        self.assertIn("逻辑方向", rep.split("## 一、")[0])
        self.assertIn("方向翻转", rep)


class TestSniffTextType(Base):
    """文本体检：不是论文就别拿学术口径的报告当真。"""

    PAPER = (
        "## 二、市场分析\n\n"
        "据贝恩公司《2024 年中国奢侈品市场报告》，2024 年内地奢侈品市场收缩约 20%"
        "（贝恩，2024）。这一变化与出境游复苏同步出现，说明消费并未消失，而是发生了转移。\n\n"
        "从渠道结构看，百货与购物中心的客流下滑更为明显；线上渠道的占比则继续抬升。"
        "奥莱业态的表现相对稳定，折扣渠道承接了部分价格敏感型需求，"
        "这也解释了为什么正价店的同店销售降幅普遍大于集团整体口径。\n\n"
        "第一，品牌直营比例上升带来的成本压力正在显现，这一点在 2023 年的财报中已有端倪。"
        "直营模式提升了对终端的控制力，同时也把租金、人力、库存三块成本压回了品牌自己身上，"
        "在客流下行周期里，这种结构的杠杆效应是反向的。\n\n"
        "第二，二级市场的价格体系受到冲击，折扣渠道的扩张削弱了正价店的定价权；"
        "这一机制在腕表品类上表现得最为清楚。腕表的保值属性一旦被打破，"
        "消费者推迟购买的动机就会增强，形成价格与需求的双向下行。\n\n"
        "第三，年轻客群的信息获取方式发生变化，社交平台成为主要入口，"
        "品牌不得不重新配置媒介预算，而这又反过来影响了传统时尚杂志的广告收入。"
        "据 QuestMobile 数据，2024 年时尚类内容的社交平台触达占比已超过六成，"
        "传统平面的到达率则持续走低，媒介采买的效率问题因此被放大。\n\n"
        "## 三、结论\n\n"
        "综合来看，2024 年的收缩是结构性的，而非周期性波动。"
        "需求端的分化、渠道端的成本刚性、以及信息端的迁移，三者在同一时间窗口叠加，"
        "使得过去依赖涨价与开店的双轮驱动失效。这一点值得在后续章节继续观察，"
        "也需要用更细的品类数据来验证，而不是停留在总量层面的判断。"
    )

    def test_paper_is_academic(self):
        r = ps.sniff_text_type(self.PAPER)
        self.assertEqual(r["verdict"], "academic", "论文应判 academic：%r" % (r,))
        self.assertGreaterEqual(r["cites"], 2, "引用标记应被识别")

    def test_short_text_is_unlikely(self):
        r = ps.sniff_text_type("今天开了个会，讨论了一下下周的安排。")
        self.assertEqual(r["verdict"], "unlikely")

    def test_plain_prose_is_not_academic(self):
        r = ps.sniff_text_type("天气很好。我们去公园走了走。回来的路上买了点水果。")
        self.assertEqual(r["verdict"], "unlikely")

    def test_warning_in_report_when_unlikely(self):
        rep = ps.render_single("x.md", "记个备忘。明天记得交表。", "记个备忘。明天记得交表。")
        self.assertIn("这不像学术论文", rep)

    def test_no_warning_for_real_paper(self):
        rep = ps.render_single("x.md", self.PAPER, self.PAPER)
        self.assertNotIn("这不像学术论文", rep)


class TestTierAndBase(Base):
    """分级与双轨对比：报告要能回答「先改哪个」「这一轮改对了吗」。"""

    def test_tier_grouping_in_summary(self):
        text = ("第一，渠道成本在上升。第二，价格体系受到冲击。第三，客群结构改变。"
                "据贝恩报告，2024 年规模达 170 亿欧元，同比下滑 20%；"
                "另有数据显示，线上占比升至 35%。这一段的目的是把三个骨架都放进来。")
        rep = ps.render_single("x.md", text, text)
        self.assertIn("T1 结构模板", rep)
        self.assertIn("先改这一级", rep, "T1 必须明确标出优先级")

    def test_tier_order_is_t1_first(self):
        """T1 必须排在 T2 前面，否则分级没有意义。"""
        text = ("第一，渠道成本在上升。第二，价格体系受到冲击。第三，客群结构改变。"
                "据贝恩报告，2024 年规模达 170 亿欧元，同比下滑 20%；"
                "另有数据显示，线上占比升至 35%。这一段的目的是把三个骨架都放进来。")
        blk = ps.render_single("x.md", text, text).split("## 摘要")[1].split("## 一、")[0]
        self.assertLess(blk.index("T1"), blk.index("T2"), "T1 应在 T2 之前：%s" % blk)

    def test_terms_go_to_tier3(self):
        text = ("VALORANT 的赛事体系在 2023 年扩张。据 Newzoo 报告，射击品类用户达 2600 万；"
                "另有统计显示，Valorant 的观赛时长同比增长 35%。"
                "这两组数据分别来自不同机构，口径并不完全一致，需要谨慎对比。")
        rep = ps.render_single("x.md", text, text)
        blk = rep.split("## 摘要")[1].split("## 一、")[0]
        self.assertIn("T3", blk, "术语写法应归入 T3：%s" % blk)

    def test_base_adds_round_section(self):
        v0 = "第一，渠道成本上升。第二，价格体系受冲击。这是最初的稿子，用于测试累计对比。"
        v1 = "渠道成本在上升。价格体系受到冲击。这是第一轮改后的稿子，用于测试本轮对比。"
        v2 = "渠道成本正在上升，价格体系也受到冲击。这是第二轮改后的稿子，用于测试。"
        rep = ps.render_diff("v0.md", v0, "v2.md", v2, base_path="v1.md", base=v1)
        self.assertIn("本轮变化", rep)
        self.assertIn("累计变化", rep)

    def test_no_base_no_round_section(self):
        v0 = "第一，渠道成本上升。第二，价格体系受冲击。这是最初的稿子，用于测试累计对比。"
        rep = ps.render_diff("v0.md", v0, "v0.md", v0)
        self.assertNotIn("本轮变化", rep)

    def test_net_gain_note_when_no_progress(self):
        """本轮改动很大但骨架净收益为零时，要明说别指望分数动。"""
        v0 = "第一，渠道成本上升。这是最初的稿子，需要足够长度才能被当成正文段落处理。"
        v1 = "第一，渠道成本上升。这是第一轮改后的稿子，字数略有变化但骨架还在原处。"
        v2 = "第一，渠道成本在上升。这是第二轮改后的稿子，措辞调整了但骨架依旧保留着。"
        rep = ps.render_diff("v0.md", v0, "v2.md", v2, base_path="v1.md", base=v1)
        self.assertIn("净收益", rep)

    def test_round_gain_is_not_subtracted_by_cumulative_gain(self):
        """原稿 3 处、上一版 2 处、本轮 1 处：本轮净收益必须是 +1，不是 -1。"""
        v0 = "第一，甲。第二，乙。第三，丙。\n\n" + ("原稿论证内容。" * 20)
        v1 = "第一，甲。第二，乙。\n\n" + ("上一版论证内容。" * 20)
        v2 = "第二，乙。\n\n" + ("本轮论证内容。" * 20)
        rep = ps.render_diff("v0.md", v0, "v2.md", v2, base_path="v1.md", base=v1)
        self.assertIn("本轮净收益为 +1 处", rep)
        self.assertNotIn("本轮净收益为 -1 处", rep)


class TestCliSafety(unittest.TestCase):
    """CLI 层的安全边界：错误参数要报错，输出不能覆盖输入，编码不能静默乱码。"""

    SCRIPT = os.path.abspath(ps.__file__)

    def run_cli(self, *args):
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        return subprocess.run(
            [sys.executable, self.SCRIPT] + list(args),
            capture_output=True, text=True, encoding="utf-8", env=env,
        )

    def test_list_genres_does_not_require_fake_file(self):
        r = self.run_cli("--list-genres")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("academic", r.stdout)

    def test_version_does_not_require_file(self):
        r = self.run_cli("--version")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(ps.__version__, r.stdout)
        self.assertIn("schema", r.stdout)

    def test_base_without_diff_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "a.md")
            with open(src, "w", encoding="utf-8") as f:
                f.write("正文。")
            r = self.run_cli(src, "--base", src)
            self.assertEqual(r.returncode, 2, "--base 不该在无 --diff 时被静默忽略")
            self.assertIn("--diff", r.stderr)

    def test_output_cannot_overwrite_input(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "paper.md")
            original = "第一段正文。\n\n第二段正文。"
            with open(src, "w", encoding="utf-8") as f:
                f.write(original)
            r = self.run_cli(src, "--out", src)
            self.assertEqual(r.returncode, 2, "报告路径指向输入稿必须拒绝执行")
            self.assertIn("输出路径冲突", r.stderr)
            with open(src, encoding="utf-8") as f:
                self.assertEqual(f.read(), original, "输入稿绝不能被改动")

    def test_diff_same_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "same.md")
            with open(src, "w", encoding="utf-8") as f:
                f.write("正文。")
            r = self.run_cli(src, "--diff", src)
            self.assertEqual(r.returncode, 2)
            self.assertIn("同一个文件", r.stderr)

    def test_gb18030_file_is_readable(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "gb.md")
            out = os.path.join(d, "report.md")
            with open(src, "wb") as f:
                f.write("据贝恩报告，2024 年规模达 170 亿欧元。".encode("gb18030"))
            r = self.run_cli(src, "--out", out)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out, encoding="utf-8") as f:
                rep = f.read()
            self.assertNotIn("\ufffd", rep, "GB18030 文稿不能被替换成乱码")
            self.assertIn("贝恩", rep)

    def test_missing_explicit_source_file_fails_loudly(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "a.md")
            missing = os.path.join(d, "missing.txt")
            with open(src, "w", encoding="utf-8") as f:
                f.write("正文。")
            r = self.run_cli(src, "--sources", missing)
            self.assertEqual(r.returncode, 1, "显式指定的词表缺失必须报错")
            self.assertIn("词表不存在", r.stderr)

    def test_json_records_tool_and_schema_version(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "a.md")
            out = os.path.join(d, "a.json")
            with open(src, "w", encoding="utf-8") as f:
                f.write("第一段正文。据贝恩报告，2024 年规模为 170 亿欧元。\n\n第二段正文。")
            r = self.run_cli(src, "--json", out)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["tool_version"], ps.__version__)
            self.assertEqual(data["schema_version"], ps.SCHEMA_VERSION)
            self.assertIn("thresholds", data)

    def test_write_output_is_atomic_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as d:
            target = os.path.join(d, "target.md")
            r = ps.write_output(target, "正文。\n")
            self.assertEqual(r, None)
            self.assertEqual(os.listdir(d), ["target.md"],
                             "不应留下 .prescan-*.tmp 临时文件")
            with open(target, encoding="utf-8") as f:
                self.assertEqual(f.read(), "正文。\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
