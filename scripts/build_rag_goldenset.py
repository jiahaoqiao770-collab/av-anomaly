#!/usr/bin/env python3
"""从母题（seed）批量生成 RAG 检索评估集 GT。

设计：
- 26 个母题，每个母题扩 6 种受控变体（正式 / 口语 / 错误归因 / 长背景 / 黑话或陷阱 / 极简）
- 变体只改措辞，不改变 primary_knowledge_ids；改变答案的就必须拆成新母题
- 生成后自动质检：泄漏、长度、重复、主卡存在性、陷阱冲突、覆盖度
- 直接覆盖写入 data/knowledge/eval/rag_goldenset_v1.jsonl

运行：python -m scripts.build_rag_goldenset
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = PROJECT_ROOT / "data/knowledge/raw/monthly_cost_troubleshooting_knowledge.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data/knowledge/eval/rag_goldenset_v1.jsonl"

RULE_ID_PATTERN = re.compile(r"\b[AB]\d{2}\b")
KNOWLEDGE_ID_PATTERN = re.compile(r"\bK\d{3}\b")

# --------------------------------------------------------------------------
# 母题定义
# formal / chat / misattr / jargon / short 五个片段 + 可选 trap_hint
# refuse 组直接用 variants 给 6 条完整问句
# --------------------------------------------------------------------------
SEEDS: list[dict] = [
    dict(seed_id="S01", group="单卡", business="泊车", rules=["A01"],
         primary=["K001"], secondary=["K003"], trap=[],
         formal="泊车单Clip成本环比上涨，需要判断是成本侧还是产出侧导致的",
         chat="这个月泊车每单成本比上月高了不少，是成本涨了还是采集量掉了",
         misattr="泊车单Clip成本涨了，肯定是车辆运维费涨了吧",
         jargon="泊车unit clip cost环比涨了，怎么定位是分子的问题还是分母的问题",
         short="泊车单Clip成本涨了",
         note="成本侧与产出侧的二分归因，K001 的判断规则即为此二分；K003 仅在分摊比例异常时成立"),
    dict(seed_id="S02", group="单卡", business="行车", rules=["A02"],
         primary=["K001"], secondary=["K003"], trap=[],
         formal="行车单Clip成本环比上涨，想确认是分摊成本涨了还是Clip数降了",
         chat="行车那边每单成本也上去了，是不是车少了所以摊得贵",
         misattr="行车成本涨了，应该是新加了测试场地吧",
         jargon="driving unit cost 环比上行，帮我拆一下 allocated cost 和 clip count",
         short="行车单Clip成本涨了",
         note="与S01同卡不同业务，用于检验业务维度是否影响召回"),
    dict(seed_id="S03", group="单卡", business="整体", rules=["A11"],
         primary=["K002"], secondary=["K003"], trap=["K014"],
         formal="月度总成本环比上涨，需要按外包服务费、车辆运维费、硬盘邮寄费、新增场地测试费四项拆分",
         chat="这个月总成本涨了不少，钱到底花在哪了",
         misattr="总成本涨了这么多，是不是标注外包涨价了",
         jargon="monthly total cost 涨了，四项费用里哪一项贡献最大",
         short="月度总成本涨了",
         note="总成本上涨必须先按四项费用拆分，K002 是唯一覆盖该拆分的卡"),
    dict(seed_id="S04", group="单卡", business="泊车", rules=["A10"],
         primary=["K003"], secondary=["K001"], trap=[],
         formal="泊车分摊成本上涨，需要判断是总成本上涨还是工时占比变化",
         chat="泊车那边摊销过来的成本变高了，是因为总成本涨了还是泊车分的工时多了",
         misattr="泊车分摊成本涨了，肯定是总成本涨了",
         jargon="allocated cost 上升，是 total cost 变了还是 work hour 占比变了",
         short="泊车分摊成本涨了",
         note="分摊口径的归因卡，需区分总成本与工时占比两个来源"),
    dict(seed_id="S05", group="单卡", business="泊车", rules=["A07"],
         primary=["K004"], secondary=["K005"], trap=[],
         formal="泊车Clip数这个月环比掉下来了，不确定是数据没齐还是真的采得少了",
         chat="泊车这个月出的Clip数明显少了，是不是有数据没入库",
         misattr="泊车Clip数少了，肯定是采集任务砍了",
         jargon="parking clip count 下降，先看工时还是先看数据完整性",
         short="泊车Clip数降了",
         note="Clip数下降卡要求先排除数据缺失，判断顺序写死在卡里"),
    dict(seed_id="S06", group="单卡", business="泊车", rules=["A05"],
         primary=["K005"], secondary=["K004"], trap=[],
         formal="泊车每小时Clip数环比下降，需要区分是产出下降还是工时上涨",
         chat="泊车每小时能出的Clip变少了，是人等得太久还是设备停了",
         misattr="每小时Clip数下降，肯定是人手不够",
         jargon="clips per hour 掉了，分子分母分别看哪一侧",
         short="泊车每小时Clip数降了",
         note="效率卡必须同时看Clip数与工时两个方向"),
    dict(seed_id="S07", group="单卡", business="行车", rules=["A06"],
         primary=["K005"], secondary=["K004"], trap=[],
         formal="行车每小时Clip数环比下降，怀疑与设备停机或场地限制有关",
         chat="行车每小时产出也在掉，是不是最近设备老出问题",
         misattr="行车每小时Clip变少，应该是排班排得太密",
         jargon="driving clips per hour 下滑，帮忙判断是停机还是场地限制",
         short="行车每小时Clip数降了",
         note="与S06同卡不同业务，用于比较业务词替换后的召回稳定性"),
    dict(seed_id="S08", group="单卡", business="泊车", rules=["A08"],
         primary=["K006"], secondary=["K005"], trap=[],
         formal="业务工时上涨但Clip数下降，需要判断新增工时是否形成有效产出",
         chat="工时加了不少，可出来的Clip反而更少了，这些工时花哪了",
         misattr="工时涨了Clip降了，肯定是人员熟练度不行",
         jargon="work hour 上行同时 clip 下行，有没有形成有效产出",
         short="工时涨了Clip反而降了",
         note="双重恶化卡，K006 明确要求判断新增工时是否形成有效产出"),
    dict(seed_id="S09", group="单卡", business="整体", rules=["A13"],
         primary=["K007"], secondary=["K003"], trap=[],
         formal="泊车工时占比变化引起分摊成本变动，需要判断占比变化来自哪一侧",
         chat="泊车分的工时比例变大了，是泊车加了任务还是行车少了",
         misattr="泊车工时占比变大，就是泊车在扩张",
         jargon="parking work hour ratio 变化，是 parking 增加还是 driving 减少",
         short="泊车工时占比变了",
         note="占比类指标要求判断变化来自哪一侧，不能只看单侧"),
    dict(seed_id="S10", group="单卡", business="整体", rules=["A14"],
         primary=["K008"], secondary=["K001"], trap=[],
         formal="行车与泊车单Clip成本比值扩大，需要分别排查两类业务",
         chat="行车比泊车的单Clip成本差距越拉越大，是哪边出问题",
         misattr="行车泊车成本差距扩大，说明行车变差了",
         jargon="driving 与 parking 的 unit cost 比值扩大，两边分别怎么查",
         short="行车泊车成本比值拉大了",
         note="比值类指标的典型陷阱是直接归因到分子一侧，K008 明确反对"),
    dict(seed_id="S11", group="单卡", business="泊车", rules=["B02"],
         primary=["K009"], secondary=[], trap=[],
         formal="泊车有效单Clip成本与名义单Clip成本出现背离，需要判断是否为试采样本结构造成",
         chat="名义上一个Clip才十几块，摊到真能用的泊车单上就贵多了，咋回事",
         misattr="有效单成本比名义高，是不是单价涨了",
         jargon="parking effective unit cost 与 unit clip cost 背离，试采占比是不是主因",
         short="有效单成本和名义成本差挺多",
         note="B02 的唯一知识卡，名义与有效单背离即其判定条件"),
    dict(seed_id="S12", group="单卡", business="泊车", rules=["B02"],
         primary=["K009"], secondary=[], trap=[],
         formal="泊车试采占比上升，试采批次增加但未计入有效产出",
         chat="最近试采的批次多了不少，会不会把成本摊薄到正式单上了",
         misattr="试采占比高了，说明量产任务变少了",
         jargon="trial clip ratio 上升，对 effective unit cost 的影响怎么算",
         short="试采占比升了",
         note="同一张卡的第二问法，侧重点在分母结构而非背离现象本身"),
    dict(seed_id="S13", group="单卡", business="整体", rules=["*"],
         primary=["K017"], secondary=[], trap=[],
         formal="关键字段空值率环比跳升，同期成本指标恶化，需要先排除数据质量",
         chat="这个月好多记录的字段是空的，是不是采集端没上报",
         misattr="成本涨了是因为采集效率下降，不用看数据",
         jargon="空值率突增，指标异常是不是 ETL 过滤收紧导致的",
         short="字段空值率突然变高",
         note="数据质量兜底卡，判断顺序为先数据后业务"),
    dict(seed_id="S14", group="单卡", business="泊车", rules=["A07"],
         primary=["K004"], secondary=["K017"], trap=[],
         formal="泊车Clip数下降但工时稳定，怀疑部分记录未入库",
         chat="泊车的Clip数少了，但工时没变，是不是有批次没入库",
         misattr="Clip数少了肯定是业务萎缩",
         jargon="parking clip count 掉了但 work hour 没动，先查落库还是先查业务",
         short="泊车记录好像少了",
         note="Clip数异常但工时正常，是数据质量卡与Clip数卡的交叉判断点"),
    dict(seed_id="S15", group="单卡", business="整体", rules=["*"],
         primary=["K018"], secondary=[], trap=[],
         formal="某项成本指标出现台阶式跳变，怀疑计价口径变更",
         chat="这个月成本突然跳了一档，前后都挺平的，是不是换了算法",
         misattr="成本跳了一下，应该是供应商涨价了",
         jargon="指标台阶跳变，是不是含税口径改成不含税了",
         short="成本数据突然跳了一档",
         note="台阶式跳变是口径变更的形态特征，与渐变型业务变化可区分"),
    dict(seed_id="S16", group="单卡", business="整体", rules=["*"],
         primary=["K018"], secondary=[], trap=[],
         formal="同一月份的数值在不同导出快照中不一致，怀疑跨月重算",
         chat="上个月的数字我明明记过，现在再看又变了",
         misattr="数据变了说明业务真的变了",
         jargon="跨月重算导致历史值被改写，怎么固定快照时点",
         short="历史数据被改了",
         note="跨月重算与口径变更是两种不同的可比性断裂，需分开识别"),
    dict(seed_id="S17", group="多卡", business="泊车", rules=["A01", "A07"],
         primary=["K001", "K003"], secondary=["K002"], trap=[],
         formal="泊车成本上涨同时Clip数下降，需要成本侧与分摊结构同时判断",
         chat="这个月泊车成本涨了、Clip数还降了，是不是采集效率出问题了",
         misattr="成本涨Clip降，肯定是单价涨了",
         jargon="cost 上行同时 clip 下行，是效率问题还是分摊问题",
         short="泊车成本涨、Clip数还降了",
         note="成本与产出反向变动，单卡无法覆盖，需要 K001 与 K003 同时支撑"),
    dict(seed_id="S18", group="多卡", business="泊车", rules=["A05", "A07"],
         primary=["K004", "K005"], secondary=[], trap=[],
         formal="泊车Clip数下降且每小时Clip数同步下降，需要产出与效率两条路径合并判断",
         chat="泊车这个月Clip数少了，每小时出的也少了，到底哪个环节卡住",
         misattr="Clip数降就是任务少了，不用看效率",
         jargon="clip count 和 clips per hour 同时下行，两条路径先看哪条",
         short="Clip数少了，每小时产出也少了",
         note="总量与效率同时恶化，必须两条排查路径合并"),
    dict(seed_id="S19", group="多卡", business="整体", rules=["A11", "A10"],
         primary=["K002", "K003"], secondary=[], trap=[],
         formal="月度总成本上涨且业务分摊成本增速超过Clip数增速，需要先拆总成本再看分摊",
         chat="总成本在涨，摊到业务上的成本涨得比Clip数还快，这正常吗",
         misattr="总成本涨了分摊自然涨，没什么可查的",
         jargon="total cost 涨、allocated cost 增速超过 clip 增速，先拆哪一层",
         short="总成本涨得比分摊快",
         note="总成本与分摊两层同时异常，K002 拆费用、K003 看分摊结构"),
    dict(seed_id="S20", group="陷阱", business="整体", rules=["A11"],
         primary=["K002"], secondary=["K003"], trap=["K010"],
         formal="单Clip成本上涨，但采集量与单价未见变化，标注成本是否需要单独看",
         chat="成本涨了，是不是标注外包那边涨的价",
         misattr="成本涨就是标注贵了",
         jargon="unit cost 上涨，annotation 外包单价是不是主因",
         short="成本涨了",
         trap_hint="这个月成本涨这么多，会不会是标注那边调价了",
         note="问句自带标注涨价假设，K010 是该假设的专用干扰卡，正确路径是先拆总成本"),
    dict(seed_id="S21", group="陷阱", business="泊车", rules=["A10"],
         primary=["K003"], secondary=["K001"], trap=["K011"],
         formal="泊车场地相关成本上升，需要区分地图采集与数据采集",
         chat="新开了几个停车场，成本上去了，是不是地图那边花的多",
         misattr="场地成本涨了就是数据采集效率差",
         jargon="场地相关成本上升，跟 high precision map 采集的关系",
         short="场地相关成本涨了",
         trap_hint="是不是高精地图重新采了一遍所以贵了",
         note="场地与采集双关，K011 地图卡容易抢位，需区分科目归属"),
    dict(seed_id="S22", group="陷阱", business="整体", rules=["A11"],
         primary=["K002"], secondary=[], trap=["K012"],
         formal="某月出现一次性成本增加，与业务量无关",
         chat="这个月成本突然多了一笔，业务量也没变",
         misattr="成本多了一笔应该是业务扩张",
         jargon="一次性 cost 跳增，是否为 compliance 投入",
         short="冒出一笔一次性成本",
         trap_hint="是不是合规审计那笔钱",
         note="一次性投入的形态特征与常规业务变化不同，K012 为干扰项"),
    dict(seed_id="S23", group="陷阱", business="整体", rules=["A11"],
         primary=["K002"], secondary=[], trap=["K014"],
         formal="成本月度波动大与业务节奏不符，需确认费用归属月份",
         chat="这个月成本冲得很高，下个月又掉下来，是不是付款节奏的问题",
         misattr="成本波动大说明业务不稳定",
         jargon="cost 波动与业务量不同步，是不是 settlement 时点造成",
         short="成本忽高忽低",
         trap_hint="是不是供应商账期赶到季末了",
         note="账期错配会造成与业务无关的波动，K014 是最接近的干扰卡"),
    dict(seed_id="S24", group="陷阱", business="泊车", rules=["A07"],
         primary=["K004"], secondary=["K005"], trap=["K016"],
         formal="泊车Clip数下降，需确认是否与天气等外部因素相关",
         chat="泊车这个月出的少了，是不是下雨天太多没法采",
         misattr="Clip少就是天气不好",
         jargon="clip count 下降，external environment 影响占多少",
         short="泊车Clip少了",
         trap_hint="是不是那几天下雨没法采集",
         note="天气是可解释的外部因素，K016 为干扰项，但真实主因仍需先查数据完整性"),
    dict(seed_id="S25", group="拒答", business="无关", rules=[],
         primary=[], secondary=[], trap=[], refuse=True,
         variants=[
             "帮我写一个月度总结的周报模板",
             "怎么用 Python 读一个 csv 文件",
             "今天下班附近有什么好吃的推荐",
             "帮我解释一下 transformer 里的注意力机制",
             "公司附近哪儿能修电脑",
             "明天上海会下雨吗",
         ],
         note="完全无关领域的请求，应拒答并说明只服务成本效率诊断"),
    dict(seed_id="S26", group="拒答", business="无关", rules=[],
         primary=[], secondary=[], trap=[], refuse=True,
         variants=[
             "给我讲个笑话",
             "把这段英文翻译成中文：the quick brown fox jumps over the lazy dog",
             "推荐几本机器学习的入门书",
             "怎么提高睡眠质量",
             "帮我写个正则匹配手机号",
             "你觉得哪个牌子的笔记本好用",
         ],
         note="通用助手类请求，应拒答而非凑合用知识库回答"),
    dict(seed_id="S27", group="陷阱", business="整体", rules=["A11"],
         primary=["K002"], secondary=["K003"], trap=["K013"],
         formal="月度成本上涨中算力费用占比提升，需要区分训练算力与采集成本",
         chat="这个月云账单涨得挺凶，是不是训练跑多了",
         misattr="成本涨了肯定是模型训练跑太多",
         jargon="GPU 账单涨幅超过数据量增幅，属于哪一类成本",
         short="云账单涨了不少",
         trap_hint="是不是仿真和训练算力花超了",
         note="算力属于研发成本，K013 为干扰项；成本排查应先拆四项费用"),
    dict(seed_id="S28", group="陷阱", business="整体", rules=["A11"],
         primary=["K002"], secondary=["K003"], trap=["K015"],
         formal="人力成本上升但采集量未增长，需要判断是否为用工结构调整",
         chat="人力这块支出变多了，可产出的Clip没见多",
         misattr="人力成本涨了就是招人招多了",
         jargon="headcount 成本上行但产出持平，是不是外包转自有的科目转移",
         short="人力支出涨了",
         trap_hint="是不是外包转自营把人算进来了",
         note="用工结构变化可能只是科目转移，K015 为干扰项"),
]

# 陷阱组问法修订：干扰话题降为背景/猜测，问句落点是症状本身
_OVERRIDES = {
    "S20": dict(
        formal="月度总成本上涨，四项费用中需要确认哪一项贡献最大",
        chat="这个月总成本比上月高了不少，想弄清楚是外包服务费、运维费还是邮寄费拉起来的",
        misattr="总成本涨了，我猜是硬盘邮寄批次变多了",
        jargon="monthly total cost 上行，帮我按四项费用拆分定位",
        short="总成本涨了，哪项费用拉起来的",
        trap_hint="标注那边确实调过价，但那块占比不大，我想先按四项费用拆一遍确认主因",
    ),
    "S21": dict(
        formal="泊车分摊成本上涨，需要判断是总成本上涨还是工时占比变化",
        chat="泊车摊过来的成本涨了，我想知道是总成本涨了还是泊车分的工时多了",
        misattr="场地成本涨了就是数据采集效率差",
        jargon="泊车 allocated cost 上行，跟 total cost 和 work hour 占比哪个关系更大",
        short="泊车分摊成本涨了，先看哪一侧",
        trap_hint="新开了几个停车场，我怀疑是场地相关费用，但想先从分摊结构上判断到底是不是",
    ),
    "S22": dict(
        formal="某月出现一次性成本增加，与业务量无关",
        chat="这个月成本突然多了一笔，业务量也没变",
        misattr="成本多了一笔应该是业务扩张",
        jargon="一次性 cost 跳增，和业务量不同步",
        short="冒出一笔一次性成本",
        trap_hint="会不会是那笔整改投入，但我不确定，想先看这笔钱落在哪一项费用里",
    ),
    "S23": dict(
        formal="成本月度波动大与业务节奏不符，需要确认费用归属月份",
        chat="这个月成本冲得很高，下个月又掉下来，跟业务量对不上",
        misattr="成本波动大说明业务不稳定",
        jargon="cost 波动与业务量不同步，先看归属月份还是业务原因",
        short="成本忽高忽低，跟业务量对不上",
        trap_hint="付款好像赶到季末了，但我更想先确认这笔费用到底该算在哪个月",
    ),
    "S24": dict(
        formal="泊车Clip数下降但工时稳定，需要先判断是数据未入库还是真实采集减少",
        chat="泊车这个月Clip数掉得比较多，工时倒没降，我想先确认是数据没入库还是真采少了",
        misattr="Clip少就是采集任务减少了",
        jargon="parking clip count 下降、work hour 稳定，先查数据完整性还是业务",
        short="泊车Clip数少了，工时没降",
        trap_hint="最近雨挺多，但我不确定是不是这个影响，想先看是不是有数据没入库",
    ),
}

# 陷阱组问法修订 v2：修 S21 文案撞车、S28 干扰话题仍是问句焦点
_OVERRIDES_V2 = {
    "S21": dict(
        formal="泊车那部分摊过来的成本变高了，需要判断是总成本涨了还是泊车工时占比变了",
    ),
    "S28": dict(
        trap_hint="听说人力口径最近调整过，但我更想先确认这笔支出是算在外包服务费里还是人力科目里",
    ),
}

for _seed in SEEDS:
    if _seed["seed_id"] in _OVERRIDES:
        _seed.update(_OVERRIDES[_seed["seed_id"]])

for _seed in SEEDS:
    if _seed["seed_id"] in _OVERRIDES_V2:
        _seed.update(_OVERRIDES_V2[_seed["seed_id"]])

GROUP_ORDER = {"单卡": 0, "多卡": 1, "陷阱": 2, "拒答": 3}


def build_variants(seed: dict) -> list[tuple[str, str]]:
    if seed.get("refuse"):
        return [(f"V{i + 1}", text) for i, text in enumerate(seed["variants"])]
    return [
        ("V1", seed["formal"] + "？"),
        ("V2", seed["chat"] + "？"),
        ("V3", seed["misattr"] + "？"),
        ("V4", "最近在做月度复盘，" + seed["formal"] + "，想确认问题到底出在哪一步。"),
        ("V5", (seed.get("trap_hint") or seed["jargon"]) + "？"),
        ("V6", seed["short"] + "，一般先从哪儿查？"),
    ]


def load_cards() -> list[dict]:
    with KNOWLEDGE_PATH.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def char_bigrams(text: str) -> set[str]:
    clean = re.sub(r"[\s，。？！、：；,.?!:;\"'（）()]", "", text)
    return {clean[i : i + 2] for i in range(len(clean) - 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def build_rows() -> list[dict]:
    rows: list[dict] = []
    counter = 0
    for seed in sorted(SEEDS, key=lambda s: (GROUP_ORDER[s["group"]], s["seed_id"])):
        for variant_type, text in build_variants(seed):
            counter += 1
            rows.append(
                {
                    "qid": f"Q{counter:03d}",
                    "seed_id": seed["seed_id"],
                    "group": seed["group"],
                    "variant_type": variant_type,
                    "source": "template",
                    "query": text,
                    "business": seed["business"],
                    "triggered_rule_ids": seed["rules"],
                    "primary_knowledge_ids": seed["primary"],
                    "secondary_knowledge_ids": seed["secondary"],
                    "trap_knowledge_ids": seed["trap"],
                    "expect_refusal": bool(seed.get("refuse", False)),
                    "notes": seed["note"],
                }
            )
    return rows


def qc(rows: list[dict], cards: list[dict]) -> tuple[list[dict], list[str]]:
    problems: list[str] = []
    valid_ids = {c["knowledge_id"] for c in cards}
    answer_texts = {
        c["knowledge_id"]: re.sub(
            r"\s", "", "".join(c["check_steps"]) + "".join(c["decision_rules"])
        )
        for c in cards
    }
    kept: list[dict] = []
    for row in rows:
        q = row["query"]

        if RULE_ID_PATTERN.search(q) or KNOWLEDGE_ID_PATTERN.search(q):
            problems.append(f"{row['qid']} 泄漏规则或知识ID：{q}")
            continue

        cleaned = re.sub(r"[\s，。？！、：；,.?!:;]", "", q)
        limit = 120 if row["variant_type"] == "V4" else 60
        min_len = 4 if row["expect_refusal"] else 8
        if not (min_len <= len(cleaned) <= limit):
            problems.append(f"{row['qid']} 长度异常({len(cleaned)})：{q}")
            continue

        leaked = False
        for kid, body in answer_texts.items():
            for i in range(len(body) - 13):
                if body[i : i + 14] in cleaned:
                    problems.append(f"{row['qid']} 疑似抄排查步骤({kid})：{q}")
                    leaked = True
                    break
            if leaked:
                break
        if leaked:
            continue

        missing = [k for k in row["primary_knowledge_ids"] if k not in valid_ids]
        if missing:
            problems.append(f"{row['qid']} 主卡不存在：{missing}")
            continue

        conflict = set(row["primary_knowledge_ids"]) & set(row["trap_knowledge_ids"])
        if conflict:
            problems.append(f"{row['qid']} 陷阱卡与主卡冲突：{conflict}")
            continue

        if row["expect_refusal"] and row["primary_knowledge_ids"]:
            problems.append(f"{row['qid']} 拒答条却标了主卡")
            continue

        kept.append(row)

    deduped: list[dict] = []
    seen: list[tuple[str, set[str]]] = []
    for row in kept:
        bg = char_bigrams(row["query"])
        duplicate = False
        for other, other_bg in seen:
            if other["business"] != row["business"]:
                continue
            if jaccard(bg, other_bg) >= 0.88:
                problems.append(f"{row['qid']} 与 {other['qid']} 语义重复（已丢弃）")
                duplicate = True
                break
        if not duplicate:
            deduped.append(row)
            seen.append((row, bg))

    for index, row in enumerate(deduped, start=1):
        row["qid"] = f"Q{index:03d}"

    return deduped, problems


def coverage_report(rows: list[dict], cards: list[dict]) -> list[str]:
    lines: list[str] = []
    primary_hits = {k for row in rows for k in row["primary_knowledge_ids"]}
    trap_hits = {k for row in rows for k in row["trap_knowledge_ids"]}

    business_cards = [
        c["knowledge_id"]
        for c in cards
        if c["related_rule_ids"] and c["related_rule_ids"] != ["*"]
    ]
    wildcard_cards = [c["knowledge_id"] for c in cards if c["related_rule_ids"] == ["*"]]
    distractor_cards = [c["knowledge_id"] for c in cards if not c["related_rule_ids"]]

    miss_biz = [k for k in business_cards if k not in primary_hits]
    miss_wild = [k for k in wildcard_cards if k not in primary_hits]
    miss_dis = [k for k in distractor_cards if k not in trap_hits]

    lines.append(f"业务卡未被当主卡：{miss_biz or '无'}")
    lines.append(f"通配卡未被当主卡：{miss_wild or '无'}")
    lines.append(f"干扰卡未出现在陷阱：{miss_dis or '无'}")
    return lines


def main() -> None:
    cards = load_cards()
    rows = build_rows()
    raw_count = len(rows)

    rows, problems = qc(rows, cards)
    rows.sort(key=lambda r: r["qid"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    from collections import Counter

    print(f"母题数：{len(SEEDS)}")
    print(f"生成条数：{raw_count} -> 质检后 {len(rows)}")
    print(f"分组分布：{dict(Counter(r['group'] for r in rows))}")
    print(f"变体分布：{dict(sorted(Counter(r['variant_type'] for r in rows).items()))}")
    print(f"种子级 Recall 可用母题：{len({r['seed_id'] for r in rows})}")

    print("\n--- 质检问题 ---")
    if problems:
        for item in problems:
            print("  ! " + item)
    else:
        print("  无")

    print("\n--- 覆盖度 ---")
    for line in coverage_report(rows, cards):
        print("  " + line)

    print(f"\n写入：{OUTPUT_PATH}")
    print(f"行数：{sum(1 for _ in OUTPUT_PATH.open(encoding='utf-8'))}")


if __name__ == "__main__":
    main()
