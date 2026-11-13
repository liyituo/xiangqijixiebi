# -*- coding: utf-8 -*-
"""
统一自然语言指令意图解析与安全冲突仲裁器 (Instruction Parser)
覆盖：急停/否定句/只读查询、中文相邻工位识别、来源区与目的区分离、
显式指令 vs 模型解析冲突仲裁、批量数量与速度约束、bbox 校验。
"""

import re
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple


VALID_STATIONS = ("NORMAL", "HOME", "P1", "P2", "P3", "P4", "P5", "SCAN")

_CN_DIGIT_MAP = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}

_STOP_RE = re.compile(
    r"(急停|紧急停止|立刻停止|马上停止|立刻停|马上停|停止运动|停止动作|停止执行|"
    r"停下来|停下|别动|不要动|暂停|\be[-_\s]?stop\b|\bemergency\s*stop\b|\bstop\b)",
    re.IGNORECASE,
)

_NEGATED_MOTION_RE = re.compile(
    r"(不要|(?<![识分])别|禁止|切勿|不能|不可|取消)\s*(?:去|再|继续|把.*?)?"
    r"(抓|夹|拿|取|搬|移动|移|放|碰|点动|运行|执行|靠近|去)",
    re.IGNORECASE,
)

_QUERY_RE = re.compile(
    r"(有几个|多少个|几个|多少|在哪里|在哪|什么颜色|看到什么|有什么|是什么|"
    r"数一数|统计一下|查看一下|检查一下|状态如何|在不在|有没有)",
    re.IGNORECASE,
)

_ACTION_VERB_RE = re.compile(
    r"(抓|夹取|夹起|拿|取|搬|移动|移到|放到|放在|放回|放至|送到|去|回到|打开|张开|闭合|关闭|扫描|环视)",
    re.IGNORECASE,
)

_DEST_TRIGGER_RE = re.compile(
    r"(?:放到|放在|放至|放回|移动到|移到|移至|搬到|搬至|送到|送至|拿到|抓到|去到|回到|前往|去)"
)
_SOURCE_TRIGGER_RE = re.compile(r"(?:从|由|把)")


def is_stop_command(text: str) -> bool:
    """识别显式急停/停止指令（高优先级拦截）。"""
    if not text or not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if stripped in ("停", "停!", "停！"):
        return True
    return bool(_STOP_RE.search(stripped))


def extract_station(text: str) -> Optional[str]:
    """
    提取字符串中的标准工位名称（修复中文相邻下 \\b 失效问题，如 '放到P2点' / '去2号工位'）。
    """
    if not text:
        return None
    upper = text.upper()

    # 匹配 P1..P5（避免误匹配 P10、IP1 等：前后不能是字母或数字）
    m_p = re.search(r"(?<![A-Z0-9])P([1-5])(?![0-9])", upper)
    if m_p:
        return f"P{m_p.group(1)}"

    m_num = re.search(r"(?:工位\s*([1-5])|([1-5])\s*号?\s*工位)", upper)
    if m_num:
        idx = m_num.group(1) or m_num.group(2)
        return f"P{idx}"

    m_cn = re.search(r"(?:工位\s*([一二三四五])|([一二三四五])\s*号?\s*工位)", text)
    if m_cn:
        ch = m_cn.group(1) or m_cn.group(2)
        return f"P{_CN_DIGIT_MAP[ch]}"

    if re.search(r"(?<![A-Z])SCAN(?![A-Z])|扫描位|俯视位|高空观察位", upper):
        return "SCAN"
    if re.search(r"(?<![A-Z])HOME(?![A-Z])|(?<![A-Z])NORMAL(?![A-Z])|待机位|复位位|原点|零位|归位|回待机|初始位姿", upper):
        return "NORMAL"

    return None


def _extract_region_phrase(clause: str) -> Optional[str]:
    """从分句中提取工位名或语义区域（如 棋盘 / 左上角 / 右侧 / 旁边）。"""
    if not clause:
        return None
    station = extract_station(clause)
    if station:
        return station
    m = re.search(
        r"(左上角|右上角|左下角|右下角|左上|右上|左下|右下|左侧|右侧|左边|右边|左上方|右上方|左下方|右下方|"
        r"棋盘内|棋盘外|棋盘上|棋盘|工作台中央|工作台中心|中心|中央|中间|旁边|旁|桌边|托盘|废料区|成品区)",
        clause,
    )
    if m:
        return m.group(1)
    return None


def split_source_and_destination(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    区分来源区域 (source_region) 与目标位置 (destination)。
    未显式指定目的地时返回 destination=None，绝不静默填充默认工位。
    """
    if not text:
        return None, None

    source_region: Optional[str] = None
    destination: Optional[str] = None

    dest_matches = list(_DEST_TRIGGER_RE.finditer(text))
    if dest_matches:
        last_dest = dest_matches[-1]
        dest_clause = text[last_dest.end():]
        destination = _extract_region_phrase(dest_clause)
        prefix = text[:last_dest.start()]
        src_match = _SOURCE_TRIGGER_RE.search(prefix)
        if src_match:
            source_region = _extract_region_phrase(prefix[src_match.end():])
        else:
            source_region = _extract_region_phrase(prefix)
    else:
        src_match = _SOURCE_TRIGGER_RE.search(text)
        if src_match:
            source_region = _extract_region_phrase(text[src_match.end():])
        else:
            # 只有孤立工位号或“去XX”以外的普通引用时，若无目标触发词，判断是否是单纯指向目的地（例如 “工位 P2”）
            st = extract_station(text)
            if st and not re.search(r"(的|里|上|中)\s*(?:棋子|方块|物体|目标)", text):
                destination = st

    if destination and source_region == destination and not dest_matches:
        source_region = None

    return source_region, destination


def extract_speed(text: str) -> Optional[int]:
    """提取显式速度设定并限制到安全区间 [5, 80]。"""
    if not text:
        return None
    m = re.search(r"速度\s*(?:调到|调至|改为|设为|为|到)?\s*(\d+)", text)
    if m:
        return max(5, min(80, int(m.group(1))))
    if "慢速" in text or "缓慢" in text or "轻一点" in text:
        return 15
    if "中速" in text:
        return 30
    if "快速" in text or "快一点" in text:
        return 55
    return None


def extract_count_and_batch(text: str) -> Tuple[Optional[int], bool]:
    """提取抓取数量或全量批处理意图。"""
    if not text:
        return None, False
    if re.search(r"(所有|全部|全都|都抓|一一|逐个|一个个|全部清理)", text):
        return None, True
    m_digit = re.search(r"(\d+)\s*(?:个|颗|块|枚|只|次)", text)
    if m_digit:
        return max(1, int(m_digit.group(1))), int(m_digit.group(1)) > 1
    m_cn = re.search(r"([一二两三四五六七八九十])\s*(?:个|颗|块|枚|只|次)", text)
    if m_cn:
        n = _CN_DIGIT_MAP[m_cn.group(1)]
        return n, n > 1
    return None, False


def validate_bbox(bbox: Any) -> Tuple[Optional[List[float]], Optional[str]]:
    """严格校验归一化检测框 [ymin, xmin, ymax, xmax] ∈ [0, 1000] 且 min < max。"""
    if bbox is None:
        return None, None
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None, "bbox 长度必须为 4"
    vals: List[float] = []
    for v in bbox:
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            return None, "bbox 必须全为有限数字"
        vals.append(float(v))
    ymin, xmin, ymax, xmax = vals
    if not (0.0 <= ymin <= 1000.0 and 0.0 <= xmin <= 1000.0 and 0.0 <= ymax <= 1000.0 and 0.0 <= xmax <= 1000.0):
        return None, f"bbox 超出 [0, 1000] 范围: {vals!r}"
    if ymax <= ymin or xmax <= xmin:
        return None, f"bbox 上下左右界倒置或面积为零: {vals!r}"
    return [round(v, 2) for v in vals], None


def _extract_target_phrase(text: str) -> Optional[str]:
    """从自然语言中抽取目标物体描述。"""
    if not text:
        return None
    m = re.search(
        r"(?:把|将|抓取|抓起|抓|夹取|夹起|夹|拿起|拿|取|搬走|搬运|搬)?\s*"
        r"(?:[这那]?\s*(?:所有|全部|\d+\s*[个颗块枚只]|[一二两三四五]\s*[个颗块枚只])?)?"
        r"((?:红|黑|蓝|绿|黄|白|橙|紫)?(?:色|方)?(?:的)?(?:中国象棋|象棋|棋子|车|马|炮|相|象|士|仕|帅|将|兵|卒|方块|积木|物块|物体|零件|小球|目标)[A-Za-z0-9\u4e00-\u9fa5]{0,4})",
        text,
    )
    if m:
        cand = m.group(1).strip()
        if cand and cand not in ("工位", "位置", "点"):
            return cand
    return None


@dataclass
class ParsedIntent:
    intent: str
    raw_text: str
    target: Optional[str] = None
    source_region: Optional[str] = None
    destination: Optional[str] = None
    count: Optional[int] = None
    batch_all: bool = False
    speed: Optional[int] = None
    model_bbox: Optional[List[float]] = None
    conflicts: List[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "raw_text": self.raw_text,
            "target": self.target,
            "source_region": self.source_region,
            "destination": self.destination,
            "count": self.count,
            "batch_all": self.batch_all,
            "speed": self.speed,
            "model_bbox": self.model_bbox,
            "conflicts": list(self.conflicts),
            "reason": self.reason,
        }


def parse_instruction(text: str, model_result: Optional[Dict[str, Any]] = None) -> ParsedIntent:
    """
    统一自然语言意图解析器：
    1. 优先识别急停、否定约束与只读查询，避免误触发运动；
    2. 显式提取工位/来源区/目的区/数量/速度；
    3. 与大模型结构化输出合并时遵守“显式约束 > 规则 > 模型猜测”，并记录冲突。
    """
    raw = (text or "").strip()
    if not raw:
        return ParsedIntent(intent="UNKNOWN", raw_text=raw, reason="空指令")

    if is_stop_command(raw):
        return ParsedIntent(intent="STOP", raw_text=raw, reason="命中显式急停指令")

    neg_match = _NEGATED_MOTION_RE.search(raw)
    if neg_match:
        return ParsedIntent(
            intent="NEGATED",
            raw_text=raw,
            target=_extract_target_phrase(raw),
            reason=f"命中否定约束 ('{neg_match.group(0)}')，已拦截动作执行",
        )

    has_query = bool(_QUERY_RE.search(raw))
    has_action = bool(_ACTION_VERB_RE.search(raw))
    if has_query and not has_action:
        src, _ = split_source_and_destination(raw)
        return ParsedIntent(
            intent="QUERY",
            raw_text=raw,
            target=_extract_target_phrase(raw),
            source_region=src,
            reason="只读查询指令，不执行物理运动",
        )

    explicit_src, explicit_dest = split_source_and_destination(raw)
    explicit_speed = extract_speed(raw)
    explicit_count, explicit_batch_all = extract_count_and_batch(raw)
    explicit_target = _extract_target_phrase(raw)
    transfer = re.search(
        r"(?:把|将|识别|检测|定位|找到|寻找)\s*(.+?)\s*"
        r"(?:[，,]\s*)?(?:并且|并|然后|再)?\s*" + _DEST_TRIGGER_RE.pattern,
        raw,
    )
    robot_target = transfer and re.fullmatch(
        r"(?:机械臂|机器人|手臂|夹爪|末端(?:执行器)?)"
        r"(?:\s*(?:慢慢|缓慢|平稳|直接|安全|匀速|慢速|快速)(?:地)?)?",
        transfer.group(1).strip(),
    )

    # 基础动作规则判别
    if re.search(r"(全景扫描|环视扫描|扫描工作台|扫描桌面|查看桌面)", raw):
        base_intent = "SCAN"
    elif re.search(r"(张开夹爪|打开夹爪|松开夹爪|放开夹爪)", raw):
        base_intent = "GRIPPER_OPEN"
    elif re.search(r"(闭合夹爪|夹紧夹爪|关闭夹爪|合上夹爪)", raw):
        base_intent = "GRIPPER_CLOSE"
    elif re.search(r"(回待机|回到待机|回到\s*NORMAL|回到\s*HOME|回零|复位|归位)", raw, re.IGNORECASE):
        base_intent = "MOVE_TO_STATION"
        if not explicit_dest:
            explicit_dest = "NORMAL"
    elif explicit_dest and transfer:
        # 物品搬运句式优先于工位号兜底，机械臂本体移动按工位动作执行。
        base_intent = "MOVE_TO_STATION" if robot_target else "PICK_AND_PLACE"
    elif re.search(r"(去|前往|移动到|移至|回到)\s*(?:工位|[一二三四五]号工位|P[1-5]|NORMAL|HOME|SCAN)", raw, re.IGNORECASE) and not re.search(r"(抓|夹|拿|搬|棋子|方块|物体|积木)", raw):
        base_intent = "MOVE_TO_STATION"
        if not explicit_dest:
            explicit_dest = extract_station(raw)
    elif explicit_dest and re.search(r"(抓|夹|拿|取|搬|放)", raw):
        base_intent = "PICK_AND_PLACE"
    elif re.search(r"(抓|夹|拿|取|搬)", raw):
        base_intent = "PICK"
    elif re.search(r"(放到|放在|放回|放置)", raw) and explicit_dest:
        base_intent = "PLACE"
    elif explicit_dest in VALID_STATIONS:
        base_intent = "MOVE_TO_STATION"
    else:
        base_intent = "UNKNOWN"

    conflicts: List[str] = []
    final_target = explicit_target
    final_dest = explicit_dest
    final_src = explicit_src
    final_speed = explicit_speed
    final_count = explicit_count
    final_batch = explicit_batch_all
    final_bbox: Optional[List[float]] = None
    reason = ""

    if isinstance(model_result, dict):
        arguments = model_result.get("arguments")
        if isinstance(arguments, dict):
            model_result = {
                **model_result,
                "target": arguments.get("target_object", model_result.get("target")),
                "destination": arguments.get("target_station", model_result.get("destination")),
                "bbox": arguments.get("normalized_bbox", model_result.get("bbox")),
            }
        # 1. 目标裁决
        m_target = model_result.get("target") or model_result.get("object")
        if m_target and isinstance(m_target, str):
            m_target = m_target.strip()
            if final_target and m_target not in final_target and final_target not in m_target:
                conflicts.append(f"target 冲突: 显式='{final_target}', 模型='{m_target}' (采用显式)")
            elif not final_target:
                final_target = m_target

        # 2. 目的地裁决 (显式目的地优先级绝对高于模型输出)
        m_dest_raw = model_result.get("destination") or model_result.get("station")
        m_dest = extract_station(str(m_dest_raw)) if m_dest_raw else None
        if not m_dest and isinstance(m_dest_raw, str) and m_dest_raw.strip():
            m_dest = m_dest_raw.strip()
        if final_dest and m_dest and final_dest != m_dest:
            conflicts.append(f"destination 冲突: 显式='{final_dest}', 模型='{m_dest}' (采用显式)")
        elif not final_dest and m_dest:
            # 若原句从某处抓取而未指定放到哪，仅当模型给出明确目的地且文本确有放置意图时才采纳
            if base_intent in ("PICK_AND_PLACE", "PLACE", "MOVE_TO_STATION"):
                final_dest = m_dest

        # 3. bbox 校验
        raw_bbox = model_result.get("bbox")
        if raw_bbox is not None:
            valid_bb, bb_err = validate_bbox(raw_bbox)
            if bb_err:
                conflicts.append(f"模型 bbox 无效 ({bb_err})，已忽略")
            else:
                final_bbox = valid_bb

        # 4. 速度与数量补充
        if final_speed is None and model_result.get("speed") is not None:
            try:
                final_speed = max(5, min(80, int(model_result["speed"])))
            except (TypeError, ValueError):
                conflicts.append(f"模型 speed 非法: {model_result.get('speed')!r}")
        if final_count is None and model_result.get("count") is not None:
            try:
                final_count = max(1, int(model_result["count"]))
                if final_count > 1:
                    final_batch = True
            except (TypeError, ValueError):
                pass
        if not final_src and isinstance(model_result.get("source_region"), str):
            final_src = model_result["source_region"].strip() or None

        # 5. 模型动作补全 (仅在规则未能判定具体动作时采纳)
        m_action = str(model_result.get("action") or model_result.get("intent") or "").upper()
        if base_intent == "UNKNOWN" and m_action in (
            "PICK", "PLACE", "PICK_AND_PLACE", "MOVE_TO_STATION",
            "GRIPPER_OPEN", "GRIPPER_CLOSE", "SCAN",
        ):
            base_intent = m_action

    return ParsedIntent(
        intent=base_intent,
        raw_text=raw,
        target=final_target,
        source_region=final_src,
        destination=final_dest,
        count=final_count,
        batch_all=final_batch,
        speed=final_speed,
        model_bbox=final_bbox,
        conflicts=conflicts,
        reason=reason or f"解析意图={base_intent}",
    )
