"""
LPI v11 競馬予想 Streamlit アプリ
=====================================
起動方法:
  streamlit run app.py

必要なパッケージ:
  pip install streamlit pandas numpy matplotlib openpyxl
"""

import streamlit as st
import pandas as pd
import numpy as np
import math
import re
import io
from collections import Counter

# ============================================================
# ページ設定
# ============================================================
st.set_page_config(
    page_title="LPI v11 競馬予想",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 定数（変更不要）
# ============================================================
SIGMOID_CENTER = -1.755
SIGMOID_SCALE  = 2.972
GOOD_BABA      = {'良', '稍'}

FULL_VENUE = {
    '東京':'東京','中山':'中山','京都':'京都','阪神':'阪神','新潟':'新潟',
    '中京':'中京','福島':'福島','小倉':'小倉','札幌':'札幌','函館':'函館',
}
GRADE_BONUS  = {'G1':1.5,'G2':1.2,'G3':1.0,'L':0.9,'OP':0.8,'':0.7}
GRADE_WEIGHT = {'G1':4.0,'G2':3.0,'G3':2.0,'L':1.5,'OP':0.6,'':0.5}
RANK_BONUS_MULT       = {1:1.5, 2:1.2, 3:1.0}
G1_PENALTY_THRESHOLD  = 0.7
G1_PENALTY_COEF       = 0.70
WEIGHT_BASE           = 56
WEIGHT_PER_KG         = 0.2
WEIGHT_TABLE = {
    ('G1','senior'):(58,56),('G2','senior'):(57,55),('G3','senior'):(56,54),
    ('G1','3yo'):(57,55),  ('G2','3yo'):(56,54),   ('G3','3yo'):(55,53),
    ('L','senior'):(56,54),('OP','senior'):(56,54), ('','senior'):(56,54),
    ('L','3yo'):(55,53),  ('OP','3yo'):(55,53),    ('','3yo'):(55,53),
}

VENUE_ELEMENT_COEF = {
    '東京': {'基礎スピード・パワー':0.77,'ロンスパ・ギアチェンジ':1.20,
             'ギアチェンジ':1.28,'ロンスパ':0.44,'パワー・ロンスパ':1.08},
    '中山': {'基礎スピード・パワー':1.25,'ロンスパ・ギアチェンジ':1.15,
             'ギアチェンジ':0.90,'ロンスパ':1.10,'パワー・ロンスパ':1.00},
    '京都': {'基礎スピード・パワー':1.07,'ロンスパ・ギアチェンジ':1.29,
             'ギアチェンジ':1.05,'ロンスパ':0.76,'パワー・ロンスパ':0.81},
    '阪神': {'基礎スピード・パワー':1.05,'ロンスパ・ギアチェンジ':1.29,
             'ギアチェンジ':1.30,'ロンスパ':0.39,'パワー・ロンスパ':0.89},
    '中京': {'基礎スピード・パワー':1.20,'ロンスパ・ギアチェンジ':1.10,
             'ギアチェンジ':1.25,'ロンスパ':0.70,'パワー・ロンスパ':0.95},
    '新潟': {'基礎スピード・パワー':0.77,'ロンスパ・ギアチェンジ':1.02,
             'ギアチェンジ':1.41,'ロンスパ':0.26,'パワー・ロンスパ':0.87},
    '福島': {'基礎スピード・パワー':1.20,'ロンスパ・ギアチェンジ':1.15,
             'ギアチェンジ':0.90,'ロンスパ':1.05,'パワー・ロンスパ':1.00},
    '小倉': {'基礎スピード・パワー':1.10,'ロンスパ・ギアチェンジ':1.15,
             'ギアチェンジ':1.20,'ロンスパ':0.60,'パワー・ロンスパ':0.95},
    '札幌': {'基礎スピード・パワー':1.05,'ロンスパ・ギアチェンジ':1.20,
             'ギアチェンジ':1.00,'ロンスパ':1.10,'パワー・ロンスパ':1.05},
    '函館': {'基礎スピード・パワー':1.10,'ロンスパ・ギアチェンジ':1.20,
             'ギアチェンジ':0.90,'ロンスパ':1.10,'パワー・ロンスパ':1.05},
}




# ============================================================
# ペース予測テーブル（コース×距離の統計的RPCI傾向）
# 出典: 2023〜2026年全距離重賞 376レース分析
# ============================================================
PACE_TABLE = {
    (1200,'函館'): (45.9,2.5, 0,83), (1200,'中山'): (46.6,4.0, 0,57),
    (1200,'京都'): (48.3,3.1,22,56), (1200,'中京'): (49.8,1.8,10,10),
    (1200,'小倉'): (48.5,3.0,10,40), (1200,'阪神'): (47.8,3.2,10,50),
    (1400,'阪神'): (46.5,2.8,10,70), (1400,'中京'): (47.4,2.8, 0,43),
    (1400,'京都'): (49.9,2.6,22,22), (1400,'東京'): (51.7,2.3,43, 0),
    (1600,'中京'): (48.9,2.3,17,33), (1600,'中山'): (49.9,2.3,22,28),
    (1600,'京都'): (51.2,3.8,44,19), (1600,'東京'): (51.4,3.1,57,14),
    (1600,'阪神'): (52.8,3.8,52,14), (1600,'新潟'): (54.4,2.6,83, 0),
    (1800,'小倉'): (46.5,2.7, 0,60), (1800,'札幌'): (50.8,2.3,50,17),
    (1800,'中山'): (50.9,4.2,31,19), (1800,'福島'): (51.4,1.6,29, 0),
    (1800,'阪神'): (52.7,4.2,43,14), (1800,'東京'): (54.7,4.1,72, 6),
    (2000,'小倉'): (48.8,3.5,20,40), (2000,'阪神'): (49.7,1.9,12,12),
    (2000,'福島'): (50.1,4.8,33,50), (2000,'京都'): (51.5,4.9,44,22),
    (2000,'中山'): (51.8,3.4,55,23), (2000,'中京'): (52.4,5.0,55,18),
    (2000,'新潟'): (55.7,2.7,86, 0), (2000,'東京'): (56.0,4.0,86, 0),
    (2200,'中山'): (52.0,3.7,60,30), (2200,'京都'): (56.2,4.2,82, 0),
    (2400,'京都'): (52.9,3.5,60, 0), (2400,'東京'): (54.5,3.9,79, 7),
    (2500,'中山'): (53.5,3.5,65,10),
}


def get_pace_prediction(dist, venue, nige_count=0, senkou_count=0):
    """
    コース×距離の統計からペースを予測。
    Returns dict: pred_rpci, slow_pct, fast_pct, label, lamp, elem_adv, comment
    """
    key = (float(dist), venue)
    if key not in PACE_TABLE:
        base_avg = 48.0 + (float(dist) - 1200) / 400
        return dict(pred_rpci=round(base_avg,1), base_rpci=round(base_avg,1),
                    std=3.5, slow_pct=30, fast_pct=30,
                    label='データ不足', lamp='⚪', elem_adv=[], comment='コースデータ不足')
    base_avg, std, slow_pct, fast_pct = PACE_TABLE[key]
    pred_rpci = round(base_avg - (nige_count * 0.8 + senkou_count * 0.3), 1)
    if   slow_pct >= 70: label, lamp = '★スロー確定', '🟠'
    elif fast_pct >= 60: label, lamp = '★ハイ確定',   '🔵'
    elif slow_pct >= 50: label, lamp = 'スロー傾向',   '🟠'
    elif fast_pct >= 40: label, lamp = 'ハイ傾向',     '🔵'
    else:                label, lamp = 'どちらも',     '⚪'
    if slow_pct >= 50:
        elem_adv = ['ギアチェンジ', 'ロンスパ・ギアチェンジ']
        comment = f'スロー率{slow_pct}% — GC型有利、先行馬の前残りも警戒'
    elif fast_pct >= 40:
        elem_adv = ['基礎スピード・パワー', 'パワー・ロンスパ']
        comment = f'ハイ率{fast_pct}% — 基礎スピード型有利、差し馬が届きやすい'
    else:
        elem_adv = []
        comment = 'どちらも起こりうる — 逃げ・先行馬の顔触れに注意'
    if nige_count >= 2:
        comment += f'（逃げ{nige_count}頭→ハイ寄り）'
    elif nige_count == 0 and slow_pct >= 50:
        comment += '（逃げ不在→更にスロー化の可能性）'
    return dict(pred_rpci=pred_rpci, base_rpci=base_avg, std=std,
                slow_pct=slow_pct, fast_pct=fast_pct,
                label=label, lamp=lamp, elem_adv=elem_adv, comment=comment)

# ============================================================
# 展開ボーナス機能（2024-2025年バックテストで検証済み）
# ------------------------------------------------------------
# 検証結果(平場全芝・厳選3レース/日・軸+相手3頭):
#   展開ボーナスなし: 馬単回収率 70.8% / 馬連回収率 70.4%
#   展開ボーナスあり: 馬単回収率 82.3% / 馬連回収率 74.0%
# ============================================================

def classify_running_style(gap_ests):
    """過去走のgap_estリストから脚質ゾーン(1逃げ/2先行/3中団/4後方)を仮判定。
    直近を重めに、最大3走の加重平均で判定する。データが無ければNoneを返す。
    """
    valid = [g for g in gap_ests if g is not None and not (isinstance(g, float) and math.isnan(g))]
    if not valid:
        return None
    weights = [0.5, 0.3, 0.2]
    vs = valid[:3]
    w = weights[:len(vs)]
    avg = sum(g * wi for g, wi in zip(vs, w)) / sum(w)
    return gap_to_zone(avg)


def precompute_running_styles(entry_bytes):
    """出走表CSVを軽く読み、各馬の脚質ゾーンを仮判定して集計する。
    calc_lpi本番の前に呼び、get_pace_predictionのnige_count/senkou_countを
    自動算出するために使う(2パス構成の1パス目)。
    Returns: dict {'nige': n, 'senkou': n, 'chudan': n, 'oikomi': n, 'fumei': n}
    """
    for enc in ['cp932', 'shift_jis', 'utf-8-sig', 'utf-8']:
        try:
            df = pd.read_csv(io.BytesIO(entry_bytes), encoding=enc)
            break
        except (UnicodeDecodeError, Exception):
            continue
    else:
        raise ValueError('出走表CSVの文字コードを判定できませんでした')

    counts = {'nige': 0, 'senkou': 0, 'chudan': 0, 'oikomi': 0, 'fumei': 0}
    for _, row in df.iterrows():
        gap_ests = []
        for wd in WALK_DEFS[:3]:
            gap_raw = row.get(wd['gap'], None)
            try:
                g = float(gap_raw)
                if not math.isnan(g):
                    gap_ests.append(g)
            except (TypeError, ValueError):
                continue
        zone = classify_running_style(gap_ests)
        if zone == 1: counts['nige'] += 1
        elif zone == 2: counts['senkou'] += 1
        elif zone == 3: counts['chudan'] += 1
        elif zone == 4: counts['oikomi'] += 1
        else: counts['fumei'] += 1
    return counts


def apply_pace_bonus(dom_elem, elem_adv, lpi, strength=3.0):
    """展開適合ボーナス。dom_elemがelem_adv(有利な要素型リスト)に含まれれば+strength、
    elem_advが空(どちらも起こりうる想定)なら補正なし、含まれなければ補正なし
    (ペナルティは課さない、検証済みの設計)。
    """
    if not elem_adv:
        return lpi
    if dom_elem in elem_adv:
        return min(100.0, round(lpi + strength, 1))
    return lpi

# ============================================================
# 上がり予測スコア
# 過去走のZスコア（コース補正済み上がり）から予測
# 検証結果: 相関r=0.27, A評価のZ平均=+0.38（C評価=-0.24）
# 競馬場ごとの基準上がりは base_dict / 稍重_dict を使用
# ============================================================

# ============================================================
# クッション値補正
# 基準クッション値=9.0。これより低い（軟）ほど上がりが遅くなる
# 補正式: (9.0 - cushion) × 0.15秒
# ============================================================
def calc_cushion_adj(cushion_val, base_cushion=9.0, coef=0.15):
    """クッション値から基準上がりへの補正値を計算"""
    if cushion_val is None:
        return 0.0
    return round((base_cushion - float(cushion_val)) * coef, 3)


# ============================================================
# ペース調整済みZスコア用 回帰係数テーブル
# 基準上がり = intercept + slope × 前半1F平均
# データ: 2020〜2026年全重賞 良馬場 (scipy.stats.linregress)
# 使用条件: PCI追走スコア機能がONかつtarget_front_1fが設定されている場合
# ============================================================
PACE_REGRESSION = {
    # (距離, 競馬場): (slope, intercept, r)
    (1000,'新潟'): (-0.9014, 43.1328, -0.245),
    (1200,'中京'): (-0.8246, 43.5797, -0.235),
    (1200,'中山'): (-2.3241, 60.6847, -0.384),
    (1200,'京都'): (-1.0894, 46.6041, -0.288),
    (1200,'函館'): (-0.6987, 42.7827, -0.185),
    (1200,'阪神'): (-2.5037, 62.9901, -0.522),
    (1400,'中京'): (-2.4420, 63.1369, -0.408),
    (1400,'京都'): (-0.7545, 43.1096, -0.216),
    (1400,'新潟'): (-2.1218, 59.7741, -0.524),
    (1400,'阪神'): (-1.8358, 56.0789, -0.371),
    (1600,'中京'): (-2.0492, 59.2988, -0.424),
    (1600,'京都'): (-1.3379, 50.5699, -0.228),
    (1600,'新潟'): (-1.2365, 48.9491, -0.281),
    (1600,'東京'): (-1.8148, 55.7700, -0.393),
    (1600,'阪神'): (-2.7806, 67.3822, -0.536),
    (1800,'中山'): (-1.3497, 51.7929, -0.264),
    (1800,'京都'): (-1.7638, 55.8201, -0.487),
    (1800,'函館'): (-2.0303, 60.1176, -0.299),
    (1800,'小倉'): (-3.8866, 81.7728, -0.463),
    (1800,'新潟'): (-2.4851, 63.9123, -0.560),
    (1800,'東京'): (-1.5440, 52.8859, -0.399),
    (1800,'阪神'): (-1.7165, 55.0031, -0.426),
    (2000,'中京'): (-1.9410, 58.6839, -0.400),
    (2000,'函館'): (-4.1172, 85.4790, -0.384),
    (2000,'小倉'): (-2.7953, 69.3979, -0.352),
    (2000,'新潟'): (-5.4022,100.2316, -0.746),
    (2000,'東京'): (-2.5017, 64.7120, -0.430),
    (2000,'阪神'): (-1.6045, 54.8995, -0.253),
    (2200,'中京'): (-3.2844, 75.9519, -0.330),
    (2200,'京都'): (-3.5713, 78.6825, -0.585),
    (2200,'阪神'): (-3.4860, 77.8011, -0.535),
    (2400,'京都'): (-3.3681, 76.8036, -0.554),
    (2400,'東京'): (-3.2628, 74.8698, -0.337),
    (2500,'中山'): (-1.7467, 57.7793, -0.188),
    (2500,'東京'): (-4.0804, 85.1470, -0.632),
}

def get_pace_adjusted_base(dist, venue, target_front_1f):
    """
    今回のペース（target_front_1f）に対応したコース基準上がりを返す。
    回帰式: 基準 = intercept + slope × target_front_1f
    登録なし → None（固定基準を使う）
    """
    key = (float(dist), str(venue))
    if key not in PACE_REGRESSION:
        return None
    slope, intercept, r = PACE_REGRESSION[key]
    return round(intercept + slope * target_front_1f, 3)

# 競馬場×距離の基準上がり（上がり予測の補正に使用）
# 対象コースで上がりが速くなる/遅くなる傾向を補正
COURSE_AGARI_BASE = {
    # (距離, 競馬場): 良馬場の基準上がり秒数
    # 数値が小さいコース = 上がりが速いコース = 高い数値のZが出やすい
    (1200,'東京'): 33.98, (1200,'中山'): 34.29, (1200,'阪神'): 34.06,
    (1200,'中京'): 34.11, (1200,'京都'): 33.93, (1200,'新潟'): 33.85,
    (1400,'東京'): 34.10, (1400,'阪神'): 34.27, (1400,'中京'): 34.22,
    (1600,'東京'): 34.38, (1600,'中山'): 34.51, (1600,'阪神'): 34.46,
    (1600,'京都'): 34.28, (1600,'中京'): 34.27, (1600,'新潟'): 34.41,
    (1800,'東京'): 34.63, (1800,'中山'): 35.08, (1800,'阪神'): 34.85,
    (2000,'東京'): 34.35, (2000,'中山'): 35.36, (2000,'阪神'): 35.55,
    (2000,'京都'): 34.86, (2000,'中京'): 34.97,
    (2200,'京都'): 34.83, (2200,'阪神'): 35.25,
    (2400,'東京'): 35.22, (2400,'京都'): 34.91,
    (2500,'中山'): 35.74,
}

def predict_agari(past_runs, target_dist, target_venue, target_baba='良',
                  predicted_pace_cat=None, pred_gap=None, pci_cs_score=None,
                  cushion_correction=0.0, base_dict_for_z=None,
                  pace_target_front_1f=None, course_base_dict=None):
    """
    上がり予測 v3：全走平均Z + 先行×H消耗補正

    course_base_dict: {(距離,競馬場): 平均上がり} の上書き用テーブル。
        None(既定)ならターフ用 COURSE_AGARI_BASE を使う。

    設計方針:
    1. ペース帯別分類をやめて全走の加重平均Zを使う
       （H/M/S分類の精度向上効果がr=0.284と同じことが判明）
    2. 先行（地点差≤0.4）×H走（RPCI≤47）は消耗走として重み×0.3
       （能力ではなく展開の犠牲なので過小評価を防ぐ）
    3. 予測ポジション（pred_gap）で位置取り補正（既存維持）
    """
    GRADE_THRESH = {'A': 0.458, 'C': -0.341}

    # 全有効走を収集
    all_runs_data = []
    for r in past_runs:
        if r.get('excluded_baba') or r.get('excluded_track'):
            continue
        z = r.get('z')
        if z is None or math.isnan(float(z)):
            continue
        z = float(z)
        rpci    = r.get('rpci', 50)
        gap_est = r.get('gap_est', 0.7)
        grade   = r.get('grade', '')
        agari   = r.get('agari')
        dist    = r.get('dist')
        venue   = r.get('venue')
        fp_z    = r.get('front_pace_z')
        pci     = r.get('pci')

        z_use = z
        if (target_dist is not None and target_venue is not None and
                pci is not None and agari is not None):
            try:
                pci_f   = float(pci); agari_f = float(agari)
                run_front_1f = (pci_f + 50) * agari_f / 100 / 3
                pace_base = get_pace_adjusted_base(
                    float(str(dist).replace('m','')),
                    str(venue),
                    run_front_1f
                )
                if pace_base is not None:
                    _base_dict = base_dict_for_z or {}
                    _, std_val = _base_dict.get(
                        (float(str(dist).replace('m','')), str(venue)), (pace_base, 1.0))
                    std_val = std_val if std_val and std_val > 0 else 1.0
                    z_use = (pace_base - agari_f) / std_val
            except Exception:
                z_use = z

        chase_bonus = 0.0
        if fp_z is not None and not math.isnan(float(fp_z)):
            fp_z_f = float(fp_z)
            if fp_z_f < -0.3 and z_use > 0.3:
                chase_bonus = round(min(abs(fp_z_f) * z_use * 0.15, 0.4), 3)
        course_w = calc_course_weight(
            dist, venue, target_dist, target_venue
        ) if (dist is not None and venue is not None) else 1.0

        all_runs_data.append({
            'z': z_use + chase_bonus, 'rpci': float(rpci),
            'gap_est': float(gap_est), 'grade': grade,
            'course_w': course_w,
        })

    if not all_runs_data:
        return None

    high_grade_runs = [r for r in all_runs_data if r['grade'] in ('G1','G2')]
    lower_runs      = [r for r in all_runs_data if r['grade'] not in ('G1','G2','G3','L')]

    use_high_grade_only = False
    if len(high_grade_runs) >= 1 and len(lower_runs) >= 1:
        z_high  = np.mean([r['z'] for r in high_grade_runs])
        z_lower = np.mean([r['z'] for r in lower_runs])
        if z_lower - z_high > 0.5:
            use_high_grade_only = True
    if (pci_cs_score is not None and
            pci_cs_score < 0.5 and
            len(high_grade_runs) >= 1):
        use_high_grade_only = True

    source_runs = high_grade_runs if use_high_grade_only else all_runs_data

    weighted_zs = []
    for r in source_runs:
        is_senkou_H = (r['gap_est'] <= 0.4 and r['rpci'] <= 47)
        pace_w   = 0.3 if is_senkou_H else 1.0
        course_w = r.get('course_w', 1.0)
        weight   = round(pace_w * course_w, 3)
        weighted_zs.append((r['z'], weight))

    if not weighted_zs:
        weighted_zs = [
            (r['z'],
             round((0.3 if (r['gap_est']<=0.4 and r['rpci']<=47) else 1.0)
                   * r.get('course_w', 1.0), 3))
            for r in all_runs_data
        ]

    if not weighted_zs:
        return None

    total_w = sum(w for _, w in weighted_zs)
    pred_z  = sum(z * w for z, w in weighted_zs) / total_w if total_w > 0 else 0.0
    pred_z  = round(pred_z, 3)
    all_z_list = [z for z, _ in weighted_zs]

    pci_cs_coef = 1.0
    if pci_cs_score is not None:
        if pci_cs_score >= 2.0:    pci_cs_coef = 1.00
        elif pci_cs_score >= 0.5:  pci_cs_coef = 0.85
        elif pci_cs_score >= -0.5: pci_cs_coef = 0.70
        else:                      pci_cs_coef = 0.55
    pred_z = round(pred_z * pci_cs_coef, 3)

    z_std = float(np.std(all_z_list, ddof=1)) if len(all_z_list) >= 2 else 0.8
    if z_std <= 0.4:   confidence = '◎安定'
    elif z_std <= 0.7: confidence = '○やや安定'
    else:              confidence = '△不安定'

    if pred_z >= GRADE_THRESH['A']:
        grade, grade_label = 'A', '🔴 切れ味A'
    elif pred_z < GRADE_THRESH['C']:
        grade, grade_label = 'C', '⚪ 切れ味C'
    else:
        grade, grade_label = 'B', '🟡 切れ味B'

    n_discounted = sum(1 for _, w in weighted_zs if w < 0.9)
    grade_filter_note = ''
    if use_high_grade_only:
        grade_filter_note = f'（PCI-CS△以下のためG1/G2走{len(high_grade_runs)}件のみ使用）'
    comment = (f'全走Z平均={pred_z:+.2f}（PCI-CS係数×{pci_cs_coef:.2f}適用後）{grade_filter_note}。'
               f'{"上位33%の切れ味" if grade=="A" else "下位33%の末脚" if grade=="C" else "標準的な末脚"}。')
    if n_discounted > 0:
        comment += f' 先行×H消耗走{n_discounted}件は重み0.3で補正済み。'
    if pci_cs_coef < 1.0:
        comment += f' PCI追走スコア{pci_cs_score:.2f}→Z×{pci_cs_coef:.2f}で割引。'

    if pace_target_front_1f is not None:
        _front_for_base = pace_target_front_1f
    elif predicted_pace_cat is not None:
        _dist_key = min(FRONT_PACE_BASE.keys(),
                        key=lambda d: abs(d - float(target_dist)))
        _base_front, _std_front = FRONT_PACE_BASE[_dist_key]
        if predicted_pace_cat == 'H':
            _front_for_base = _base_front - 0.24
        elif predicted_pace_cat == 'S':
            _front_for_base = _base_front + 0.21
        else:
            _front_for_base = _base_front
    else:
        _front_for_base = None

    pace_adj_base = None
    if _front_for_base is not None:
        pace_adj_base = get_pace_adjusted_base(
            float(target_dist), str(target_venue), _front_for_base)

    if pace_adj_base is not None:
        course_base = pace_adj_base
    else:
        course_base = (course_base_dict or COURSE_AGARI_BASE).get((float(target_dist), target_venue), 34.5)
    if str(target_baba).strip() == '稍':
        course_base += 0.4
    if cushion_correction != 0.0:
        course_base = round(course_base + cushion_correction, 3)
    GAP_CORRECTION = 0.383
    BASE_GAP       = 0.7
    if pred_gap is not None:
        gap_adj    = GAP_CORRECTION * (pred_gap - BASE_GAP)
        pred_agari = round(course_base - pred_z + gap_adj, 1)
        gap_note   = f'（位置取り補正: 地点差{pred_gap:.1f}秒 {gap_adj:+.2f}秒）'
    else:
        pred_agari = round(course_base - pred_z, 1)
        gap_note   = '（位置取り不明: 地点差0.7秒想定）'

    return {
        'pace_cat':     None,
        'pred_z':       pred_z,
        'z_by_pace':    {'H': None, 'M': None, 'S': None},
        'n_by_pace':    {'H': 0, 'M': 0, 'S': 0},
        'grade':        grade,
        'grade_label':  grade_label,
        'pred_agari':   pred_agari,
        'gap_note':     gap_note,
        'course_base':  course_base,
        'confidence':   confidence,
        'z_std':        round(z_std, 3),
        'comment':      comment,
        'n_valid':      len(weighted_zs),
        'n_discounted': n_discounted,
        'past_zs':      [round(z, 2) for z, _ in weighted_zs],
    }


POS_ZONE_LABELS = {
    1: ('逃げ',  '平均0.1以下',  '#1A237E', '🟦'),
    2: ('先行',  '0.2〜0.4秒',   '#1B5E20', '🟩'),
    3: ('中団',  '0.5〜1.0秒',   '#E65100', '🟨'),
    4: ('後方',  '1.1秒〜',      '#B71C1C', '🟥'),
}

def gap_to_zone(gap):
    """地点差(秒) → ポジション帯番号"""
    if gap is None or (isinstance(gap, float) and math.isnan(gap)):
        return None
    if gap <= 0.1: return 1
    if gap <= 0.4: return 2
    if gap <= 1.0: return 3
    return 4

def predict_position(past_gaps, rpci_pred=None):
    """
    過去走の地点差リスト → ポジション予測。
    """
    valid_gaps = [g for g in past_gaps if g is not None and not math.isnan(float(g))]
    if not valid_gaps:
        return None

    weights = [0.50, 0.30, 0.15, 0.05]
    total_w, weighted_sum = 0.0, 0.0
    for i, g in enumerate(valid_gaps[:4]):
        w = weights[i] if i < len(weights) else 0.05
        weighted_sum += g * w
        total_w += w
    pred_gap = round(weighted_sum / total_w, 2) if total_w > 0 else valid_gaps[0]

    if rpci_pred is not None:
        if rpci_pred >= 54:    pred_gap -= 0.05
        elif rpci_pred <= 46:  pred_gap += 0.05

    pred_gap = max(0.0, pred_gap)
    pred_zone = gap_to_zone(pred_gap)

    if len(valid_gaps) >= 2:
        gap_std = float(np.std(valid_gaps, ddof=1))
    else:
        gap_std = 0.5

    if gap_std <= 0.25:   confidence = '◎安定'
    elif gap_std <= 0.45: confidence = '○やや安定'
    else:                 confidence = '△不安定'

    zone_info = POS_ZONE_LABELS.get(pred_zone, (str(pred_zone),'','#888','⚪'))

    return {
        'pred_gap':   pred_gap,
        'pred_zone':  pred_zone,
        'zone_name':  zone_info[0],
        'zone_range': zone_info[1],
        'label':      f"{zone_info[0]}（{zone_info[1]}）",
        'confidence': confidence,
        'icon':       zone_info[3],
        'color':      zone_info[2],
        'gap_std':    round(gap_std, 3),
        'n_valid':    len(valid_gaps),
        'past_gaps':  valid_gaps,
    }


# ============================================================
# 前半ペース速度（追走能力）の評価
# ============================================================
FRONT_PACE_BASE = {
    1000: (11.244, 0.219),
    1200: (11.462, 0.253),
    1400: (11.674, 0.269),
    1500: (12.123, 0.160),
    1600: (11.840, 0.220),
    1800: (12.045, 0.247),
    2000: (12.117, 0.214),
    2200: (12.131, 0.177),
    2400: (12.248, 0.169),
    2500: (12.323, 0.178),
    3000: (12.400, 0.200),
    3200: (12.420, 0.200),
}

def calc_front_pace_z(pci, agari, dist):
    """
    PCIと上がりから「前半ペースZスコア」を計算。
    """
    try:
        pci = float(pci); agari = float(agari); dist = int(dist)
    except: return None
    if math.isnan(pci) or math.isnan(agari): return None

    ave3f = (pci + 50) * agari / 100
    front_1f = ave3f / 3

    dists = sorted(FRONT_PACE_BASE.keys())
    nearest = min(dists, key=lambda d: abs(d - dist))
    base_1f, std_1f = FRONT_PACE_BASE[nearest]
    std_1f = std_1f if std_1f > 0 else 0.2

    return round((base_1f - front_1f) / std_1f, 3)


def calc_pci_cs(past_runs, target_front_1f, tolerance_good=0.15, tolerance_near=0.30):
    """
    PCI追走スコア（PCI Chasing Score）
    """
    valid_runs = []
    for r in past_runs:
        if r.get('excluded_baba') or r.get('excluded_track'):
            continue
        pci = r.get('pci')
        agari = r.get('agari')
        if pci is None or agari is None:
            continue
        try:
            pci_f = float(pci); agari_f = float(agari)
            if math.isnan(pci_f) or math.isnan(agari_f):
                continue
        except:
            continue
        ave3f = (pci_f + 50) * agari_f / 100
        front_1f = ave3f / 3
        valid_runs.append({
            'race':     r.get('race', ''),
            'front_1f': front_1f,
            'rank':     r.get('rank_int'),
            'agari':    agari_f,
            'pci':      pci_f,
        })

    if not valid_runs:
        return {'score': 0.0, 'judge': '△', 'fastest_1f': None,
                'best_run': None, 'n_fast': 0, 'detail': 'PCIデータなし'}

    score = 0.0
    best_run = None
    detail_parts = []

    for r in valid_runs:
        diff = target_front_1f - r['front_1f']
        rank = r['rank']

        if diff >= 0:
            if rank and rank <= 3:
                s = min(3.0, diff * 10 + 2.0)
                score += s
                if best_run is None or r['front_1f'] < best_run['front_1f']:
                    best_run = r
                mark = '✅'
            elif rank and rank <= 6:
                score += min(1.0, diff * 5) * 0.3
                mark = '△'
            else:
                score -= 0.5
                mark = '❌'
            detail_parts.append(f'{r["race"][:8]}(1F={r["front_1f"]:.3f},{rank}着{mark})')

        elif diff >= -tolerance_good:
            if rank and rank <= 3:
                score += 1.0
                mark = '✅近'
            elif rank and rank <= 6:
                score += 0.2
                mark = '△近'
            else:
                mark = ''
            if mark:
                detail_parts.append(f'{r["race"][:8]}(1F={r["front_1f"]:.3f},{rank}着{mark})')

        elif diff >= -tolerance_near:
            if rank and rank <= 3:
                score += 0.3

    fastest = min(r['front_1f'] for r in valid_runs)
    n_fast  = sum(1 for r in valid_runs if r['front_1f'] <= target_front_1f + tolerance_good)
    if fastest > target_front_1f + tolerance_near:
        score -= 1.5
        detail_parts.append(f'最速前半={fastest:.3f}秒/F（経験不足）')

    score = round(score, 2)
    if score >= 2.0:   judge = '◎'
    elif score >= 0.5: judge = '○'
    elif score >= -0.5: judge = '△'
    else:              judge = '×'

    detail = ' / '.join(detail_parts[:3]) if detail_parts else f'最速={fastest:.3f}秒/F'
    best_name = best_run['race'][:10] if best_run else None

    return {
        'score':      score,
        'judge':      judge,
        'fastest_1f': round(fastest, 3),
        'best_run':   best_name,
        'n_fast':     n_fast,
        'detail':     detail,
    }


def calc_race_env_score(rpci, dist, grade, venue):
    """
    「今回出走するレース」自体の紛れやすさを判定する。
    """
    score = 0
    try:
        if rpci is not None and float(rpci) <= 47: score += 1
    except Exception:
        pass
    try:
        if dist is not None and float(dist) <= 1400: score += 1
    except Exception:
        pass
    if grade == 'G3': score += 1
    if venue in ('函館', '新潟', '小倉'): score += 1
    return score


def calc_chase_env_discount(rpci, dist, grade, venue):
    """
    レース環境スコア（紛れの起きやすさ）からZへの割引係数を返す。
    """
    score = 0
    try:
        if rpci is not None and float(rpci) <= 47: score += 1
    except Exception:
        pass
    try:
        if dist is not None and float(dist) <= 1400: score += 1
    except Exception:
        pass
    if grade == 'G3': score += 1
    if venue in ('函館', '新潟', '小倉'): score += 1

    if score <= 0:   return 1.00
    elif score == 1: return 0.90
    elif score == 2: return 0.80
    else:            return 0.70


def calc_course_weight(run_dist, run_venue, target_dist, target_venue):
    """
    過去走のコース・距離と今回の一致度から重みを計算。
    """
    try:
        run_dist    = float(run_dist)
        target_dist = float(target_dist)
        same_venue  = str(run_venue) == str(target_venue)
        dist_diff   = abs(run_dist - target_dist)
    except Exception:
        return 1.0

    if same_venue and dist_diff == 0:     return 3.0
    elif same_venue and dist_diff <= 200: return 2.0
    elif same_venue and dist_diff <= 600: return 1.5
    elif dist_diff == 0:                  return 1.2
    elif dist_diff <= 200:                return 0.8
    elif dist_diff <= 400:                return 0.5
    else:                                 return 0.3

WALK_DEFS = [
    {'n':1,'agari':'上り3F',  'rpci':'RPCI',  'venue':'場所',  'dist':'距離',
     'baba':'馬場状態',  'rank':'着順',  'race':'ﾚｰｽ名･1走前','td':'TD','gap':'-3F差'},
    {'n':2,'agari':'上り3F.1','rpci':'RPCI.1','pci':'PCI.1','venue':'場所.1','dist':'距離.1',
     'baba':'馬場状態.1','rank':'着順.1','race':'ﾚｰｽ名･2走前','td':'TD.1','gap':'-3F差.1'},
    {'n':3,'agari':'上り3F.2','rpci':'RPCI.2','pci':'PCI.2','venue':'場所.2','dist':'距離.2',
     'baba':'馬場状態.2','rank':'着順.2','race':'ﾚｰｽ名･3走前','td':'TD.2','gap':'-3F差.2'},
    {'n':4,'agari':'上り3F.3','rpci':'RPCI.3','pci':'PCI.3','venue':'場所.3','dist':'距離.3',
     'baba':'馬場状態.3','rank':'着順.3','race':'ﾚｰｽ名･4走前','td':'TD.3','gap':'-3F差.3'},
    {'n':5,'agari':'上り3F.4','rpci':'RPCI.4','pci':'PCI.4','venue':'場所.4','dist':'距離.4',
     'baba':'馬場状態.4','rank':'着順.4','race':'ﾚｰｽ名･5走前','td':'TD.4','gap':'-3F差.4'},
]

ELEM_COLOR = {
    '基礎スピード・パワー': '#e8a030',
    'パワー・ロンスパ':     '#E24B4A',
    'ロンスパ・ギアチェンジ':'#378ADD',
    'ギアチェンジ':         '#639922',
    'ロンスパ':             '#888780',
    '不明':                 '#aaa',
}

# ============================================================
# ユーティリティ関数（変更不要）
# ============================================================
def sigmoid_score(r):
    z = (r - SIGMOID_CENTER) / SIGMOID_SCALE
    s = 1 / (1 + math.exp(-max(-10, min(10, z))))
    return round(40 + s * 60, 1)

def extract_grade(n):
    m = re.search(r'G[1-3]', str(n))
    return m.group() if m else ('L' if 'L' in str(n) else '')

def to_int_rank(s):
    try:
        return int(str(s).translate(str.maketrans('１２３４５６７８９０','1234567890')))
    except:
        return None

def get_venue_from_kaisan(s):
    kanji = re.sub(r'[0-9０-９A-Za-zａ-ｚＡ-Ｚ]', '', str(s).strip())
    VMAP = {
        '東京':'東京','中山':'中山','京都':'京都','阪神':'阪神','新潟':'新潟',
        '中京':'中京','福島':'福島','小倉':'小倉','札幌':'札幌','函館':'函館',
        '東':'東京','京':'京都','阪':'阪神','新':'新潟','札':'札幌',
        '函':'函館','小':'小倉','福':'福島','中':'中山','名':'中京',
    }
    return VMAP.get(kanji.strip(), '東京')

def get_std_weight(grade, age, sex):
    key = (grade, '3yo' if age == 3 else 'senior')
    if key not in WEIGHT_TABLE:
        key = ('', key[1])
    p = WEIGHT_TABLE[key]
    return p[1] if sex == '牝' else p[0]

def weight_correction_sec(grade, age, sex):
    return (WEIGHT_BASE - get_std_weight(grade, age, sex)) * WEIGHT_PER_KG

def classify_element(rpci, gap_est, z=0.0):
    if rpci is None or gap_est is None:
        return '不明'
    try:
        if math.isnan(float(rpci)) or math.isnan(float(gap_est)):
            return '不明'
    except:
        return '不明'
    rpci, gap_est = float(rpci), float(gap_est)
    is_後傾 = rpci >= 50
    is_先行 = gap_est < 0.6
    is_速   = float(z) > 0.3
    if is_後傾:
        if not is_先行: return 'ギアチェンジ' if is_速 else 'ロンスパ'
        else:           return 'ロンスパ・ギアチェンジ' if is_速 else '基礎スピード・パワー'
    else:
        if is_先行: return '基礎スピード・パワー'
        else:       return 'パワー・ロンスパ'

def calc_pb_v11(rpci, gap_est, z):
    rpci_dev = abs(rpci - 50) / 10
    is_後傾  = rpci >= 50
    is_先行  = gap_est < 0.6
    is_差し  = gap_est >= 0.8
    adj = 0.0
    if   is_後傾 and is_先行: adj = -rpci_dev * (0.6 - gap_est) / 0.6 * 0.4
    elif is_後傾 and is_差し: adj =  rpci_dev * (gap_est - 0.8) / 0.4 * 0.3
    elif not is_後傾 and is_先行: adj = rpci_dev * (0.6 - gap_est) / 0.6 * 0.3
    elif not is_後傾 and is_差し: adj = -rpci_dev * (gap_est - 0.8) / 0.4 * 0.4
    if adj > 0:   adj *= max(0.3, min(2.0, 1.0 + z * 0.3))
    elif adj < 0 and z > 0.5: adj += min(abs(adj) * 0.5, z * 0.1)
    return round(adj, 3)

def get_venue_bonus(venue, elem):
    return VENUE_ELEMENT_COEF.get(venue, {}).get(elem, 1.0)

def apply_venue_bonus(venue, elem, lpi, strength=0.15):
    coef  = get_venue_bonus(venue, elem)
    delta = (coef - 1.0) * strength * 100
    return max(40.0, min(100.0, round(lpi + delta, 1)))

# ============================================================
# 基準テーブル構築（キャッシュ）
# ------------------------------------------------------------
# 【重要】ここに読み込ませるCSVは「重賞のみ」ではなく、
# 平場（未勝利〜3勝クラス）を含む水準のデータを推奨。
# 検証済み: 重賞級のみで基準を作ると、平場の馬を過小評価する
# 系統的なズレが生じる(2024-2025年バックテストで確認、
# 較正修正後に馬単回収率69.4%→82.3%まで改善)。
#
# 京都競馬場をアップする場合、2020年-2023年4月のデータは
# 改修前の旧コースなので、新コース(2023年4月以降)のみに
# 絞ったCSVにすることを推奨(旧コース混在で回収率53%→82%に改善した実績あり)。
# ============================================================
@st.cache_data
def build_base_table(file_bytes):
    # エンコードを自動判定
    for enc in ['cp932', 'shift_jis', 'utf-8-sig', 'utf-8']:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            break
        except (UnicodeDecodeError, Exception):
            continue
    else:
        raise ValueError('CSVの文字コードを判定できませんでした')

    # ===== 新形式(年/月/日が別列、場所、上がり3Fタイム等)を検出して旧形式に変換 =====
    # 例: 2020-2025平場結果.csv のようなJRA-VAN生データ形式
    is_new_schema = '上り3F' not in df.columns and '上がり3Fタイム' in df.columns
    if is_new_schema:
        if {'年', '月', '日'}.issubset(df.columns):
            df['日付'] = (df['年'].astype(int) * 10000 +
                          df['月'].astype(int) * 100 +
                          df['日'].astype(int))
        if '開催' not in df.columns and '場所' in df.columns:
            df['開催'] = df['場所']
        if '上り3F' not in df.columns and '上がり3Fタイム' in df.columns:
            df['上り3F'] = df['上がり3Fタイム']
        if '距離' in df.columns:
            # 数値のみの距離を文字列化(str.extractは既に astype(str) 済みなのでこのままでOK)
            pass

    df['距離_num'] = df['距離'].astype(str).str.extract(r'(\d+)').astype(float)
    df['上がり']   = pd.to_numeric(df['上り3F'], errors='coerce')
    df['競馬場']   = df['開催'].apply(get_venue_from_kaisan)
    df['馬場']     = df['馬場状態'].astype(str).str.strip()
    df['日付_num'] = pd.to_numeric(df['日付'], errors='coerce')
    df['年']       = (df['日付_num'] // 10000).fillna(0).astype(int)
    df['レース名_s'] = df['レース名'].astype(str).str.strip() if 'レース名' in df.columns else ''

    # ===== 年度重み（直近ほど重い）=====
    def yr_weight(y):
        try: y = int(y)
        except: return 1.0
        return 2.0 if y >= 25 else (1.5 if y >= 23 else 1.0)
    df['yr_w'] = df['年'].apply(yr_weight)

    valid = df[df['馬場'].isin(['良', '稍'])].copy()

    # 年度重み付き平均・std
    def weighted_stats(g):
        v = g.dropna(subset=['上がり'])
        if len(v) == 0:
            return pd.Series({'avg': np.nan, 'std': np.nan, 'n': 0})
        w = v['yr_w']
        wsum = w.sum()
        wavg = (v['上がり'] * w).sum() / wsum
        wvar = (w * (v['上がり'] - wavg) ** 2).sum() / wsum
        return pd.Series({'avg': wavg, 'std': max(np.sqrt(wvar), 0.3), 'n': len(v)})

    stats = valid.groupby(['距離_num', '競馬場', '馬場']).apply(weighted_stats).reset_index()
    stats = stats[stats['n'] >= 5]
    良_s  = stats[stats['馬場'] == '良']
    稍_s  = stats[stats['馬場'] == '稍']
    base_dict = {(r['距離_num'], r['競馬場']): (r['avg'], r['std']) for _, r in 良_s.iterrows()}
    稍重_dict = {(r['距離_num'], r['競馬場']): (r['avg'], r['std']) for _, r in 稍_s.iterrows()}

    # ===== 同名レース＋直近重み付き基準辞書 =====
    def normalize_name(n):
        return re.sub(r'[ＨＧＳＬ０-９G0-9HLS\s\u3000・Ｐ]', '', str(n))

    race_base_dict = {}
    vg = valid[valid['馬場'] == '良'].copy()
    if len(vg) > 0 and '日付_num' in vg.columns:
        race_avgs = vg.groupby(
            ['距離_num', '競馬場', '日付_num', 'レース名_s', '年']
        )['上がり'].agg(avg='mean', n='count').reset_index()
        race_avgs = race_avgs[race_avgs['n'] >= 5]

        for (dist, venue), grp in race_avgs.groupby(['距離_num', '競馬場']):
            grp = grp.sort_values('日付_num', ascending=False).reset_index(drop=True)
            seen = set()
            for _, row in grp.iterrows():
                tname = normalize_name(row['レース名_s'])
                if not tname or len(tname) < 2 or tname in seen:
                    continue
                seen.add(tname)

                w_rows = []
                for idx2, row2 in grp.iterrows():
                    rname2 = normalize_name(row2['レース名_s'])
                    yr_w2  = yr_weight(row2['年'])
                    # 同名ボーナス
                    if tname == rname2:
                        name_w = 3.0
                    elif len(tname) >= 3 and (tname[:3] in rname2 or rname2[:3] in tname):
                        name_w = 2.0
                    else:
                        name_w = 1.0
                    recency_w = max(0.3, 1.0 - idx2 * 0.08)
                    w_rows.append((row2['avg'], yr_w2 * name_w * recency_w))

                # 同名全件＋直近5件（他）
                same  = [(a, w) for i, (a, w) in enumerate(w_rows)
                         if normalize_name(grp.iloc[i]['レース名_s'])[:3] == tname[:3]]
                other = [(a, w) for i, (a, w) in enumerate(w_rows)
                         if normalize_name(grp.iloc[i]['レース名_s'])[:3] != tname[:3]][:5]
                use = same + other
                total_w = sum(w for _, w in use)
                if total_w == 0:
                    continue
                wavg = sum(a * w for a, w in use) / total_w
                race_base_dict[(float(dist), str(venue), tname)] = round(wavg, 3)

    return base_dict, 稍重_dict, race_base_dict

def get_z(agari, dist, venue, baba, base_dict, 稍重_dict,
          race_base=None, race_name=None):
    """
    Zスコア計算。race_base（同名レース重み付き基準）があれば優先使用。
    """
    fb  = {1000:32.5,1200:34.0,1400:34.2,1600:34.4,
           1800:34.6,2000:35.0,2200:35.2,2400:35.3,2500:35.5}
    key = (float(dist), venue)
    b   = str(baba).strip()

    # 同名レース重み付き基準（良馬場のみ）
    if race_base and race_name and b == '良':
        tname = re.sub(r'[ＨＧＳＬOP０-９G0-9HLS\s\u3000・]','',str(race_name))
        rkey  = (float(dist), venue, tname)
        if rkey in race_base:
            base = race_base[rkey]
            # stdは通常の基準テーブルから取得
            _, std = base_dict.get(key, (base, 1.0))
            std = std if (std and std > 0) else 1.0
            return (base - agari) / std

    if b == '稍' and key in 稍重_dict:
        base, std = 稍重_dict[key]
    elif key in base_dict:
        base, std = base_dict[key]
        if b == '重': base += 0.8
        elif b == '不': base += 1.5
    else:
        base, std = fb.get(int(dist), 34.4), 1.0
    std = std if (std and std > 0) else 1.0
    return (base - agari) / std

# ============================================================
# LPI計算メイン
# ------------------------------------------------------------
# 【2024-2025年バックテストで検証済みの変更点】
#   ・pace_elem_adv / pace_bonus_strength: 展開ボーナス機能を追加(採用)
#   ・disable_g1_streak_bonus: 既定でTrue(G1好走・連続好走ボーナスを廃止)
#     検証: 廃止した方が馬連回収率が両年で一貫して改善(62.8%→65.6%,
#     94.2%→102.7%)。会場適性ボーナス(bonus_strength)は逆に必須で、
#     外すと馬単回収率が大幅悪化するため既定0.15のまま維持。
#   ・target_dist: 対象レース自身の距離を明示指定できるように変更
#     (旧: 馬の直近過去走の距離で代用していたバグを修正)
# ============================================================
def calc_lpi(entry_bytes, base_dict, 稍重_dict,
             target_track='T', target_venue='東京', bonus_strength=0.15,
             pace_pred_rpci=51.0, race_base_dict=None, target_race_name='',
             target_front_1f_input=None, cushion_correction_input=0.0,
             pace_elem_adv=None, pace_bonus_strength=0.0,
             disable_g1_streak_bonus=True, target_dist=None):
    for enc in ['cp932', 'shift_jis', 'utf-8-sig', 'utf-8']:
        try:
            df = pd.read_csv(io.BytesIO(entry_bytes), encoding=enc)
            break
        except (UnicodeDecodeError, Exception):
            continue
    else:
        raise ValueError('出走表CSVの文字コードを判定できませんでした')

    sex_map = {}
    age_map = {}
    if '性別' in df.columns:
        for _, row in df.iterrows():
            sex_map[str(row['馬名S'])] = str(row['性別']).strip()
    if '年齢' in df.columns:
        for _, row in df.iterrows():
            try:    age_map[str(row['馬名S'])] = int(row['年齢'])
            except: age_map[str(row['馬名S'])] = 4

    results = []
    for _, row in df.iterrows():
        horse = str(row['馬名S'])
        sex   = sex_map.get(horse, '牡')
        age   = age_map.get(horse, 4)
        run_data = []

        for wd in WALK_DEFS:
            try:
                agari_raw = str(row[wd['agari']]).strip()
                if agari_raw in ['----','---','','nan']: continue
                agari = float(agari_raw)
                rpci_raw = str(row[wd['rpci']]).strip()
                if rpci_raw in ['','nan','NaN']: continue
                rpci  = float(rpci_raw)
                venue = FULL_VENUE.get(str(row[wd['venue']]).strip(), '東京')
                dist  = float(str(row[wd['dist']]).strip().replace('m','').replace('芝','').replace('ダ',''))
                baba  = str(row[wd['baba']]).strip()
                baba  = baba if baba not in ['nan','NaN',''] else '良'
                rank  = row[wd['rank']]
                race  = str(row[wd['race']]).strip()
                td    = str(row.get(wd['td'], 'T')).strip().upper()
                track = td if td in ('T','D') else 'T'
                gap_raw = row.get(wd['gap'], None)
                gap   = float(gap_raw) if str(gap_raw).strip() not in ['nan','NaN','','None','----'] else None
                pci_raw = row.get(wd.get('pci',''), None)
                pci_val = float(str(pci_raw).strip()) if str(pci_raw).strip() not in ['nan','NaN','','None','----'] else None
            except:
                continue
            if math.isnan(agari) or math.isnan(rpci): continue

            gap_est  = gap if (gap is not None and not math.isnan(gap)) else 1.5
            grade    = extract_grade(race)
            wt_corr  = weight_correction_sec(grade, age, sex)
            agari_adj = agari + wt_corr
            z        = get_z(agari_adj, dist, venue, baba, base_dict, 稍重_dict, race_base_dict, race)
            pb       = calc_pb_v11(rpci, gap_est, z)
            pm       = 1.0 + abs(rpci - 50) / 25 * 0.4
            rank_int = to_int_rank(rank)

            g1_pen = 1.0
            if grade == 'G1' and rank_int and rank_int > 3 and gap_est > G1_PENALTY_THRESHOLD:
                g1_pen = G1_PENALTY_COEF

            hb, hb_r = 0.0, ''
            if rank_int and rank_int <= 3:
                is_先行_b = gap_est <= 0.4
                is_差し_b = gap_est >= 0.8
                base_b = 0.0
                if rpci <= 50 and is_先行_b:
                    base_b = min((50-rpci)/10*(0.4-gap_est+0.1)*2*0.8, 1.5)
                    hb_r   = f'前傾×先行{rank_int}着[{grade}]'
                elif rpci > 50 and is_差し_b:
                    base_b = min((rpci-50)/10*(gap_est-0.8+0.1)*2*0.8, 1.5)
                    hb_r   = f'後傾×差し{rank_int}着[{grade}]'
                if base_b > 0:
                    if z > 0.5: base_b *= 1.2; hb_r += ' +速上がり'
                    base_b *= GRADE_BONUS.get(grade, 0.7)
                    base_b *= RANK_BONUS_MULT.get(rank_int, 1.0)
                    hb = round(min(base_b, 2.5), 3)

            elem = classify_element(rpci, gap_est, z)

            # 環境スコア割引: 紛れが起きやすいレース環境での好走（高Z）を軽く割り引く
            env_discount = 1.0
            if rank_int and rank_int <= 3 and z > 0:
                env_discount = calc_chase_env_discount(rpci, dist, grade, venue)
                z_for_lpi = z * env_discount
            else:
                z_for_lpi = z

            lpi  = sigmoid_score((z_for_lpi + pb + hb) * pm * g1_pen)
            gw   = GRADE_WEIGHT.get(grade, GRADE_WEIGHT[''])

            # 前半ペースZスコア（PCIから逆算）
            fp_z = calc_front_pace_z(pci_val, agari, dist) if pci_val is not None else None

            run_data.append({
                'n': wd['n'], 'race': race, 'dist': dist, 'venue': venue,
                'rpci': rpci, 'pci': pci_val, 'gap_est': round(gap_est, 2),
                'agari': agari, 'agari_adj': round(agari_adj, 2),
                'wt_corr': round(wt_corr, 2),
                'z': round(z_for_lpi, 3), 'z_raw': round(z, 3),
                'env_discount': env_discount,
                'rank': rank, 'rank_int': rank_int,
                'baba': baba, 'track': track, 'grade': grade,
                'pb': pb, 'pm': round(pm, 3), 'hb': hb, 'hb_r': hb_r,
                'elem': elem, 'lpi': lpi, 'grade_weight': gw,
                'front_pace_z': round(fp_z, 3) if fp_z is not None else None,
                'excluded_baba':  baba not in GOOD_BABA,
                'excluded_track': track != target_track,
            })

        if not run_data: continue
        valid = [r for r in run_data if not r['excluded_baba'] and not r['excluded_track']]
        use   = valid if valid else run_data

        # 改善③: 大敗（壊滅的なZの低下）が平均を過剰に押し下げないようキャップする。
        Z_FLOOR = -1.5
        LPI_FLOOR = sigmoid_score(Z_FLOOR)
        use = [
            {**r, 'lpi': max(r['lpi'], LPI_FLOOR)} if r['z'] < Z_FLOOR else r
            for r in use
        ]

        total_w = sum(r['grade_weight'] for r in use)
        avg_lpi = round(sum(r['lpi']*r['grade_weight'] for r in use)/total_w, 1) \
                  if total_w > 0 else round(np.mean([r['lpi'] for r in use]), 1)

        good     = [r for r in use if r['rank_int'] and r['rank_int'] <= 3]
        elem_src = good if good else use
        dom_elem = Counter([classify_element(r['rpci'], r['gap_est'], r['z'])
                            for r in elem_src]).most_common(1)[0][0] if elem_src else '不明'

        adj_lpis = [apply_venue_bonus(target_venue, r['elem'], r['lpi'], bonus_strength)
                    for r in use]
        tw = sum(r['grade_weight'] for r in use)
        avg_venue = round(sum(a*r['grade_weight'] for a,r in zip(adj_lpis, use))/tw, 1) \
                    if tw > 0 else round(np.mean(adj_lpis), 1)

        coef = get_venue_bonus(target_venue, dom_elem)

        # ポジション予測（有効走の地点差から）
        past_gaps_for_pred = [r['gap_est'] for r in use
                               if r['gap_est'] < 5.0][:5]  # 大外れ値除外
        pos_pred = predict_position(past_gaps_for_pred, pace_pred_rpci)

        # 上がり予測（過去走のZスコアから）
        _pred_rpci = pace_pred_rpci
        _pace_cat  = 'H' if _pred_rpci <= 47 else ('S' if _pred_rpci >= 54 else 'M')

        _pred_gap = pos_pred['pred_gap'] if pos_pred else None
        _pci_cs_score = None
        if target_front_1f_input is not None and target_front_1f_input > 0:
            _cs = calc_pci_cs(use[:5], target_front_1f_input)
            _pci_cs_score = _cs['score'] if _cs else None

        _target_dist = target_dist if target_dist is not None else float(str(run_data[0]['dist']).replace('m',''))
        agari_pred = predict_agari(
            past_runs           = use[:5],
            target_dist         = _target_dist,
            target_venue        = target_venue,
            target_baba         = '良',
            predicted_pace_cat  = _pace_cat,
            pred_gap            = _pred_gap,
            pci_cs_score        = _pci_cs_score,
            cushion_correction  = cushion_correction_input,
            base_dict_for_z     = base_dict,
            pace_target_front_1f= target_front_1f_input,
        )

        # ===== 好走LPIボーナス（既定で無効化・検証済み）=====
        GRADE_RANK_BONUS_TABLE = {
            'G1': {1: 3.0, 2: 2.0, 3: 1.0},
            'G2': {1: 2.0, 2: 1.3, 3: 0.7},
            'G3': {1: 1.5, 2: 1.0, 3: 0.5},
            'L':  {1: 1.0, 2: 0.6, 3: 0.3},
        }
        g1_lpi_bonus = 0.0
        g1_bonus_detail = []
        for rn in run_data[:5]:
            if rn.get('excluded_track'):  # トラック不一致の実績は混ぜない
                continue
            grade = rn['grade']
            if grade not in GRADE_RANK_BONUS_TABLE:
                continue
            if rn['rank_int'] and rn['rank_int'] <= 3:
                b = GRADE_RANK_BONUS_TABLE[grade].get(rn['rank_int'], 0)
                if b > 0:
                    g1_lpi_bonus += b
                    g1_bonus_detail.append(f"{rn['race']}_{rn['rank_int']}着+{b}")
        g1_lpi_bonus = min(g1_lpi_bonus, 8.0)

        streak_bonus = 0.0
        streak_detail = ''
        streak_len = 0
        for rn in run_data[:5]:
            if rn.get('excluded_track'):
                continue
            if rn['rank_int'] and rn['rank_int'] <= 3:
                streak_len += 1
            else:
                break
        if streak_len >= 2:
            streak_bonus = min((streak_len - 1) * 1.0, 4.0)
            streak_detail = f'直近{streak_len}走連続3着以内+{streak_bonus:.1f}'
        g1_lpi_bonus = min(g1_lpi_bonus + streak_bonus, 10.0)
        if streak_detail:
            g1_bonus_detail.append(streak_detail)

        if disable_g1_streak_bonus:
            g1_lpi_bonus = 0.0
            g1_bonus_detail = []

        # ボーナスをLPIに加算（上限100）
        avg_lpi_adj      = min(100.0, round(avg_lpi      + g1_lpi_bonus, 1))
        avg_venue_lpi_adj = min(100.0, round(avg_venue   + g1_lpi_bonus, 1))

        # 展開ボーナス（検証済み・既定で有効）
        if pace_elem_adv and pace_bonus_strength:
            avg_venue_lpi_adj = apply_pace_bonus(dom_elem, pace_elem_adv, avg_venue_lpi_adj,
                                                  strength=pace_bonus_strength)

        results.append({
            'horse': horse, 'sex': sex, 'age': age,
            'avg_lpi': avg_lpi_adj, 'avg_venue_lpi': avg_venue_lpi_adj,
            'avg_lpi_raw': avg_lpi, 'avg_venue_raw': avg_venue,
            'g1_lpi_bonus': round(g1_lpi_bonus, 1),
            'g1_bonus_detail': ' / '.join(g1_bonus_detail),
            'max_lpi': round(max(r['lpi'] for r in run_data), 1),
            'latest_lpi': run_data[0]['lpi'],
            'n_valid': len(valid), 'n_total': len(run_data), 'n_good': len(good),
            'dom_elem': dom_elem, 'coef': round(coef, 2),
            'venue_delta': round(avg_venue - avg_lpi, 1),
            'pos_pred':   pos_pred,
            'agari_pred': agari_pred,
            'runs': run_data, 'valid_runs': valid, 'good_runs': good,
            'pci_cs_runs': use[:5],
        })

    results.sort(key=lambda x: -x['avg_venue_lpi'])
    return results

# ============================================================
# グラフ描画
# ============================================================
def plot_ranking(results, race_name, target_venue):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.font_manager as fm
    jp_fonts = ['Noto Sans CJK JP','IPAexGothic','IPAPGothic',
                'Hiragino Sans','Yu Gothic','Meiryo','MS Gothic']
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((f for f in jp_fonts if f in available), None)
    if chosen:
        matplotlib.rcParams['font.family'] = chosen
    else:
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    matplotlib.rcParams['axes.unicode_minus'] = False

    names   = [r['horse'] for r in results]
    avgs_v  = [r['avg_venue_lpi'] for r in results]
    avgs_b  = [r['avg_lpi'] for r in results]
    elems   = [r['dom_elem'] for r in results]
    n = len(names)

    colors = [ELEM_COLOR.get(e, '#888') for e in elems]
    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.45 + 1.5)))

    y = list(range(n))
    ax.barh(y, avgs_b,  color='#e0e0e0', edgecolor='none', height=0.65, label='基本LPI')
    ax.barh(y, avgs_v,  color=colors,    edgecolor='none', height=0.42,
            alpha=0.92, label=f'{target_venue}補正LPI')
    ax.axvline(80, color='#E24B4A', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_yticks(y)
    if chosen:
        ax.set_yticklabels([f'{i+1}. {n}' for i,n in enumerate(names)], fontsize=9)
    else:
        ax.set_yticklabels([f'{i+1}.' for i in range(len(names))], fontsize=9)
        for i, name in enumerate(names):
            ax.text(56.5, i, name, va='center', fontsize=7,
                    color='gray', fontfamily='DejaVu Sans')
    ax.set_xlabel('LPI スコア', fontsize=10)
    ax.set_xlim(55, 96)
    ax.set_title(f'{race_name}  [{target_venue}適合補正]  LPI v11', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.2)

    for bar, val, dval in zip(
        ax.patches[n:],
        avgs_v,
        [v-b for v,b in zip(avgs_v, avgs_b)]
    ):
        sign = '+' if dval >= 0 else ''
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f} ({sign}{dval:.1f})',
                va='center', fontsize=8)

    patches = [mpatches.Patch(color=c, label=e) for e,c in ELEM_COLOR.items() if e != '不明']
    ax.legend(handles=patches, fontsize=7, loc='lower right', ncol=2)
    plt.tight_layout()
    return fig

# ============================================================
# Streamlit UI
# ============================================================
# ============================================================
# 1日厳選レース機能 v5 用の関数定義
# (split_multi_race_csv, parse_jra_program はv4由来、
#  classify_race_class/passes_race_filter/select_top_gap_races はv5で複合フィルター対応)
# ============================================================

def split_multi_race_csv(file_bytes):
    """
    「枠番」ヘッダー行が複数回出現する、1日分の全レースが縦に連結された
    CSVを、レースごとのDataFrameに分割する。
    """
    for enc in ['cp932', 'shift_jis', 'utf-8-sig', 'utf-8']:
        try:
            df_raw = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            break
        except Exception:
            continue
    else:
        raise ValueError('文字コードを判定できませんでした')

    if '枠番' not in df_raw.columns:
        raise ValueError('「枠番」列が見つかりません。レース区切りを検出できる形式のCSVをアップロードしてください。')

    header_rows = df_raw[df_raw['枠番'].astype(str) == '枠番'].index.tolist()
    boundaries = [0] + header_rows + [len(df_raw)]

    races = []
    block_no = 1
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        block = df_raw.iloc[start:end].copy()
        if len(block) > 0 and str(block.iloc[0]['枠番']) == '枠番':
            block = block.iloc[1:].copy()
        block = block.dropna(subset=['馬名S'])
        block = block[block['馬名S'].astype(str) != '馬名S'].reset_index(drop=True)
        if len(block) < 2:
            continue
        races.append((block_no, block))
        block_no += 1
    return races


# ============================================================
# JRA番組表テキストのパース(クラス名抽出を追加)
# ============================================================
VENUE_NAMES = ['東京', '中山', '京都', '阪神', '新潟', '中京', '福島', '小倉', '札幌', '函館']

DIST_TRACK_RE = re.compile(
    r'([\d,，]{3,5})\s*[（(]\s*(芝|ダート|ダ)(?:[・][^）)]*)?\s*[)）]'
)

def classify_race_class(text):
    """番組表のクラス名テキストから大まかなクラス区分を判定する。(v5版が下で上書きするので参照のみ)"""
    s = str(text)
    if '新馬' in s: return '新馬'
    if '未勝利' in s: return '未勝利'
    if '1勝' in s or '１勝' in s: return '1勝クラス'
    if '2勝' in s or '２勝' in s: return '2勝クラス'
    if '3勝' in s or '３勝' in s: return '3勝クラス'
    if re.search(r'G[1-3]', s): return re.search(r'G[1-3]', s).group()
    if 'オープン' in s or 'Ｌ' in s or '(L)' in s: return 'オープン特別等'
    return '不明'

def parse_jra_program(text):
    """
    JRA公式サイトの「開催日程（番組表）」ページからブラウザでそのままコピペした
    テキストを解析し、会場ごとのレース一覧（R番号・距離・トラック・クラス）を返す。

    実際のコピペ結果は「レース番号」「レース」「クラス名」「距離（芝/ダ）」「発走時刻」
    がそれぞれ別行になる(表のセルが縦に展開される)ため、行単位ではなく
    「数字だけの行の直後に'レース'という行がある」＝レコード開始、として検出し、
    そこから数行以内にある距離パターンを探す方式にしている。
    クラス名は、レコード開始行から距離パターンが見つかった行までの間の
    テキストを結合して判定する。

    Returns: list of dict [{'venue','race_no','dist','track','race_class'}], 出現順
    """
    lines = [l.strip() for l in text.splitlines()]
    results = []
    current_venue = None
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]
        if not line:
            i += 1
            continue

        is_record_start = (
            bool(re.fullmatch(r'\d{1,2}', line))
            and i + 1 < n
            and lines[i + 1].strip() == 'レース'
        )
        if not is_record_start:
            for v in VENUE_NAMES:
                if v in line:
                    current_venue = v
                    break
            i += 1
            continue

        race_no = int(line)
        j = i + 2
        found = False
        class_text_parts = []
        search_limit = min(j + 8, n)
        while j < search_limit:
            m = DIST_TRACK_RE.search(lines[j])
            if m:
                dist = float(m.group(1).replace(',', '').replace('，', ''))
                track = 'D' if m.group(2) in ('ダート', 'ダ') else 'T'
                race_class = classify_race_class(' '.join(class_text_parts))
                results.append({
                    'venue': current_venue or '(不明)',
                    'race_no': race_no,
                    'dist': dist,
                    'track': track,
                    'race_class': race_class,
                })
                found = True
                j += 1
                break
            class_text_parts.append(lines[j])
            j += 1
        i = j if found else i + 1

    return results




# ============================================================
# クラス判定(未勝利・新馬・1勝〜3勝クラス・重賞等を区別する)
# ============================================================
def classify_race_class(text):
    """番組表のクラス名テキストから大まかなクラス区分を判定する。
    (parse_jra_program内の同名関数と同じロジック。ここでは候補フィルター用に単体でも呼べるようにしている)
    """
    s = str(text)
    if '新馬' in s: return '新馬'
    if '未勝利' in s: return '未勝利'
    if '1勝' in s or '１勝' in s: return '1勝クラス'
    if '2勝' in s or '２勝' in s: return '2勝クラス'
    if '3勝' in s or '３勝' in s: return '3勝クラス'
    if re.search(r'G[1-3]', s): return re.search(r'G[1-3]', s).group()
    if 'オープン' in s or 'Ｌ' in s or '(L)' in s: return 'オープン特別等'
    return '不明'


CLASS_1PLUS = {'1勝クラス', '2勝クラス', '3勝クラス', 'L', 'オープン特別等', 'G1', 'G2', 'G3'}


def passes_race_filter(dist, n_horses, venue, race_class,
                        min_horses=11, max_dist=1400,
                        exclude_venues=('東京',), allowed_classes=CLASS_1PLUS):
    """
    検証済みの複合フィルター。4条件すべてを満たすレースだけTrueを返す。
    厳選対象の「候補プール」を絞り込むために使う(軸・相手の選び方自体は変えない)。
    """
    if n_horses < min_horses:
        return False
    if dist is not None and dist > max_dist:
        return False
    if venue in exclude_venues:
        return False
    if race_class not in allowed_classes:
        return False
    return True


def select_top_gap_races(race_lpi_results, n_select=3, use_race_filter=True,
                          min_horses=11, max_dist=1400,
                          exclude_venues=('東京',), allowed_classes=CLASS_1PLUS):
    """
    各レースのLPI計算結果から、LPI1位と2位のスコア差(gap)が大きい順にn_select件を選ぶ。
    use_race_filter=True(既定)の場合、検証済みの複合フィルター
    (頭数11以上・距離1400m以下・東京以外・1勝クラス以上)を先にかけてから選定する。
    """
    scored = []
    for r in race_lpi_results:
        ranked = r['ranked']
        if len(ranked) < 2:
            continue
        if use_race_filter:
            if not passes_race_filter(
                dist=r.get('dist'), n_horses=r.get('n_horses', len(ranked)),
                venue=r.get('venue'), race_class=r.get('class'),
                min_horses=min_horses, max_dist=max_dist,
                exclude_venues=exclude_venues, allowed_classes=allowed_classes,
            ):
                continue
        gap = ranked[0]['avg_venue_lpi'] - ranked[1]['avg_venue_lpi']
        scored.append({**r, 'gap': gap})
    scored.sort(key=lambda x: -x['gap'])
    return scored[:n_select]


# ============================================================
# DN形式ファイル(JRA-VAN等の出馬表エクスポート、固定幅TXT)のパーサー
# ------------------------------------------------------------
# 過去走データ(33列・ヘッダーなしCSV)と、当日の全レース(DN形式TXT)から、
# 直接LPI計算用のエントリーを組み立てる(中間CSVの手動生成が不要)。
# ============================================================

# 過去走データの列マッピング(選択済み項目33項目の順番通り。実データ突き合わせで確認済み)
HIST_COL_年 = 0; HIST_COL_月 = 1; HIST_COL_日 = 2; HIST_COL_場所 = 4; HIST_COL_芝ダ = 6
HIST_COL_距離 = 7; HIST_COL_馬場状態 = 8; HIST_COL_馬名 = 9; HIST_COL_確定着順 = 16
HIST_COL_上がり3F = 27; HIST_COL_地点差 = 30; HIST_COL_PCI = 31; HIST_COL_RPCI = 32

DN_HEADER_RE = re.compile(
    r'(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日\([^)]+\)\s*(\d)回(\S+?)(\d+)日目\s*([\d:]+)発走'
)
DN_RACE_NO_RE = re.compile(r'([０-９\d]+)Ｒ')
DN_CLASS_DIST_RE = re.compile(r'(.+?)\s*(芝|ダ)\s*(\d+)m\s*(\d+)頭立')
DN_ENTRY_RE = re.compile(
    r'^\s*(?:B)?(\d+)\s+(\d+)\$?\s*(\S+?)\s+(牡|牝|セ)(\d+)\s*\*?(\S+?)\s*(\d+(?:\.\d+)?)',
    re.MULTILINE
)


def zen2han(s):
    return s.translate(str.maketrans('０１２３４５６７８９', '0123456789'))


@st.cache_data
def load_history_data(file_bytes):
    """過去走データ(33列・ヘッダーなしCSV)を読み込み、馬名検索用に整形する。"""
    for enc in ['cp932', 'shift_jis', 'utf-8-sig', 'utf-8']:
        try:
            hist = pd.read_csv(io.BytesIO(file_bytes), encoding=enc, header=None)
            break
        except (UnicodeDecodeError, Exception):
            continue
    else:
        raise ValueError('過去走データCSVの文字コードを判定できませんでした')

    if hist.shape[1] != 33:
        raise ValueError(f'過去走データの列数が{hist.shape[1]}列です(想定は33列)。ファイル形式を確認してください。')

    hist['日付_num'] = hist[HIST_COL_年] * 10000 + hist[HIST_COL_月] * 100 + hist[HIST_COL_日]
    hist['上がり_num'] = pd.to_numeric(hist[HIST_COL_上がり3F], errors='coerce')
    hist['レース名_簡易'] = (hist[HIST_COL_場所].astype(str) + hist[HIST_COL_距離].astype(str)
                            + hist[HIST_COL_芝ダ].astype(str))
    return hist.dropna(subset=['上がり_num'])


def build_walk_columns_from_history(hist_valid, horse_name, n_past_runs=5):
    """馬名から、直近n_past_runs走ぶんのWALK_DEFS互換列(dict)を作る。"""
    sub = (hist_valid[hist_valid[HIST_COL_馬名] == horse_name]
           .sort_values('日付_num', ascending=False)
           .head(n_past_runs)
           .reset_index(drop=True))
    rec = {}
    for n in range(1, n_past_runs + 1):
        suffix = '' if n == 1 else f'.{n-1}'
        if n - 1 < len(sub):
            r = sub.iloc[n - 1]
            rec[f'上り3F{suffix}']    = r[HIST_COL_上がり3F]
            rec[f'RPCI{suffix}']      = r[HIST_COL_RPCI]
            if n > 1:
                rec[f'PCI{suffix}']  = r[HIST_COL_PCI]
            rec[f'場所{suffix}']      = r[HIST_COL_場所]
            rec[f'距離{suffix}']      = r[HIST_COL_距離]
            rec[f'馬場状態{suffix}']  = r[HIST_COL_馬場状態]
            rec[f'着順{suffix}']      = r[HIST_COL_確定着順]
            rec[f'ﾚｰｽ名･{n}走前']    = r['レース名_簡易']
            rec[f'TD{suffix}']        = r[HIST_COL_芝ダ]
            rec[f'-3F差{suffix}']     = r[HIST_COL_地点差]
        else:
            keys = ['上り3F', 'RPCI', '場所', '距離', '馬場状態', '着順', 'TD', '-3F差']
            if n > 1:
                keys.append('PCI')
            for key in keys:
                rec[f'{key}{suffix}'] = np.nan
            rec[f'ﾚｰｽ名･{n}走前'] = np.nan
    return rec


def parse_dn_file(dn_bytes):
    """
    DN形式ファイル(JRA-VAN等の出馬表エクスポート、固定幅TXT)を、
    レースごとのメタ情報(会場・R・クラス・距離・トラック・頭数)と
    出走馬一覧(枠番・馬番・馬名・性別・年齢・騎手・斤量)に分解する。

    Returns: list of dict [{'venue','race_no','race_name','race_class',
                             'track','dist','n_horses','n_entries_parsed','entries'}]
    """
    for enc in ['cp932', 'shift_jis', 'utf-8-sig', 'utf-8']:
        try:
            text = dn_bytes.decode(enc, errors='strict')
            break
        except (UnicodeDecodeError, Exception):
            text = dn_bytes.decode('cp932', errors='replace')
            break

    header_matches = list(DN_HEADER_RE.finditer(text))
    if not header_matches:
        raise ValueError('DN形式のレースヘッダーが見つかりませんでした。ファイル形式を確認してください。')

    races = []
    for i, m in enumerate(header_matches):
        start = m.end()
        end = header_matches[i + 1].start() if i + 1 < len(header_matches) else len(text)
        body = text[start:end]

        year, month, day, kai, venue, nichi, time_ = m.groups()
        rno_m = DN_RACE_NO_RE.search(body)
        race_no = int(zen2han(rno_m.group(1))) if rno_m else None

        cd_m = DN_CLASS_DIST_RE.search(body)
        if cd_m:
            race_name_raw, track_jp, dist, n_horses = cd_m.groups()
            track = 'T' if track_jp == '芝' else 'D'
            dist = int(dist)
            n_horses = int(n_horses)
        else:
            race_name_raw, track, dist, n_horses = '', 'T', None, None

        entries = DN_ENTRY_RE.findall(body)
        races.append({
            'date': f'{year}{int(month):02d}{int(day):02d}',
            'venue': venue, 'race_no': race_no, 'race_name': race_name_raw.strip(),
            'race_class': classify_race_class(race_name_raw),
            'track': track, 'dist': dist, 'n_horses': n_horses,
            'n_entries_parsed': len(entries),
            'entries': [
                {'waku': e[0], 'umaban': e[1], 'horse': e[2], 'sex': e[3], 'age': e[4],
                 'jockey': e[5], 'weight': e[6]}
                for e in entries
            ],
        })
    return races




st.title('🏇 LPI v11 競馬予想ツール')
st.caption('LPI (ラップ強さ指数) v11 — 展開ボーナス対応版。2024-2025年バックテスト済み設定を既定値として採用。')

tab_single, tab_daily = st.tabs(['🔍 単一レース予想', '📅 1日厳選レース'])

with tab_single:

    # ---- サイドバー ----
    with st.sidebar:
        st.header('⚙️ 設定')

        st.subheader('① 基準テーブル用CSV')
        st.caption('⚠️ 平場（未勝利〜3勝クラス）水準のデータ推奨。重賞のみだと較正がズレます。')
        base_file = st.file_uploader(
            '2020〜2025年など、複数年分の平場結果CSVをアップ',
            type='csv', key='base')
        if base_file:
            st.success(f'{base_file.name} 読み込み済み')

        st.subheader('② 出走表CSV')
        entry_file = st.file_uploader(
            '予想するレースの出走表CSVをアップ',
            type='csv', key='entry')

        st.subheader('③ レース設定')
        race_name    = st.text_input('レース名', value='2026 レース名')
        target_venue = st.selectbox(
            '競馬場',
            ['東京','中山','京都','阪神','中京','新潟','福島','小倉','札幌','函館'])
        target_track = st.radio('トラック', ['T（芝）','D（ダート）'])
        track_code   = 'T' if target_track.startswith('T') else 'D'
        bonus_strength = st.slider(
            '競馬場ボーナス強度', 0.0, 0.30, 0.15, 0.05,
            help='0=ボーナスなし / 0.15=標準(検証済み・推奨) / 0.30=強め。'
                 '検証済み: このボーナスを外すと馬単回収率が大幅に悪化します。')

        st.subheader('④ ペース予測（自動判定・上書き可）')
        race_dist = st.number_input('レース距離（m）', min_value=1000, max_value=3600, value=1600, step=200)

        auto_pace = st.checkbox('出走馬の脚質から自動でペースを判定する', value=True,
                                 help='検証済み(展開ボーナス): 各馬の過去走地点差から脚質を自動判定し、'
                                      'ペース予測・展開適合ボーナスに反映します。')
        if not auto_pace:
            nige_count   = st.number_input('逃げ馬頭数', min_value=0, max_value=10, value=0, step=1)
            senkou_count = st.number_input('先行馬頭数', min_value=0, max_value=16, value=0, step=1)
        else:
            nige_count, senkou_count = 0, 0  # 自動判定時は実行時に上書きする

        use_pace_bonus = st.checkbox('展開適合ボーナスをLPIに反映する', value=True,
                                      help='検証済み: 有効にすると馬単回収率70.8%→82.3%(2024-2025年バックテスト)')
        pace_bonus_strength_input = st.slider('展開ボーナスの強さ', 0.0, 6.0, 3.0, 0.5,
                                               disabled=not use_pace_bonus)

        st.markdown('**ペース直接指定（任意・自動判定を上書き）**')
        manual_pace = st.radio(
            'ペース帯を選択',
            options=['自動推定（コース統計/脚質判定から）', '🔵 H（ハイ）', '🟢 M（ミドル）', '🟠 S（スロー）'],
            index=0,
            horizontal=False,
        )
        if '自動' in manual_pace:
            manual_rpci = 0.0
        elif 'H' in manual_pace:
            manual_rpci = 45.0
        elif 'M' in manual_pace:
            manual_rpci = 51.0
        else:
            manual_rpci = 57.0

        st.subheader('⑤ 好走ボーナス（既定オフ・検証済み）')
        enable_g1_streak_bonus = st.checkbox(
            'G1好走・連続好走ボーナスを使う', value=False,
            help='検証済み: オフの方が馬連回収率が2024-2025年両年で一貫して良い結果でした'
                 '(62.8%→65.6%, 94.2%→102.7%)。会場適性ボーナスとは別物です。'
        )

        st.subheader('⑥ PCI追走スコア（任意）')
        use_pci_cs = st.checkbox(
            '逃げ馬ペースへの追走能力を評価する',
            value=False,
        )
        if use_pci_cs:
            target_front_1f = st.number_input(
                '逃げ馬の想定前半1F（秒）',
                min_value=10.5, max_value=13.5, value=11.9, step=0.05,
            )
        else:
            target_front_1f = None

        st.subheader('⑦ クッション値（任意）')
        use_cushion = st.checkbox('クッション値で基準上がりを補正する', value=False)
        if use_cushion:
            cushion_val = st.number_input('クッション値', min_value=5.0, max_value=14.0, value=9.0, step=0.1)
            cushion_adj = round((9.0 - cushion_val) * 0.15, 3)
            if cushion_adj > 0:
                st.caption(f'クッション値{cushion_val:.1f} → 基準上がり{cushion_adj:+.2f}秒（遅い馬場）')
            elif cushion_adj < 0:
                st.caption(f'クッション値{cushion_val:.1f} → 基準上がり{cushion_adj:+.2f}秒（速い馬場）')
            else:
                st.caption('クッション値9.0 → 補正なし（標準）')
        else:
            cushion_val  = 9.0
            cushion_adj  = 0.0

        run_btn = st.button('🔍 LPI計算実行', type='primary', use_container_width=True)

    # ---- メインエリア ----
    if not base_file:
        st.info('← サイドバーから基準テーブル用CSVをアップしてください（平場水準のデータ推奨）。'
                '基準テーブルだけで厳選レースを使いたい場合は「📅 1日厳選レース」タブへどうぞ。')
    elif not entry_file:
        st.info('← サイドバーから出走表CSVをアップしてください')
    elif run_btn or (base_file and entry_file):
        entry_bytes_for_pace = entry_file.read()
        entry_file.seek(0)

        # ---- ペース予測 ----
        if manual_rpci > 0:
            _mr = manual_rpci
            if _mr <= 47:
                _label, _lamp = 'ハイペース（直接指定）', '🔵'
                _elem_adv = ['基礎スピード・パワー', 'パワー・ロンスパ']
                _comment  = 'H（ハイ）指定 — 前半速く先行馬が消耗。基礎スピード型・差し馬有利'
            elif _mr >= 54:
                _label, _lamp = 'スローペース（直接指定）', '🟠'
                _elem_adv = ['ギアチェンジ', 'ロンスパ・ギアチェンジ']
                _comment  = 'S（スロー）指定 — 上がり勝負。GC型有利、先行馬の前残りも警戒'
            else:
                _label, _lamp = 'ミドルペース（直接指定）', '🟢'
                _elem_adv = []
                _comment  = 'M（ミドル）指定 — 平均的なペース。どちらも起こりうる'
            pace = dict(pred_rpci=_mr, base_rpci=_mr, std=0.0,
                        slow_pct=100 if _mr>=54 else 0,
                        fast_pct=100 if _mr<=47 else 0,
                        label=_label, lamp=_lamp,
                        elem_adv=_elem_adv, comment=_comment)
            auto_style_info = None
        elif auto_pace:
            # 展開ボーナス: 2パス構成(1パス目=脚質自動判定)
            try:
                auto_style_info = precompute_running_styles(entry_bytes_for_pace)
                pace = get_pace_prediction(race_dist, target_venue,
                                            auto_style_info['nige'], auto_style_info['senkou'])
            except Exception as e:
                st.warning(f'脚質自動判定に失敗したため手動値(0,0)で計算します: {e}')
                auto_style_info = None
                pace = get_pace_prediction(race_dist, target_venue, 0, 0)
        else:
            pace = get_pace_prediction(race_dist, target_venue, nige_count, senkou_count)
            auto_style_info = None

        with st.spinner('基準テーブルを構築中...'):
            base_dict, 稍重_dict, race_base_dict = build_base_table(base_file.read())
            base_file.seek(0)  # 再読み込みのためリセット

        with st.spinner('LPI計算中...'):
            results = calc_lpi(
                entry_file.read(),
                base_dict, 稍重_dict,
                target_track=track_code,
                target_venue=target_venue,
                bonus_strength=bonus_strength,
                pace_pred_rpci=pace['pred_rpci'],
                race_base_dict=race_base_dict,
                target_race_name=race_name,
                target_front_1f_input=target_front_1f if use_pci_cs else None,
                cushion_correction_input=cushion_adj if use_cushion else 0.0,
                pace_elem_adv=pace['elem_adv'] if use_pace_bonus else None,
                pace_bonus_strength=pace_bonus_strength_input if use_pace_bonus else 0.0,
                disable_g1_streak_bonus=not enable_g1_streak_bonus,
                target_dist=float(race_dist),
            )

        if not results:
            st.error('計算できるデータがありませんでした。CSVの形式を確認してください。')
            st.stop()

        st.success(f'✅ {len(results)}頭 計算完了')

        if auto_style_info:
            st.caption(
                f"🐎 脚質自動判定結果: 逃げ{auto_style_info['nige']}頭 / 先行{auto_style_info['senkou']}頭 / "
                f"中団{auto_style_info['chudan']}頭 / 後方{auto_style_info['oikomi']}頭 / 不明{auto_style_info['fumei']}頭"
            )

        # ---- ペース予測バナー ----
        lamp_color  = {'🟠': '#E65100', '🔵': '#0D47A1', '⚪': '#424242'}
        border_color = {'🟠': '#FF6D00', '🔵': '#1565C0', '⚪': '#616161'}
        bc  = lamp_color.get(pace['lamp'], '#424242')
        brd = border_color.get(pace['lamp'], '#616161')

        adv_html = ''
        if pace['elem_adv']:
            adv_html = '  有利な要素型: **' + ' / '.join(pace['elem_adv']) + '**'

        cushion_html = ''
        if use_cushion and cushion_adj != 0.0:
            c_label = '軟らかめ' if cushion_val < 8 else ('やや軟らかめ' if cushion_val < 9 else ('やや硬め' if cushion_val < 11 else '硬め'))
            cushion_html = f'　🌿 クッション値{cushion_val:.1f}({c_label}) → 基準上がり{cushion_adj:+.2f}秒補正'

        st.markdown(
            f"""<div style="background:{bc};border:2px solid {brd};border-radius:8px;
            padding:12px 16px;margin:8px 0;font-size:13px;line-height:1.8;color:#FFFFFF;font-weight:500">
            <b>{pace['lamp']} ペース予測: {pace['label']}</b>　
            予測RPCI <b>{pace['pred_rpci']:.1f}</b>（基準{pace['base_rpci']:.1f}±{pace['std']:.1f}）<br>
            スロー率 <b>{pace['slow_pct']}%</b>　ハイ率 <b>{pace['fast_pct']}%</b><br>
            {pace['comment']}{adv_html}{cushion_html}
            </div>""",
            unsafe_allow_html=True
        )

        # ---- レース環境スコア（堅実軸）バナー ----
        _race_grade = extract_grade(race_name)
        race_env_score = calc_race_env_score(pace['pred_rpci'], race_dist, _race_grade, target_venue)
        if race_env_score <= 1:
            env_label = '🟢 堅実軸が機能しやすいレース'
            env_detail = ('少頭数寄り・Sペース寄り・マイル以上・主要場グレード戦の傾向'
                           'がそろっており、過去データではLPI上位馬の単勝回収率が高い（複勝率29.5%・回収率98.4%）。')
            env_color, env_border = '#1B5E20', '#2E7D32'
        elif race_env_score >= 3:
            env_label = '🔴 紛れが起きやすいレース（LPI上位でも過信注意）'
            env_detail = ('短距離・Hペース・G3・小場開催の条件が重なっており、'
                           '過去データではLPI上位馬でも単勝回収率が下がる傾向（複勝率は同水準・回収率66.1%）。'
                           'LPI下位の馬も含めて広めに見る方が無難。')
            env_color, env_border = '#B71C1C', '#C62828'
        else:
            env_label = '🟡 標準的な紛れやすさのレース'
            env_detail = '極端な傾向はない。通常通りLPI評価を参考にする。'
            env_color, env_border = '#5D4037', '#6D4C41'

        st.markdown(
            f"""<div style="background:{env_color};border:2px solid {env_border};border-radius:8px;
            padding:10px 16px;margin:4px 0 8px 0;font-size:12.5px;line-height:1.7;color:#FFFFFF;font-weight:500">
            <b>{env_label}</b>（環境スコア{race_env_score}/4）<br>
            {env_detail}
            </div>""",
            unsafe_allow_html=True
        )

        # ---- タブで表示 ----
        tab1, tab2, tab3, tab4 = st.tabs(['📊 ランキング表', '📈 グラフ', '🔍 過去走詳細', '🎲 シミュレーション'])

        # ===== タブ1: ランキング表 =====
        with tab1:
            st.subheader(f'{race_name}  LPI v11 ランキング')

            pci_cs_map = {}
            if use_pci_cs and target_front_1f:
                for r in results:
                    cs = calc_pci_cs(r.get('pci_cs_runs', []), target_front_1f)
                    pci_cs_map[r['horse']] = cs

            rows = []
            for i, r in enumerate(results):
                bonus_runs = [rn for rn in r['runs'] if rn.get('hb', 0) > 0]
                bonus_str  = ' / '.join(
                    [f"{rn.get('race','-')}({rn.get('hb_r','')})" for rn in bonus_runs])
                g1_bonus_str = (f"+{r['g1_lpi_bonus']:.1f}({r['g1_bonus_detail']})"
                               if r.get('g1_lpi_bonus', 0) > 0 else '-')
                past  = [rn for rn in r['runs']
                         if not rn.get('excluded_baba') and not rn.get('excluded_track')][:5]
                plpi  = [round(rn['lpi'], 1) for rn in past]
                while len(plpi) < 5: plpi.append('-')

                delta = r.get('venue_delta', 0.0)
                pace_match = r['dom_elem'] in pace['elem_adv'] if pace['elem_adv'] else None
                pace_mark = '◎' if pace_match else ('△' if pace_match is False else '-')

                if r.get('agari_pred'):
                    ap = r['agari_pred']
                    z_val   = round(ap['pred_z'], 3)
                    grade   = ap['grade_label']
                    conf    = ap['confidence']
                    n_valid = ap['n_valid']
                    n_disc  = ap.get('n_discounted', 0)
                    hg_only = '★G1/G2限定' if ap.get('comment','') and 'G1/G2走' in ap.get('comment','') else ''
                    matsu_str = f'{grade} {conf}  Z={z_val:+.3f}({n_valid}走{hg_only})'
                else:
                    matsu_str = '-'; z_val = '-'

                pci_str = (pci_cs_map[r['horse']]['judge'] + ' ' +
                           str(pci_cs_map[r['horse']]['score'])
                           + '（' + pci_cs_map[r['horse']]['detail'][:15] + '）')                        if r['horse'] in pci_cs_map else '-'

                if r.get('pos_pred'):
                    pp = r['pos_pred']
                    avg_gap = (sum(pp['past_gaps'])/len(pp['past_gaps'])
                               if pp['past_gaps'] else pp['pred_gap'])
                    pos_str = (f"{pp['icon']}{pp['zone_name']} {pp['confidence']}"
                               f"  予測{pp['pred_gap']:.1f}秒（平均{avg_gap:.1f}秒）")
                else:
                    pos_str = '-'

                pp = r.get('pos_pred')
                if pp and pp.get('past_gaps'):
                    gaps = pp['past_gaps']
                    avg_gap = sum(gaps)/len(gaps)
                    gap_label = ('🏇逃げ' if avg_gap<=0.1 else
                                 '🔵先行' if avg_gap<=0.6 else
                                 '🟡中団' if avg_gap<=1.2 else '🔴後方')
                    past_pos_str = f'{gap_label} 平均{avg_gap:.1f}秒({len(gaps)}走)'
                else:
                    past_pos_str = '-'

                if race_env_score <= 1 and (i + 1) <= 5:
                    kentaku_str = '🟢堅実軸'
                elif race_env_score >= 3 and (i + 1) <= 5:
                    kentaku_str = '🔴過信注意'
                else:
                    kentaku_str = '-'

                rows.append({
                    '順位':           i + 1,
                    '馬名':           r['horse'],
                    f'LPI[{target_venue}補正]': r['avg_venue_lpi'],
                    'LPI基本':        r['avg_lpi'],
                    '展開適合':       pace_mark,
                    'G1好走B':        g1_bonus_str,
                    '要素型':         r['dom_elem'],
                    '係数':           r['coef'],
                    '有効/全走':      f"{r['n_valid']}/{r['n_total']}",
                    '1走前':          plpi[0],
                    '2走前':          plpi[1],
                    '3走前':          plpi[2],
                    '4走前':          plpi[3],
                    '5走前':          plpi[4],
                    'PCI追走':        pci_str,
                    '末脚能力':       matsu_str,
                    '過去ポジション':  past_pos_str,
                    '不利ボーナス':   bonus_str,
                    '堅実軸':         kentaku_str,
                })

            result_df = pd.DataFrame(rows)
            lpi_col   = f'LPI[{target_venue}補正]'

            def highlight_with_t(row):
                try:
                    lpi_rank = int(row.get('順位', 99) or 99)
                except (TypeError, ValueError):
                    lpi_rank = 99
                if lpi_rank == 1: return ['background-color: #F9A825; color: #000; font-weight:bold'] * len(row)
                if lpi_rank == 2: return ['background-color: #1565C0; color: #fff; font-weight:bold'] * len(row)
                if lpi_rank == 3: return ['background-color: #BF360C; color: #fff; font-weight:bold'] * len(row)
                if lpi_rank <= 5: return ['background-color: #1B1B2F; color: #E0E0E0'] * len(row)
                return [''] * len(row)

            fmt = {lpi_col: '{:.1f}', 'LPI基本': '{:.1f}',
                   'LPI最高': '{:.1f}', 'LPI直近': '{:.1f}', '係数': '{:.2f}'}

            st.dataframe(
                result_df.style
                    .apply(highlight_with_t, axis=1)
                    .format(fmt, na_rep='-')
                    .set_properties(**{'border': '1px solid #444', 'font-size': '13px'})
                    .hide(axis='index'),
                use_container_width=True,
                height=min(600, 45 + len(rows) * 38),
            )

            buf = io.BytesIO()
            result_df.to_excel(buf, index=False)
            buf.seek(0)
            st.download_button(
                '📥 Excelダウンロード',
                data=buf,
                file_name=f'lpi_{race_name}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            )

        # ===== タブ2: グラフ =====
        with tab2:
            st.subheader('LPI ランキンググラフ')
            fig = plot_ranking(results, race_name, target_venue)
            st.pyplot(fig)

            with st.expander('要素型の見方'):
                st.markdown("""
    | 要素型 | 説明 | 東京係数 | 阪神係数 |
    |--------|------|---------|---------|
    | 🟢 ギアチェンジ | 後傾×差し+速上がり | **1.28** | **1.30** |
    | 🔵 ロンスパ・GC | 後傾×先行+速上がり | 1.20 | 1.29 |
    | 🟠 基礎スピード | 前傾×先行 | 0.77 | 1.05 |
    | 🔴 パワー・ロンスパ | 前傾×差し | 1.08 | 0.89 |
    | ⚫ ロンスパ | 後傾×差し+遅上がり | 0.44 | 0.39 |
    """)

        # ===== タブ3: 過去走詳細 =====
        with tab3:
            st.subheader('過去走 詳細データ')
            sel = st.selectbox('馬を選択', [r['horse'] for r in results])
            hr  = next((r for r in results if r['horse'] == sel), None)

            if hr:
                col1, col2, col3, col4 = st.columns(4)
                if hr.get('pos_pred'):
                    pp = hr['pos_pred']
                    st.markdown(
                        f"**予測ポジション:** {pp['icon']} {pp['label']}　"
                        f"予測地点差 **{pp['pred_gap']:.2f}秒**　"
                        f"安定度: **{pp['confidence']}**（過去{pp['n_valid']}走 std={pp['gap_std']:.3f}）",
                    )
                if hr.get('agari_pred'):
                    ap = hr['agari_pred']
                    pace_z_parts = []
                    for p, lbl in [('H','🔵ハイ'),('M','🟢ミドル'),('S','🟠スロー')]:
                        z_val = ap['z_by_pace'].get(p)
                        n_val = ap['n_by_pace'].get(p, 0)
                        if z_val is not None:
                            pace_z_parts.append(f'{lbl}: **{z_val:+.2f}** (n={n_val})')
                        else:
                            pace_z_parts.append(f'{lbl}: データなし')
                    pace_z_str = '　'.join(pace_z_parts)

                    st.markdown(
                        f"**上がり予測:** {ap['grade_label']}　{ap['confidence']}　"
                        f"予測上がり **{ap['pred_agari']}秒**（コース基準{ap['course_base']:.1f}秒）{ap.get('gap_note','')}\n\n"
                        f"ペース帯別Z → {pace_z_str}\n\n"
                        f"{ap['comment']}",
                    )
                col1.metric('LPI補正', f"{hr['avg_venue_lpi']:.1f}")
                col2.metric('LPI基本', f"{hr['avg_lpi']:.1f}")
                col3.metric('要素型',  hr['dom_elem'])
                col4.metric('有効走',  f"{hr['n_valid']}/{hr['n_total']}走（好走{hr['n_good']}）")

                st.markdown('---')
                run_rows = []
                for rn in hr['runs']:
                    excl_reason = []
                    if rn.get('excluded_baba'):  excl_reason.append('重/不良')
                    if rn.get('excluded_track'): excl_reason.append('トラック違い')
                    run_rows.append({
                        '走前':     rn.get('n', '-'),
                        'レース名': rn.get('race', '-'),
                        '競馬場':   rn.get('venue', '-'),
                        '距離':     int(rn['dist']) if rn.get('dist') is not None else '-',
                        '馬場':     rn.get('baba', '-'),
                        'RPCI':     rn.get('rpci', '-'),
                        '地点差':   rn.get('gap_est', '-'),
                        '上がり':   rn.get('agari', '-'),
                        '斤量補正': rn.get('wt_corr', 0.0),
                        'Zスコア':  rn.get('z', '-'),
                        'pb(位置補正)': rn.get('pb', 0.0),
                        'hb(不利B)':   rn.get('hb', 0.0),
                        'LPI':      rn.get('lpi', '-'),
                        '要素型':   rn.get('elem', '-'),
                        '前半速度Z': rn.get('front_pace_z', '-'),
                        '除外':     '⚠️ ' + '/'.join(excl_reason) if excl_reason else '✅',
                        '不利理由': rn.get('hb_r', ''),
                    })

                run_df = pd.DataFrame(run_rows)

                def highlight_run(row):
                    if row['除外'] != '✅':
                        return ['opacity: 0.4; color: gray'] * len(row)
                    return [''] * len(row)

                st.dataframe(
                    run_df.style
                        .apply(highlight_run, axis=1)
                        .format({'LPI': '{:.1f}', 'Zスコア': '{:.3f}',
                                 'pb(位置補正)': '{:.3f}', 'hb(不利B)': '{:.3f}',
                                 '斤量補正': '{:+.2f}'}),
                    use_container_width=True,
                )

        # ===== タブ4: モンテカルロシミュレーション =====
        with tab4:
            st.subheader('🎲 モンテカルロ シミュレーション')
            st.warning(
                '⚠️ 参考情報: このシミュレーション方式(予測地点差+予測上がりの絶対値を'
                '足し合わせて順位を決める方式)は、2024-2025年バックテストで'
                'LPIスコアによる順位付けより不安定という結果が出ています'
                '(年によって精度の方向が逆転)。参考程度に留めてください。'
            )
            st.markdown(
                '予測上がり・予測地点差の**誤差範囲でランダムにばらつかせて**1万回レースを試行し、'
                '各馬の勝利確率・複勝確率を推定します。'
            )

            sim_horses = [(r['horse'],
                           r['pos_pred']['pred_gap'],
                           r['agari_pred']['pred_agari'])
                          for r in results
                          if r.get('pos_pred') and r.get('agari_pred')]
            sim_stability = {
                r['horse']: {
                    'agari_conf': r['agari_pred'].get('confidence','△不安定'),
                    'gap_conf':   r['pos_pred'].get('confidence','△不安定'),
                    'z_std':      r['agari_pred'].get('z_std', 0.5),
                    'gap_std':    r['pos_pred'].get('gap_std', 0.5),
                }
                for r in results if r.get('pos_pred') and r.get('agari_pred')
            }

            if len(sim_horses) < 2:
                st.warning('予測ポジション・上がり予測が計算できた馬が2頭未満のため実行できません。')
            else:
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    n_trials = st.select_slider(
                        '試行回数',
                        options=[1000, 3000, 5000, 10000],
                        value=5000,
                    )
                with col_s2:
                    show_top3 = st.checkbox('複勝確率（3着以内）も表示', value=True)

                if st.button('▶️ シミュレーション実行', type='primary'):
                    names      = [h[0] for h in sim_horses]
                    pred_gaps  = np.array([h[1] for h in sim_horses])
                    pred_agari = np.array([h[2] for h in sim_horses])
                    pred_total = pred_gaps + pred_agari
                    n_horses   = len(names)

                    AGARI_BASE_STD = 1.065
                    GAP_BASE_STD   = 0.589

                    sigma_agari = np.array([
                        max(0.5, min(2.0,
                            r['agari_pred']['z_std'] * 1.4
                            if r.get('agari_pred') and r['agari_pred'].get('z_std')
                            else AGARI_BASE_STD))
                        for r in results if r.get('pos_pred') and r.get('agari_pred')
                    ])
                    sigma_gap = np.array([
                        max(0.1, min(1.2,
                            r['pos_pred']['gap_std']
                            if r.get('pos_pred') and r['pos_pred'].get('gap_std')
                            else GAP_BASE_STD))
                        for r in results if r.get('pos_pred') and r.get('agari_pred')
                    ])
                    z_stds = np.array([
                        r['agari_pred']['z_std']
                        if r.get('agari_pred') and r['agari_pred'].get('z_std')
                        else 0.5
                        for r in results if r.get('pos_pred') and r.get('agari_pred')
                    ])

                    wins  = np.zeros(n_horses)
                    top3s = np.zeros(n_horses)

                    np.random.seed(None)
                    with st.spinner(f'{n_trials:,}回シミュレーション中...'):
                        for _ in range(n_trials):
                            sim_gaps = np.maximum(0,
                                pred_gaps + np.random.normal(0, sigma_gap))

                            base_noise = np.random.normal(0, sigma_agari)
                            for i in range(n_horses):
                                if z_stds[i] > 0.7 and base_noise[i] < 0:
                                    base_noise[i] *= 0.5
                            sim_agaris = pred_agari + base_noise

                            sim_totals = sim_gaps + sim_agaris
                            order      = np.argsort(sim_totals)
                            wins[order[0]]   += 1
                            top3s[order[:3]] += 1

                    sim_rows = []
                    for i in range(n_horses):
                        wp = wins[i]  / n_trials * 100
                        pp = top3s[i] / n_trials * 100
                        lpi_rank = next((j+1 for j,r in enumerate(results) if r['horse']==names[i]), '-')
                        stab = sim_stability.get(names[i], {})
                        sim_rows.append({
                            '勝利確率順':  i+1,
                            '馬名':        names[i],
                            'LPI順位':     lpi_rank,
                            '予測通過T':   f'{pred_total[i]:.2f}秒',
                            '上がり安定':  stab.get('agari_conf','-'),
                            'gap安定':     stab.get('gap_conf','-'),
                            '勝利確率':    f'{wp:.1f}%',
                            '複勝確率':    f'{pp:.1f}%',
                            '勝利回数':    int(wins[i]),
                        })

                    sim_df = pd.DataFrame(sim_rows)
                    sim_df = sim_df.sort_values('勝利回数', ascending=False).reset_index(drop=True)
                    sim_df['勝利確率順'] = range(1, len(sim_df)+1)

                    def sim_highlight(row):
                        rank = row['勝利確率順']
                        if rank == 1: return ['background-color:#F9A825;color:#000;font-weight:bold']*len(row)
                        if rank == 2: return ['background-color:#1565C0;color:#fff;font-weight:bold']*len(row)
                        if rank == 3: return ['background-color:#BF360C;color:#fff;font-weight:bold']*len(row)
                        return ['']*len(row)

                    display_cols = ['勝利確率順','馬名','LPI順位','予測通過T',
                                    '上がり安定','gap安定','勝利確率']
                    if show_top3:
                        display_cols.append('複勝確率')

                    st.dataframe(
                        sim_df[display_cols].style
                            .apply(sim_highlight, axis=1)
                            .hide(axis='index'),
                        use_container_width=True,
                        height=min(600, 45 + len(sim_df)*38),
                    )

                    st.caption(
                        f'試行回数: {n_trials:,}回 ／ '
                        f'上がりσ: 馬個別（z_std×1.4, 範囲0.5〜2.0秒）／ '
                        f'地点差σ: 馬個別（gap_std, 範囲0.1〜1.2秒）／ '
                        f'不安定馬（z_std>0.7）は速い方向のばらつきを半減'
                    )
                    st.info(
                        '💡 勝利確率はLPI順位と異なる場合があります。'
                        '予測通過Tが近い馬は誤差の影響を受けやすく、確率が均等に近くなります。'
                        'LPI上位でも通過Tが遅い馬は勝利確率が低く出ます。'
                    )


with tab_daily:
    st.markdown('---')
    st.header('📅 1日厳選レース（v5: 頭数・距離・会場・クラスの複合フィルター対応）')
    st.caption(
        '1日分の全レースが連結された出走表CSVを読み込み、JRA公式サイトの番組表とレース単位で対応づけたうえで、'
        '検証済みの複合フィルター（頭数11頭以上・距離1400m以下・東京以外・1勝クラス以上）を満たすレースの中から、'
        'LPI1位と2位のスコア差(gap)が最も大きい上位レースを自動選定します。'
    )
    st.info(
        '💡 検証結果(2024-2025年・平場全芝・厳選3レース・軸+相手3頭流し):'
        '複合フィルターなしの馬単回収率88.7%・馬連回収率82.9%に対し、'
        'フィルターありでは馬単回収率127.2%・馬連回収率143.0%（2年間とも改善を確認済み）。'
    )

    with st.expander('📅 1日厳選レースを使う', expanded=False):

        st.markdown('**① 過去走データファイルをアップロード**')
        col_a, col_b = st.columns(2)
        with col_a:
            daily_base_file = st.file_uploader(
                '基準テーブル用CSV（平場水準のデータを推奨。重賞級のみだと較正がズレることを確認済み）',
                type='csv', key='daily_base')
        with col_b:
            daily_history_file = st.file_uploader(
                '過去走データファイル（33列・ヘッダーなし形式）', type=['csv', 'txt'], key='daily_history')

        col_c, col_d = st.columns(2)
        with col_c:
            daily_n_select = st.slider('厳選するレース数', 1, 10, 3, key='daily_n')
        with col_d:
            use_race_filter = st.checkbox(
                '検証済みの複合フィルターを使う', value=True, key='daily_use_filter',
                help='頭数11頭以上・距離1400m以下・東京以外・1勝クラス以上の4条件。'
                     'オフにすると未勝利のみ除外した従来方式になります。'
            )

        with st.expander('複合フィルターの詳細設定（通常は変更不要）'):
            filter_min_horses = st.number_input('最低頭数', min_value=1, max_value=18, value=11, key='filter_min_horses')
            filter_max_dist = st.number_input('最大距離(m)', min_value=1000, max_value=3600, value=1400, step=100, key='filter_max_dist')
            filter_exclude_tokyo = st.checkbox('東京を除外する', value=True, key='filter_exclude_tokyo')
            filter_exclude_maiden_only = st.checkbox(
                '未勝利・新馬のみ除外する(1勝クラス以上に絞らない)', value=False, key='filter_maiden_only',
                help='オンにすると、2勝クラス以上等ではなく単に未勝利・新馬だけを除外する緩めの設定になります。'
            )

        n_partners = st.slider('相手の頭数（軸+相手のn点流し）', 2, 5, 3, key='daily_n_partners')

        st.markdown('---')
        st.markdown(
            '**② DN形式ファイルをアップロード**　'
            'JRA-VAN等から出力した、当日の全レース分の出馬表テキスト（DN形式・.TXT）をそのままアップしてください。'
            '会場・R番号・クラス・距離・トラック・頭数・出走馬（枠番・馬番・馬名・性別・年齢・騎手・斤量）を'
            'このファイル1つから自動抽出します。番組表の手コピペや、別途出走表CSVを作る必要はありません。'
        )
        daily_dn_file = st.file_uploader(
            'DN形式ファイル（当日の全レース、.TXT）', type=['txt', 'csv'], key='daily_dn')

        split_btn = st.button('③ 解析 + 自動対応づけ', key='daily_split')

        if split_btn:
            if not daily_history_file:
                st.error('過去走データファイルをアップロードしてください。')
            elif not daily_dn_file:
                st.error('DN形式ファイルをアップロードしてください。')
            else:
                try:
                    hist_valid = load_history_data(daily_history_file.read())
                    daily_history_file.seek(0)
                except Exception as e:
                    st.error(f'過去走データの読み込みに失敗しました: {e}')
                    hist_valid = None

                try:
                    dn_races = parse_dn_file(daily_dn_file.read())
                    daily_dn_file.seek(0)
                except Exception as e:
                    st.error(f'DN形式ファイルの解析に失敗しました: {e}')
                    dn_races = []

                if hist_valid is None or not dn_races:
                    st.warning('解析できませんでした。ファイル形式を確認してください。')
                else:
                    n_mismatch = sum(1 for r in dn_races if r['n_entries_parsed'] != r['n_horses'])
                    if n_mismatch == 0:
                        st.success(f'✅ {len(dn_races)}レース、全て頭数と出走馬数が一致しました。')
                    else:
                        st.warning(
                            f'⚠️ {n_mismatch}レースで頭数と解析できた出走馬数が一致しませんでした。'
                            '下の表で頭数を確認してください（DN形式の想定外パターンの可能性があります）。'
                        )

                    mapping_rows = []
                    for i, r in enumerate(dn_races, start=1):
                        mapping_rows.append({
                            'block_no': i,
                            '頭数': r['n_entries_parsed'],
                            '会場': r['venue'] if r['venue'] in VENUE_NAMES else None,
                            'R': r['race_no'],
                            '距離': r['dist'],
                            'トラック': r['track'],
                            'クラス': r['race_class'],
                        })

                    st.session_state['daily_dn_races'] = dn_races
                    st.session_state['daily_hist_valid'] = hist_valid
                    st.session_state['daily_mapping_df'] = pd.DataFrame(mapping_rows)

        if 'daily_mapping_df' in st.session_state:
            st.markdown('**④ 対応づけの確認・修正**（頭数・クラスを見て、明らかにおかしい対応は修正してください）')
            edited_map = st.data_editor(
                st.session_state['daily_mapping_df'],
                column_config={
                    '会場': st.column_config.SelectboxColumn('会場', options=VENUE_NAMES),
                    'R': st.column_config.NumberColumn('R番号', min_value=1, max_value=12, step=1),
                    '距離': st.column_config.NumberColumn('距離(m)', min_value=800, max_value=3600, step=100),
                    'トラック': st.column_config.SelectboxColumn('トラック', options=['T', 'D']),
                    'クラス': st.column_config.SelectboxColumn(
                        'クラス', options=['新馬', '未勝利', '1勝クラス', '2勝クラス', '3勝クラス',
                                          'オープン特別等', 'G1', 'G2', 'G3', '不明']),
                },
                disabled=['block_no', '頭数'],
                hide_index=True,
                use_container_width=True,
                key='daily_map_editor',
            )

            missing = edited_map['距離'].isna().sum() + edited_map['会場'].isna().sum()
            if missing > 0:
                st.warning(f'⚠️ 未確定の項目が{missing}件あります。表を編集して埋めてから計算してください。')

            # フィルター条件に該当するレース数を事前表示
            if use_race_filter:
                allowed_cls = {'1勝クラス','2勝クラス','3勝クラス','L','オープン特別等','G1','G2','G3'}
                if filter_exclude_maiden_only:
                    allowed_cls = allowed_cls | {'不明'}
                n_pass = 0
                for _, row in edited_map.iterrows():
                    venue_ex = {'東京'} if filter_exclude_tokyo else set()
                    if passes_race_filter(row['距離'], row['頭数'], row['会場'], row['クラス'],
                                           min_horses=filter_min_horses, max_dist=filter_max_dist,
                                           exclude_venues=venue_ex, allowed_classes=allowed_cls):
                        n_pass += 1
                st.caption(f'🔍 複合フィルター該当レース: {n_pass}件 / 全{len(edited_map)}件')

            daily_run = st.button(
                '⑤ 厳選レースを計算する', type='primary', key='daily_run',
                disabled=(missing > 0),
            )

            if daily_run:
                if not daily_base_file:
                    st.error('基準テーブルCSVをアップロードしてください。')
                else:
                    with st.spinner('基準テーブルを構築中...'):
                        d_base_dict, d_稍重_dict, d_race_base_dict = build_base_table(daily_base_file.read())

                    dn_races = st.session_state['daily_dn_races']
                    hist_valid = st.session_state['daily_hist_valid']
                    map_by_block = {row['block_no']: row for _, row in edited_map.iterrows()}

                    race_lpi_results = []
                    progress = st.progress(0)
                    for i, race in enumerate(dn_races, start=1):
                        m = map_by_block.get(i)
                        if m is None or len(race['entries']) < 2:
                            progress.progress(i / len(dn_races))
                            continue

                        r_dist  = float(m['距離'])
                        r_track = m['トラック'] or 'T'
                        r_venue = m['会場']
                        r_no    = int(m['R']) if pd.notna(m['R']) else None
                        r_class = m['クラス']

                        try:
                            # 過去走データと突き合わせて、その場でWALK_DEFS横持ちの出走表を作る
                            out_rows = []
                            for e in race['entries']:
                                row = {'枠番': e['waku'], '馬名S': e['horse'],
                                       '性別': e['sex'], '年齢': e['age']}
                                row.update(build_walk_columns_from_history(hist_valid, e['horse']))
                                out_rows.append(row)
                            entry_df = pd.DataFrame(out_rows)
                            csv_bytes = entry_df.to_csv(index=False).encode('cp932', errors='replace')

                            styles = precompute_running_styles(csv_bytes)
                            pace = get_pace_prediction(r_dist, r_venue, styles.get('nige', 0), styles.get('senkou', 0))

                            results = calc_lpi(
                                csv_bytes, d_base_dict, d_稍重_dict,
                                target_track=r_track, target_venue=r_venue,
                                bonus_strength=0.15, pace_pred_rpci=pace['pred_rpci'],
                                race_base_dict=d_race_base_dict,
                                pace_elem_adv=pace['elem_adv'], pace_bonus_strength=3.0,
                                target_dist=r_dist,
                            )
                            ranked = sorted(results, key=lambda x: -x['avg_venue_lpi'])

                            if len(ranked) >= 2:
                                race_lpi_results.append({
                                    'block_no': i, 'race_label': f'{r_venue}{r_no}R' if r_no else r_venue,
                                    'n_horses': len(ranked), 'ranked': ranked,
                                    'dist': r_dist, 'track': r_track, 'venue': r_venue,
                                    'class': r_class, 'date': race.get('date'), 'race_no': r_no,
                                })
                        except Exception as e:
                            st.caption(f'block{i}: 計算スキップ（{e}）')
                        progress.progress(i / len(dn_races))

                    if not race_lpi_results:
                        st.warning('LPIを計算できたレースがありませんでした。')
                    else:
                        venue_ex = {'東京'} if filter_exclude_tokyo else set()
                        allowed_cls = {'1勝クラス','2勝クラス','3勝クラス','L','オープン特別等','G1','G2','G3'}
                        if filter_exclude_maiden_only:
                            allowed_cls = allowed_cls | {'不明'}

                        selected = select_top_gap_races(
                            race_lpi_results, n_select=daily_n_select,
                            use_race_filter=use_race_filter,
                            min_horses=filter_min_horses, max_dist=filter_max_dist,
                            exclude_venues=venue_ex, allowed_classes=allowed_cls,
                        )

                        if use_race_filter and not selected:
                            st.warning(
                                '⚠️ 複合フィルターに該当するレースがありませんでした。'
                                'フィルターをオフにするか、条件を緩めてください。'
                            )

                        st.subheader(f'🏆 本日の厳選{len(selected)}レース（gap = LPI1位と2位のスコア差）')
                        for s in selected:
                            ranked = s['ranked']
                            axis = ranked[0]
                            partners = ranked[1:1 + n_partners]
                            st.markdown(
                                f"**{s['race_label']}**　{int(s['dist'])}m {s['track']}　{s['class']}　"
                                f"gap = **{s['gap']:.1f}**　（出走{s['n_horses']}頭）"
                            )
                            rows = [{
                                'LPI順位': 1, '馬名': axis['horse'],
                                'LPI': axis['avg_venue_lpi'], '役割': '🎯 軸（1着固定）',
                            }]
                            for j, p in enumerate(partners, start=2):
                                rows.append({
                                    'LPI順位': j, '馬名': p['horse'],
                                    'LPI': p['avg_venue_lpi'], '役割': '相手候補（2着）',
                                })
                            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
                            st.caption(
                                f"馬単{n_partners}点: {axis['horse']} → "
                                f"{' / '.join(p['horse'] for p in partners)}"
                            )
                            st.markdown('')

                        st.session_state['daily_selected_for_log'] = selected
                        st.session_state['daily_n_partners_for_log'] = n_partners

                        st.subheader('📋 全レース一覧（gap順、複合フィルター適用状況つき）')
                        all_rows = []
                        for r in sorted(race_lpi_results,
                                         key=lambda x: -(x['ranked'][0]['avg_venue_lpi'] - x['ranked'][1]['avg_venue_lpi'])):
                            gap = r['ranked'][0]['avg_venue_lpi'] - r['ranked'][1]['avg_venue_lpi']
                            is_selected = r['block_no'] in [s['block_no'] for s in selected]
                            venue_ex_check = {'東京'} if filter_exclude_tokyo else set()
                            passes = passes_race_filter(
                                r['dist'], r['n_horses'], r['venue'], r['class'],
                                min_horses=filter_min_horses, max_dist=filter_max_dist,
                                exclude_venues=venue_ex_check, allowed_classes=allowed_cls,
                            )
                            all_rows.append({
                                'レース': r['race_label'], 'クラス': r['class'], '距離': f"{int(r['dist'])}m",
                                '出走頭数': r['n_horses'],
                                'LPI1位': r['ranked'][0]['horse'], 'LPI2位': r['ranked'][1]['horse'],
                                'gap': round(gap, 1),
                                'フィルター該当': '✅' if passes else '-',
                                '厳選対象': '🏆' if is_selected else '-',
                            })
                        st.dataframe(pd.DataFrame(all_rows), hide_index=True, use_container_width=True)

                        # ============================================================
                        # ログ記録セクション
                        # ------------------------------------------------------------
                        # Streamlit Cloud等はファイルシステムが再起動で消えるため、
                        # 「既存ログCSVをアップロード → 今回の結果を追記 → ダウンロード」
                        # という手元管理型にしている。ダウンロードしたファイルを
                        # 次回また①でアップロードすれば、履歴が積み上がっていく。
                        # ============================================================
                        st.markdown('---')
                        st.subheader('📝 ログに記録する')
                        st.caption(
                            '今回の厳選レースを記録します。着順・配当は後で結果が出てから'
                            'ダウンロードしたCSVを直接編集して埋めてください（この欄では空欄のまま出力します）。'
                        )
                        existing_log_file = st.file_uploader(
                            '既存のログCSV（あれば）。無ければ空のまま次に進んでOKです。',
                            type='csv', key='daily_log_upload'
                        )

                        if st.button('📝 今回の結果をログに追加してダウンロード', key='daily_log_append'):
                            log_rows = []
                            for s in selected:
                                ranked = s['ranked']
                                axis = ranked[0]
                                partners = ranked[1:1 + n_partners]
                                log_rows.append({
                                    '日付': s.get('date', ''),
                                    '会場': s.get('venue', ''),
                                    'R': s.get('race_no', ''),
                                    'クラス': s.get('class', ''),
                                    '距離': s.get('dist', ''),
                                    'トラック': s.get('track', ''),
                                    '出走頭数': s.get('n_horses', ''),
                                    'gap': round(s.get('gap', 0), 1),
                                    '軸': axis['horse'],
                                    '軸LPI': axis['avg_venue_lpi'],
                                    '相手1': partners[0]['horse'] if len(partners) > 0 else '',
                                    '相手2': partners[1]['horse'] if len(partners) > 1 else '',
                                    '相手3': partners[2]['horse'] if len(partners) > 2 else '',
                                    '相手4': partners[3]['horse'] if len(partners) > 3 else '',
                                    '実際1着': '',   # 後で手入力
                                    '実際2着': '',   # 後で手入力
                                    '馬単的中': '',   # 後で手入力(○/×)
                                    '馬単配当': '',   # 後で手入力
                                    '馬連的中': '',   # 後で手入力(○/×)
                                    '馬連配当': '',   # 後で手入力
                                    'メモ': '',
                                })
                            new_log_df = pd.DataFrame(log_rows)

                            if existing_log_file is not None:
                                try:
                                    old_log_df = pd.read_csv(existing_log_file, encoding='utf-8-sig')
                                except Exception:
                                    existing_log_file.seek(0)
                                    old_log_df = pd.read_csv(existing_log_file, encoding='cp932')
                                combined_log_df = pd.concat([old_log_df, new_log_df], ignore_index=True)
                                combined_log_df = combined_log_df.drop_duplicates(
                                    subset=['日付', '会場', 'R', '軸'], keep='last'
                                )
                                st.success(
                                    f'既存ログ{len(old_log_df)}件 + 今回{len(new_log_df)}件 '
                                    f'= 合計{len(combined_log_df)}件（重複日付･会場･R･軸は最新で上書き）'
                                )
                            else:
                                combined_log_df = new_log_df
                                st.success(f'新規ログを作成しました（{len(new_log_df)}件）')

                            st.dataframe(combined_log_df, hide_index=True, use_container_width=True)

                            log_csv_bytes = combined_log_df.to_csv(index=False).encode('utf-8-sig')
                            st.download_button(
                                '📥 ログCSVをダウンロード',
                                data=log_csv_bytes,
                                file_name=f'lpi_log_{selected[0].get("date", "unknown") if selected else "unknown"}.csv',
                                mime='text/csv',
                                key='daily_log_download',
                            )
                            st.caption(
                                '⚠️ このファイルを保存しておき、次回はここの「既存のログCSV」に'
                                'アップロードしてから同じ操作をすると、履歴が積み上がっていきます。'
                            )

# ============================================================
# フッター
# ============================================================
st.markdown('---')
st.caption('LPI v11 | 展開ボーナス対応版 | 2024-2025年バックテスト済み設定 | 平場基準推奨 | 会場適性ボーナスON・G1好走ボーナスOFF既定')
