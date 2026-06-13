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
GRADE_WEIGHT = {'G1':3.0,'G2':2.0,'G3':1.5,'L':1.0,'OP':1.0,'':0.8}
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
# 上がり予測スコア
# 過去走のZスコア（コース補正済み上がり）から予測
# 検証結果: 相関r=0.27, A評価のZ平均=+0.38（C評価=-0.24）
# 競馬場ごとの基準上がりは base_dict / 稍重_dict を使用
# ============================================================

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
                  predicted_pace_cat=None, pred_gap=None):
    """
    上がり予測 v2：ペース帯別Zスコアを別管理して精度向上。

    過去走を H/M/S のペース帯に分類し、
    コース予測ペースに対応するZ平均を返す。
    精度: 相関 r=0.76（ペース帯揃え時）

    Args:
        past_runs:          過去走データのリスト（新しい順, run_dataのdict）
        target_dist:        今回の距離
        target_venue:       今回の競馬場
        target_baba:        今回の馬場
        predicted_pace_cat: 予測ペース帯 'H'/'M'/'S'（PACE_TABLEから取得）

    Returns:
        dict: {
          'pace_cat':       使用したペース帯,
          'pred_z':         予測Zスコア,
          'z_by_pace':      {'H':float,'M':float,'S':float} ペース帯別Z平均,
          'n_by_pace':      {'H':int,'M':int,'S':int} ペース帯別走数,
          'grade':          'A'/'B'/'C',
          'grade_label':    '🔴切れ味A'等,
          'pred_agari':     予測上がり秒数,
          'confidence':     '◎安定/○やや安定/△不安定',
          'comment':        コメント文,
          'n_valid':        有効走総数,
        }
    """
    # ペース帯別グレード境界（2020〜2026年全重賞 16,691走から算出）
    PACE_GRADE_THRESH = {
        'H': {'A': -0.006, 'C': -0.988},
        'M': {'A':  0.399, 'C': -0.340},
        'S': {'A':  0.769, 'C':  0.179},
        None:{'A':  0.458, 'C': -0.341},
    }

    # 有効走をペース帯別に分類
    z_by_pace  = {'H': [], 'M': [], 'S': []}
    all_z_list = []

    for r in past_runs:
        if r.get('excluded_baba') or r.get('excluded_track'):
            continue
        z = r.get('z')
        if z is None or math.isnan(float(z)):
            continue
        z = float(z)
        rpci = r.get('rpci')
        if rpci is not None and not math.isnan(float(rpci)):
            rpci = float(rpci)
            if rpci <= 47:   pace = 'H'
            elif rpci >= 54: pace = 'S'
            else:            pace = 'M'
        else:
            pace = 'M'  # 不明はミドル扱い
        z_by_pace[pace].append(z)
        all_z_list.append(z)

    if not all_z_list:
        return None

    # ペース帯別の平均Z（直近重み付き：新しい走を優先）
    def weighted_avg(zlist):
        if not zlist: return None
        weights = [0.45, 0.30, 0.15, 0.07, 0.03]
        total_w = weighted_sum = 0.0
        for i, z in enumerate(zlist[:5]):
            w = weights[i] if i < len(weights) else 0.03
            weighted_sum += z * w; total_w += w
        return round(weighted_sum / total_w, 3) if total_w > 0 else zlist[0]

    z_avg = {pace: weighted_avg(zs) for pace, zs in z_by_pace.items()}
    n_by_pace = {pace: len(zs) for pace, zs in z_by_pace.items()}

    # 予測ペース帯に対応するZを選択
    # 優先順位: 予測ペース帯 → 隣接ペース帯 → 全体平均
    def pick_z(pred_pace):
        if pred_pace and z_avg.get(pred_pace) is not None:
            return pred_pace, z_avg[pred_pace]
        # フォールバック: 隣接ペース帯
        fallback_order = {
            'H': ['H','M','S'],
            'S': ['S','M','H'],
            'M': ['M','H','S'],
            None:['M','H','S'],
        }
        for p in fallback_order.get(pred_pace, ['M','H','S']):
            if z_avg.get(p) is not None:
                return p, z_avg[p]
        # 全体加重平均
        return None, weighted_avg(all_z_list)

    used_pace, pred_z = pick_z(predicted_pace_cat)
    if pred_z is None:
        pred_z = weighted_avg(all_z_list)
        used_pace = None

    # 安定度（全走のstd）
    z_std = float(np.std(all_z_list, ddof=1)) if len(all_z_list) >= 2 else 0.8
    if z_std <= 0.4:   confidence = '◎安定'
    elif z_std <= 0.7: confidence = '○やや安定'
    else:              confidence = '△不安定'

    # ペース帯に応じたグレード判定
    thresh = PACE_GRADE_THRESH.get(used_pace, PACE_GRADE_THRESH[None])
    if pred_z >= thresh['A']:
        grade, grade_label = 'A', '🔴 切れ味A'
    elif pred_z < thresh['C']:
        grade, grade_label = 'C', '⚪ 切れ味C'
    else:
        grade, grade_label = 'B', '🟡 切れ味B'

    # フォールバック説明
    pace_label = {'H':'ハイペース','M':'ミドル','S':'スロー',None:'全体平均'}
    used_label = pace_label.get(used_pace, '全体平均')
    fallback_note = ''
    if used_pace != predicted_pace_cat and predicted_pace_cat is not None:
        predicted_label = pace_label.get(predicted_pace_cat,'')
        fallback_note = f'（{predicted_label}走データなし→{used_label}で代替）'

    # コメント生成
    n_used = n_by_pace.get(used_pace, len(all_z_list))
    comment = (f'{used_label}時Z={pred_z:+.2f} {fallback_note}。'
               f'{"上位33%の切れ味" if grade=="A" else "下位33%の末脚" if grade=="C" else "標準的な末脚"}。')
    if grade == 'A' and used_pace == 'S' and predicted_pace_cat == 'S':
        comment += 'スロー展開での切れ味は特に信頼できる。'
    elif grade == 'C' and used_pace == 'H' and predicted_pace_cat == 'H':
        comment += 'ハイペース時に上がりが鈍る傾向。先行有利な展開でのみ買える。'

    # 予測上がり秒数
    course_base = COURSE_AGARI_BASE.get((float(target_dist), target_venue), 34.5)
    if str(target_baba).strip() == '稍': course_base += 0.4
    pred_agari = round(course_base - pred_z * 1.0, 1)

    return {
        'pace_cat':      used_pace,
        'pred_z':        round(pred_z, 3),
        'z_by_pace':     {p: round(v, 3) if v is not None else None for p,v in z_avg.items()},
        'n_by_pace':     n_by_pace,
        'grade':         grade,
        'grade_label':   grade_label,
        'pred_agari':    pred_agari,
        'gap_note':      gap_note,
        'course_base':   course_base,
        'confidence':    confidence,
        'z_std':         round(z_std, 3),
        'comment':       comment,
        'n_valid':       len(all_z_list),
        'past_zs':       [round(z,2) for z in all_z_list[:5]],
    }


POS_ZONE_LABELS = {
    1: ('逃げ',  '平均0.1以下',  '#1A237E', '🟦'),
    2: ('先行',  '0.2〜0.4秒',   '#1B5E20', '🟩'),
    3: ('中団',  '0.5〜1.0秒',   '#E65100', '🟨'),
    4: ('後方',  '1.1秒〜',      '#B71C1C', '🟥'),
}

def gap_to_zone(gap):
    """
    地点差(秒) → ポジション帯番号
    1=逃げ（平均0.1以下）
    2=先行（0.2〜0.4）
    3=中団（0.5〜1.0）
    4=後方（1.1〜）
    """
    if gap is None or (isinstance(gap, float) and math.isnan(gap)):
        return None
    if gap <= 0.1: return 1   # 逃げ：先頭からほぼ0
    if gap <= 0.4: return 2   # 先行：0.2〜0.4秒
    if gap <= 1.0: return 3   # 中団：0.5〜1.0秒
    return 4                   # 後方：1.1秒〜

def predict_position(past_gaps, rpci_pred=None):
    """
    過去走の地点差リスト → ポジション予測。

    Args:
        past_gaps: 過去走の地点差リスト（新しい順）。Noneは除外。
        rpci_pred: 予測RPCI。スロー/ハイによる補正に使用。

    Returns:
        dict: {
          'pred_gap':     予測地点差（秒）,
          'pred_zone':    予測ゾーン番号（1〜4）,
          'label':        'ゾーン名（範囲）',
          'confidence':   '◎安定/○やや安定/△不安定',
          'icon':         絵文字,
          'color':        カラーコード,
          'gap_std':      過去走の地点差std,
          'n_valid':      有効走数,
        }
    """
    valid_gaps = [g for g in past_gaps if g is not None and not math.isnan(float(g))]
    if not valid_gaps:
        return None

    # 直近重み付き平均（1走前50%・2走前30%・3走前以前20%）
    weights = [0.50, 0.30, 0.15, 0.05]
    total_w, weighted_sum = 0.0, 0.0
    for i, g in enumerate(valid_gaps[:4]):
        w = weights[i] if i < len(weights) else 0.05
        weighted_sum += g * w
        total_w += w
    pred_gap = round(weighted_sum / total_w, 2) if total_w > 0 else valid_gaps[0]

    # RPCIによる補正（スロー→縦長化抑制, ハイ→縦長化）
    # スローでは馬群がコンパクトになる傾向（地点差std↓）
    if rpci_pred is not None:
        if rpci_pred >= 54:    pred_gap -= 0.05   # 超スロー: やや前詰まり
        elif rpci_pred <= 46:  pred_gap += 0.05   # 超ハイ: やや後ろ広がり

    pred_gap = max(0.0, pred_gap)
    pred_zone = gap_to_zone(pred_gap)

    # 安定度（std）
    if len(valid_gaps) >= 2:
        gap_std = float(np.std(valid_gaps, ddof=1))
    else:
        gap_std = 0.5  # デフォルト

    # 安定度ラベル
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

WALK_DEFS = [
    {'n':1,'agari':'上り3F',  'rpci':'RPCI',  'venue':'場所',  'dist':'距離',
     'baba':'馬場状態',  'rank':'着順',  'race':'ﾚｰｽ名･1走前','td':'TD','gap':'-3F差'},
    {'n':2,'agari':'上り3F.1','rpci':'RPCI.1','venue':'場所.1','dist':'距離.1',
     'baba':'馬場状態.1','rank':'着順.1','race':'ﾚｰｽ名･2走前','td':'TD.1','gap':'-3F差.1'},
    {'n':3,'agari':'上り3F.2','rpci':'RPCI.2','venue':'場所.2','dist':'距離.2',
     'baba':'馬場状態.2','rank':'着順.2','race':'ﾚｰｽ名･3走前','td':'TD.2','gap':'-3F差.2'},
    {'n':4,'agari':'上り3F.3','rpci':'RPCI.3','venue':'場所.3','dist':'距離.3',
     'baba':'馬場状態.3','rank':'着順.3','race':'ﾚｰｽ名･4走前','td':'TD.3','gap':'-3F差.3'},
    {'n':5,'agari':'上り3F.4','rpci':'RPCI.4','venue':'場所.4','dist':'距離.4',
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
# ============================================================
@st.cache_data
def build_base_table(file_bytes):
    # エンコードを自動判定（shift_jis → cp932 → utf-8 の順で試行）
    for enc in ['cp932', 'shift_jis', 'utf-8-sig', 'utf-8']:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            break
        except (UnicodeDecodeError, Exception):
            continue
    else:
        raise ValueError('CSVの文字コードを判定できませんでした')
    df['距離_num'] = df['距離'].str.extract(r'(\d+)').astype(float)
    df['上がり']   = pd.to_numeric(df['上り3F'], errors='coerce')
    df['競馬場']   = df['開催'].apply(get_venue_from_kaisan)
    df['馬場']     = df['馬場状態'].str.strip()
    valid = df[df['馬場'].isin(['良', '稍'])].copy()
    stats = valid.groupby(['距離_num','競馬場','馬場'])['上がり'].agg(
        avg='mean', std='std', n='count').reset_index()
    stats = stats[stats['n'] >= 5]
    良_s  = stats[stats['馬場'] == '良']
    稍_s  = stats[stats['馬場'] == '稍']
    base_dict  = {(r['距離_num'], r['競馬場']): (r['avg'], r['std']) for _, r in 良_s.iterrows()}
    稍重_dict  = {(r['距離_num'], r['競馬場']): (r['avg'], r['std']) for _, r in 稍_s.iterrows()}
    return base_dict, 稍重_dict

def get_z(agari, dist, venue, baba, base_dict, 稍重_dict):
    fb  = {1000:32.5,1200:34.0,1400:34.2,1600:34.4,
           1800:34.6,2000:35.0,2200:35.2,2400:35.3,2500:35.5}
    key = (float(dist), venue)
    b   = str(baba).strip()
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
# ============================================================
def calc_lpi(entry_bytes, base_dict, 稍重_dict,
             target_track='T', target_venue='東京', bonus_strength=0.15):
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
            except:
                continue
            if math.isnan(agari) or math.isnan(rpci): continue

            gap_est  = gap if (gap is not None and not math.isnan(gap)) else 1.5
            grade    = extract_grade(race)
            wt_corr  = weight_correction_sec(grade, age, sex)
            agari_adj = agari + wt_corr
            z        = get_z(agari_adj, dist, venue, baba, base_dict, 稍重_dict)
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
            lpi  = sigmoid_score((z + pb + hb) * pm * g1_pen)
            gw   = GRADE_WEIGHT.get(grade, GRADE_WEIGHT[''])

            run_data.append({
                'n': wd['n'], 'race': race, 'dist': dist, 'venue': venue,
                'rpci': rpci, 'gap_est': round(gap_est, 2),
                'agari': agari, 'agari_adj': round(agari_adj, 2),
                'wt_corr': round(wt_corr, 2),
                'z': round(z, 3), 'rank': rank, 'rank_int': rank_int,
                'baba': baba, 'track': track, 'grade': grade,
                'pb': pb, 'pm': round(pm, 3), 'hb': hb, 'hb_r': hb_r,
                'elem': elem, 'lpi': lpi, 'grade_weight': gw,
                'excluded_baba':  baba not in GOOD_BABA,
                'excluded_track': track != target_track,
            })

        if not run_data: continue
        valid = [r for r in run_data if not r['excluded_baba'] and not r['excluded_track']]
        use   = valid if valid else run_data
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
        pos_pred = predict_position(past_gaps_for_pred, pace['pred_rpci'])

        # 上がり予測（過去走のZスコアから）
        # コースのペース予測からペース帯を決定
        _pred_rpci = pace['pred_rpci']
        _pace_cat  = 'H' if _pred_rpci <= 47 else ('S' if _pred_rpci >= 54 else 'M')

        # 予測ポジション（地点差）を上がり予測に渡す
        _pred_gap = pos_pred['pred_gap'] if pos_pred else None
        agari_pred = predict_agari(
            past_runs           = use[:5],
            target_dist         = float(str(run_data[0]['dist']).replace('m','')),
            target_venue        = target_venue,
            target_baba         = '良',
            predicted_pace_cat  = _pace_cat,
            pred_gap            = _pred_gap,
        )

        results.append({
            'horse': horse, 'sex': sex, 'age': age,
            'avg_lpi': avg_lpi, 'avg_venue_lpi': avg_venue,
            'max_lpi': round(max(r['lpi'] for r in run_data), 1),
            'latest_lpi': run_data[0]['lpi'],
            'n_valid': len(valid), 'n_total': len(run_data), 'n_good': len(good),
            'dom_elem': dom_elem, 'coef': round(coef, 2),
            'venue_delta': round(avg_venue - avg_lpi, 1),
            'pos_pred':   pos_pred,
            'agari_pred': agari_pred,
            'runs': run_data, 'valid_runs': valid, 'good_runs': good,
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
    # 日本語フォントを環境に応じて自動選択
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
    # 日本語フォントが見つからない場合は順位番号のみ表示
    if chosen:
        ax.set_yticklabels([f'{i+1}. {n}' for i,n in enumerate(names)], fontsize=9)
    else:
        ax.set_yticklabels([f'{i+1}.' for i in range(len(names))], fontsize=9)
        # 代わりにグラフ右に馬名テキストを追加
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
st.title('🏇 LPI v11 競馬予想ツール')
st.caption('LPI (ラップ強さ指数) v11 — 位置取り補正・グレード加重・斤量補正・競馬場適合ボーナス対応')

# ---- サイドバー ----
with st.sidebar:
    st.header('⚙️ 設定')

    st.subheader('① 基準テーブル用CSV')
    base_file = st.file_uploader(
        '2023〜2026年全距離重賞CSVをアップ',
        type='csv', key='base')
    if base_file:
        st.success(f'{base_file.name} 読み込み済み')

    st.subheader('② 出走表CSV')
    entry_file = st.file_uploader(
        '予想するレースの出走表CSVをアップ',
        type='csv', key='entry')

    st.subheader('③ レース設定')
    race_name    = st.text_input('レース名', value='2026 安田記念G1')
    target_venue = st.selectbox(
        '競馬場',
        ['東京','中山','京都','阪神','中京','新潟','福島','小倉','札幌','函館'])
    target_track = st.radio('トラック', ['T（芝）','D（ダート）'])
    track_code   = 'T' if target_track.startswith('T') else 'D'
    bonus_strength = st.slider(
        '競馬場ボーナス強度', 0.0, 0.30, 0.15, 0.05,
        help='0=ボーナスなし / 0.15=標準 / 0.30=強め')

    st.subheader('④ ペース予測（任意）')
    race_dist = st.number_input('レース距離（m）', min_value=1000, max_value=3600, value=1600, step=200)
    nige_count   = st.number_input('逃げ馬頭数', min_value=0, max_value=10, value=0, step=1,
                                   help='出走表の決め手=逃げの馬の頭数')
    senkou_count = st.number_input('先行馬頭数', min_value=0, max_value=16, value=0, step=1,
                                   help='出走表の決め手=先行の馬の頭数')

    st.markdown('**ペース直接指定（任意）**')
    use_manual_rpci = st.checkbox(
        'RPCIを直接入力する',
        value=False,
        help='逃げ馬1頭でもハイになりそうな場合など、自動推定より自分の読みを優先したいときに使用'
    )
    if use_manual_rpci:
        manual_rpci = st.number_input(
            'RPCI値を入力',
            min_value=30.0, max_value=75.0, value=50.0, step=0.5,
            help='40台=ハイ / 50前後=ミドル / 55以上=スロー'
        )
        pace_zone_label = (
            '🔵 ハイペース（RPCI≤47）' if manual_rpci <= 47 else
            '🟠 スローペース（RPCI≥54）' if manual_rpci >= 54 else
            '🟢 ミドルペース（RPCI 48〜53）'
        )
        st.caption(f'→ {pace_zone_label}')
    else:
        manual_rpci = 0.0

    run_btn = st.button('🔍 LPI計算実行', type='primary', use_container_width=True)

# ---- メインエリア ----
if not base_file:
    st.info('← サイドバーから基準テーブル用CSVをアップしてください（2023〜2026年全距離重賞データ）')
    st.stop()

if not entry_file:
    st.info('← サイドバーから出走表CSVをアップしてください')
    st.stop()

if run_btn or (base_file and entry_file):
    # ペース予測（LPI計算内の pos_pred からも参照するため先に実行）
    pace = get_pace_prediction(race_dist, target_venue, nige_count, senkou_count)

    with st.spinner('基準テーブルを構築中...'):
        base_dict, 稍重_dict = build_base_table(base_file.read())
        base_file.seek(0)  # 再読み込みのためリセット

    with st.spinner('LPI計算中...'):
        results = calc_lpi(
            entry_file.read(),
            base_dict, 稍重_dict,
            target_track=track_code,
            target_venue=target_venue,
            bonus_strength=bonus_strength,
        )

    if not results:
        st.error('計算できるデータがありませんでした。CSVの形式を確認してください。')
        st.stop()

    st.success(f'✅ {len(results)}頭 計算完了')

    # ---- ペース予測バナー ----
    lamp_color  = {'🟠': '#E65100', '🔵': '#0D47A1', '⚪': '#424242'}
    border_color = {'🟠': '#FF6D00', '🔵': '#1565C0', '⚪': '#616161'}
    bc  = lamp_color.get(pace['lamp'], '#424242')
    brd = border_color.get(pace['lamp'], '#616161')

    adv_html = ''
    if pace['elem_adv']:
        adv_html = '  有利な要素型: **' + ' / '.join(pace['elem_adv']) + '**'

    st.markdown(
        f"""<div style="background:{bc};border:2px solid {brd};border-radius:8px;
        padding:12px 16px;margin:8px 0;font-size:13px;line-height:1.8;color:#FFFFFF;font-weight:500">
        <b>{pace['lamp']} ペース予測: {pace['label']}</b>　
        予測RPCI <b>{pace['pred_rpci']:.1f}</b>（基準{pace['base_rpci']:.1f}±{pace['std']:.1f}）<br>
        スロー率 <b>{pace['slow_pct']}%</b>　ハイ率 <b>{pace['fast_pct']}%</b><br>
        {pace['comment']}{adv_html}
        </div>""",
        unsafe_allow_html=True
    )


    # ---- タブで表示 ----
    tab1, tab2, tab3 = st.tabs(['📊 ランキング表', '📈 グラフ', '🔍 過去走詳細'])

    # ===== タブ1: ランキング表 =====
    with tab1:
        st.subheader(f'{race_name}  LPI v11 ランキング')

        # 表データ作成
        rows = []
        for i, r in enumerate(results):
            bonus_runs = [rn for rn in r['runs'] if rn['hb'] > 0]
            bonus_str  = ' / '.join(
                [f"{rn['race']}({rn['hb_r']})" for rn in bonus_runs])
            past  = [rn for rn in r['runs']
                     if not rn['excluded_baba'] and not rn['excluded_track']][:5]
            plpi  = [round(rn['lpi'], 1) for rn in past]
            while len(plpi) < 5: plpi.append('-')

            delta = r['venue_delta']
            # ペース適合判定
            pace_match = r['dom_elem'] in pace['elem_adv'] if pace['elem_adv'] else None
            pace_mark = '◎' if pace_match else ('△' if pace_match is False else '-')

            rows.append({
                '順位':           i + 1,
                '馬名':           r['horse'],
                f'LPI[{target_venue}補正]': r['avg_venue_lpi'],
                'LPI基本':        r['avg_lpi'],
                '補正幅':         f"+{delta}" if delta >= 0 else str(delta),
                '要素型':         r['dom_elem'],
                '係数':           r['coef'],
                'LPI最高':        r['max_lpi'],
                'LPI直近':        r['latest_lpi'],
                '有効/全走':      f"{r['n_valid']}/{r['n_total']}",
                '好走走':         r['n_good'],
                '1走前':          plpi[0],
                '2走前':          plpi[1],
                '3走前':          plpi[2],
                '4走前':          plpi[3],
                '5走前':          plpi[4],
                'ペース適合':      pace_mark,
                '予測ポジション':  (r['pos_pred']['icon'] + r['pos_pred']['zone_name']
                                    + r['pos_pred']['confidence'])
                                   if r.get('pos_pred') else '-',
                '予測地点差':      round(r['pos_pred']['pred_gap'],1) if r.get('pos_pred') else '-',
                '平均-3F差':       (f"{sum(r['pos_pred']['past_gaps'])/len(r['pos_pred']['past_gaps']):.1f}秒"
                                    f"(n={r['pos_pred']['n_valid']})")
                                   if r.get('pos_pred') and r['pos_pred']['past_gaps'] else '-',
                '上がり予測':      (r['agari_pred']['grade_label']
                                    + ' ' + r['agari_pred']['confidence'])
                                   if r.get('agari_pred') else '-',
                '予測上がり秒':    r['agari_pred']['pred_agari']
                                   if r.get('agari_pred') else '-',
                '不利ボーナス':   bonus_str,
            })

        result_df = pd.DataFrame(rows)
        lpi_col   = f'LPI[{target_venue}補正]'

        # カラーハイライト
        def highlight(row):
            if row['順位'] == 1:
                return ['background-color: #F9A825; color: #000000; font-weight: bold'] * len(row)
            if row['順位'] == 2:
                return ['background-color: #1565C0; color: #FFFFFF; font-weight: bold'] * len(row)
            if row['順位'] == 3:
                return ['background-color: #BF360C; color: #FFFFFF; font-weight: bold'] * len(row)
            if row['順位'] <= 5:
                return ['background-color: #1B1B2F; color: #E0E0E0'] * len(row)
            return [''] * len(row)

        st.dataframe(
            result_df.style
                .apply(highlight, axis=1)
                .format({lpi_col: '{:.1f}', 'LPI基本': '{:.1f}',
                         'LPI最高': '{:.1f}', 'LPI直近': '{:.1f}', '係数': '{:.2f}'})
                .set_properties(**{'border': '1px solid #444', 'font-size': '13px'})
                .hide(axis='index'),
            use_container_width=True,
            height=min(600, 45 + len(rows) * 38),
        )

        # Excel ダウンロード
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

        # 要素型の説明
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
            # ポジション予測サマリー
            if hr.get('pos_pred'):
                pp = hr['pos_pred']
                st.markdown(
                    f"**予測ポジション:** {pp['icon']} {pp['label']}　"
                    f"予測地点差 **{pp['pred_gap']:.2f}秒**　"
                    f"安定度: **{pp['confidence']}**（過去{pp['n_valid']}走 std={pp['gap_std']:.3f}）",
                )
            if hr.get('agari_pred'):
                ap = hr['agari_pred']
                # ペース帯別Z表示
                pace_z_parts = []
                for p, lbl in [('H','🔵ハイ'),('M','🟢ミドル'),('S','🟠スロー')]:
                    z_val = ap['z_by_pace'].get(p)
                    n_val = ap['n_by_pace'].get(p, 0)
                    if z_val is not None:
                        pace_z_parts.append(f'{lbl}: **{z_val:+.2f}** (n={n_val})')
                    else:
                        pace_z_parts.append(f'{lbl}: データなし')
                pace_z_str = '　'.join(pace_z_parts)

                used_pace_lbl = {'H':'ハイペース','M':'ミドル','S':'スロー',None:'全体'}
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
                if rn['excluded_baba']:  excl_reason.append('重/不良')
                if rn['excluded_track']: excl_reason.append('トラック違い')
                run_rows.append({
                    '走前':     rn['n'],
                    'レース名': rn['race'],
                    '競馬場':   rn['venue'],
                    '距離':     int(rn['dist']),
                    '馬場':     rn['baba'],
                    'RPCI':     rn['rpci'],
                    '地点差':   rn['gap_est'],
                    '上がり':   rn['agari'],
                    '斤量補正': rn['wt_corr'],
                    'Zスコア':  rn['z'],
                    'pb(位置補正)': rn['pb'],
                    'hb(不利B)':   rn['hb'],
                    'LPI':      rn['lpi'],
                    '要素型':   rn['elem'],
                    '除外':     '⚠️ ' + '/'.join(excl_reason) if excl_reason else '✅',
                    '不利理由': rn['hb_r'],
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

# ============================================================
# フッター
# ============================================================
st.markdown('---')
st.caption('LPI v11 | 基準: 2023〜2026年全距離重賞 | 良・稍重有効 | グレード加重平均 | 斤量補正あり')
