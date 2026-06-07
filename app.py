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
    df = pd.read_csv(io.BytesIO(file_bytes), encoding='cp932')
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
    df = pd.read_csv(io.BytesIO(entry_bytes), encoding='shift_jis')

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

        results.append({
            'horse': horse, 'sex': sex, 'age': age,
            'avg_lpi': avg_lpi, 'avg_venue_lpi': avg_venue,
            'max_lpi': round(max(r['lpi'] for r in run_data), 1),
            'latest_lpi': run_data[0]['lpi'],
            'n_valid': len(valid), 'n_total': len(run_data), 'n_good': len(good),
            'dom_elem': dom_elem, 'coef': round(coef, 2),
            'venue_delta': round(avg_venue - avg_lpi, 1),
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
    matplotlib.rcParams['font.family'] = ['Hiragino Sans', 'Meiryo', 'sans-serif']

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
    ax.set_yticklabels([f'{i+1}. {n}' for i,n in enumerate(names)], fontsize=9)
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

    run_btn = st.button('🔍 LPI計算実行', type='primary', use_container_width=True)

# ---- メインエリア ----
if not base_file:
    st.info('← サイドバーから基準テーブル用CSVをアップしてください（2023〜2026年全距離重賞データ）')
    st.stop()

if not entry_file:
    st.info('← サイドバーから出走表CSVをアップしてください')
    st.stop()

if run_btn or (base_file and entry_file):
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
                '不利ボーナス':   bonus_str,
            })

        result_df = pd.DataFrame(rows)
        lpi_col   = f'LPI[{target_venue}補正]'

        # カラーハイライト
        def highlight(row):
            if row['順位'] == 1:   return ['background-color: #fff9c4'] * len(row)
            if row['順位'] == 2:   return ['background-color: #f0f4ff'] * len(row)
            if row['順位'] == 3:   return ['background-color: #fff0e6'] * len(row)
            return [''] * len(row)

        st.dataframe(
            result_df.style
                .apply(highlight, axis=1)
                .format({lpi_col: '{:.1f}', 'LPI基本': '{:.1f}',
                         'LPI最高': '{:.1f}', 'LPI直近': '{:.1f}', '係数': '{:.2f}'})
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
