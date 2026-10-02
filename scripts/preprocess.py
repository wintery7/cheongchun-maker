# -*- coding: utf-8 -*-
"""
「청춘을 만들어가자」 데이터 전처리 스크립트
====================================================================
원천 데이터 (모두 서울올림픽기념국민체육진흥공단 제공)
  [D1] 체력측정 및 운동처방 종합 데이터
       파일: KS_NFA_FTNESS_MESURE_MVN_PRSCRPTN_GNRLZ_INFO_202605/06/07.json
       기간: 측정일(MESURE_DE) 2026-05-01 ~ 2026-07-31
  [D2] 전국체육시설현황 데이터
       파일: KS_WNTY_PHSTRN_FCLTY_STTUS_202607.csv
  [D3] 체력측정 항목별 측정 데이터
       파일: KS_NFA_FTNESS_MESURE_ITEM_MESURE_INFO_YYYYMM.csv (2022.01~2026.07, 55개)
       운동처방·회원번호는 없고 측정값·인증등급(COAW_FLAG_NM)이 있다.
       → 동년배 순위·체력나이·예상 등급의 표본을 늘리는 데 쓴다.

출력 (../data/*.json, 웹앱이 그대로 사용)
  norms.json      성별 x 연령대별 측정항목 백분위표 (동년배 순위 계산)
  age_curve.json  성별 x 만나이별 측정항목 중앙값 곡선 (체력나이 계산)
  knn.json        비슷한 체력의 어르신 찾기용 비식별 축약 레코드 + 운동처방 ID
  exercises.json  운동처방 운동명 사전 + 장소 분류(집/소도구/체육시설/수영장)
  centers.json    국민체력100 측정 센터별 어르신 측정 건수 + 시도/시군구 매핑
  facilities.json 공공 체육시설(체육센터·수영장·동네 운동시설) 위치
  meta.json       처리 건수, 컬럼 매핑 검증 결과, 예측 정확도(홀드아웃) 등

실행:  python3 scripts/preprocess.py --raw <원천파일 폴더> --out data
필요:  Python 3.10+, pandas, numpy, scikit-learn
====================================================================
"""
import argparse, json, os, re, collections, datetime
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors

# --------------------------------------------------------------------
# 0. [D1] 측정항목 컬럼 매핑
#    문화빅데이터플랫폼 '컬럼 정의서' 기준. 원천은 측정항목을
#    MESURE_IEM_###_VALUE 코드로 제공한다.
#    ※ 정의서는 026(3m 표적 돌아오기) 단위를 '회'로 적고 있으나 실제 값의
#      95.7%가 소수점(중앙값 5.5)이라 초 단위로 처리한다.
#    verify_mapping()은 계산 컬럼(BMI, 악력)과 검사 간 상관이 정상인지
#    확인하는 내부 품질 점검이다(결과는 meta.json).
# --------------------------------------------------------------------
COL = {
    "height":  "MESURE_IEM_001_VALUE",  # 신장(cm)
    "weight":  "MESURE_IEM_002_VALUE",  # 체중(kg)
    "grip_l":  "MESURE_IEM_007_VALUE",  # 악력 좌(kg)
    "grip_r":  "MESURE_IEM_008_VALUE",  # 악력 우(kg)
    "reach":   "MESURE_IEM_012_VALUE",  # 앉아윗몸앞으로굽히기(cm)
    "bmi":     "MESURE_IEM_018_VALUE",  # BMI(kg/m2)  = 체중/신장^2 와 일치 검증
    "chair":   "MESURE_IEM_023_VALUE",  # 의자에 앉았다 일어서기(회/30초)
    "step":    "MESURE_IEM_025_VALUE",  # 2분 제자리 걷기(회)
    "tug":     "MESURE_IEM_026_VALUE",  # 의자에 앉아 3m 표적 돌아오기(초)
    "fig8":    "MESURE_IEM_027_VALUE",  # 8자 보행(초)
    "relgrip": "MESURE_IEM_028_VALUE",  # 상대악력(%) = 절대악력/체중*100 과 일치 검증
    "grip":    "MESURE_IEM_052_VALUE",  # 절대악력(kg) = max(좌,우) 와 일치 검증
}

# 앱이 사용하는 '집에서 잴 수 있는' 핵심 3항목과 방향(높을수록 좋음 여부)
TESTS = {"chair": True, "step": True, "tug": False}

# 이상치/미측정 처리 규칙 (값이 범위를 벗어나면 결측으로 처리)
#  - 0회는 '측정 불가 또는 미측정'으로 보고 제외
#  - 3m 표적 돌아오기 100초 등은 입력 오류로 보고 30초 초과 제외
VALID = {"chair": (1, 60), "step": (1, 250), "tug": (2.0, 30.0), "bmi": (10, 50)}

AGE_BANDS = [(65, 69, "65-69"), (70, 74, "70-74"), (75, 79, "75-79"),
             (80, 84, "80-84"), (85, 200, "85+")]
CURVE_AGES = list(range(65, 91))   # 체력나이 곡선: 65~90세 (90 = 90세 이상)
KNN_K = 100                        # '비슷한 어르신' 이웃 수
SEED = 42


def band_of(age):
    for lo, hi, name in AGE_BANDS:
        if lo <= age <= hi:
            return name
    return None


# --------------------------------------------------------------------
# 1. [D1] 적재 · 어르신 추출 · 정제
# --------------------------------------------------------------------
def load_fitness(raw_dir):
    files = sorted(f for f in os.listdir(raw_dir)
                   if f.startswith("KS_NFA_FTNESS_MESURE_MVN_PRSCRPTN_GNRLZ_INFO") and f.endswith(".json"))
    rows = []
    for f in files:
        with open(os.path.join(raw_dir, f), encoding="utf-8") as fp:
            rows += json.load(fp)
    df = pd.DataFrame(rows)
    n_all = len(df)
    # 같은 회원·같은 측정회차·같은 측정일 중복 제거 (월별 파일 간 중복 대비)
    df = df.drop_duplicates(["MBER_SEQ_NO_VALUE", "MESURE_SEQ_NO", "MESURE_DE"])
    info = {"files": files, "records_all_ages": int(n_all),
            "records_after_dedup": int(len(df)),
            "measure_date_min": df.MESURE_DE.min(), "measure_date_max": df.MESURE_DE.max(),
            "agegroup_counts": df.AGRDE_FLAG_NM.value_counts().to_dict()}
    return df, info


def load_items(raw_dir, covered_months):
    """[D3] 항목별 측정 데이터 적재.
    - 바이트 단위로 같은 파일은 한 번만 쓴다(원천 202210 파일이 202309와 동일).
    - [D1]이 있는 달(2026.05~07)은 [D1]을 쓴다. 두 데이터의 해당 기간 어르신
      기록 99.9%가 일치해, 함께 쓰면 같은 측정을 두 번 세게 되기 때문.
    - 컬럼명을 [D1]과 맞춘다: AGE_FLAG_NM→AGRDE_FLAG_NM, MESURE_DAY→MESURE_DE,
      COAW_FLAG_NM→CRTFC_FLAG_NM. 운동처방·회원번호는 없음."""
    import hashlib
    files = sorted(f for f in os.listdir(raw_dir)
                   if f.startswith("KS_NFA_FTNESS_MESURE_ITEM_MESURE_INFO") and f.endswith(".csv"))
    seen, used, dup, skipped, frames = {}, [], [], [], []
    for f in files:
        path = os.path.join(raw_dir, f)
        h = hashlib.md5(open(path, "rb").read()).hexdigest()
        if h in seen:
            dup.append({"file": f, "same_as": seen[h]})
            continue
        seen[h] = f
        d = pd.read_csv(path, dtype=str, encoding="utf-8-sig", engine="python", on_bad_lines="skip")
        d = d.rename(columns={"AGE_FLAG_NM": "AGRDE_FLAG_NM", "MESURE_DAY": "MESURE_DE", "COAW_FLAG_NM": "CRTFC_FLAG_NM"})
        d["_month"] = d.MESURE_DE.str[:6]
        drop = d._month.isin(covered_months)
        if drop.any():
            skipped.append({"file": f, "rows_replaced_by_D1": int(drop.sum())})
        d = d[~drop]
        d["MVM_PRSCRPTN_CN"] = ""
        frames.append(d)
    it = pd.concat(frames, ignore_index=True)
    info = {"files": files, "duplicate_files": dup, "replaced_by_D1": skipped,
            "records_used_all_ages": int(len(it)),
            "senior_rows_by_month": it[it.AGRDE_FLAG_NM == "어르신"]._month.value_counts().sort_index().to_dict()}
    return it, info


def clean_seniors(df):
    # 연령대구분(AGRDE_FLAG_NM)='어르신' 이고 만 65세 이상인 기록만 사용
    s = df[df.AGRDE_FLAG_NM == "어르신"].copy()
    out = pd.DataFrame({
        "sex": s.SEXDSTN_FLAG_CD,                                   # M/F
        "age": pd.to_numeric(s.MESURE_AGE_CO, errors="coerce"),      # 측정 당시 만나이
        "grade": pd.to_numeric(s.CRTFC_FLAG_NM.str.extract(r"(\d)")[0], errors="coerce"),  # 인증등급 'n등급'→n
        "rx": s.MVM_PRSCRPTN_CN.fillna(""),                          # 운동처방 내용(문자열)
        "center": s.CNTER_NM,                                       # 측정 센터명
        "de": s.MESURE_DE.astype(str),                               # 측정일(YYYYMMDD)
        "place": s.MESURE_PLACE_FLAG_NM,                              # 측정장소(센터/출장)
    })
    for k, c in COL.items():
        out[k] = pd.to_numeric(s[c], errors="coerce")
    out = out[out.sex.isin(["M", "F"]) & out.age.between(65, 100)]
    for k, (lo, hi) in VALID.items():
        out.loc[~out[k].between(lo, hi), k] = np.nan
    out["band"] = out.age.map(band_of)
    return out.reset_index(drop=True)


def verify_mapping(s):
    """컬럼 매핑 검증: 공식으로 계산한 값과 원천 값의 일치율"""
    v = {}
    calc_bmi = s.weight / (s.height / 100) ** 2
    v["bmi_formula_match"] = float((abs(calc_bmi - s.bmi) < 0.15).mean())
    v["grip_is_max_lr"] = float((abs(np.fmax(s.grip_l, s.grip_r) - s.grip) < 0.05).mean())
    v["relgrip_formula_match"] = float((abs(s.grip / s.weight * 100 - s.relgrip) < 0.15).mean())
    # 검사 의미 검증: 하체근력(의자)·이동(3m)·8자보행 간 상관 방향
    v["spearman_chair_tug"] = float(s[["chair", "tug"]].corr("spearman").iloc[0, 1])
    v["spearman_tug_fig8"] = float(s[["tug", "fig8"]].corr("spearman").iloc[0, 1])
    v["spearman_age_tug"] = float(s[["age", "tug"]].corr("spearman").iloc[0, 1])
    return v


# --------------------------------------------------------------------
# 2. 동년배 백분위표 (기능: '동년배 100명 중 몇 번째')
#    성별 x 5세 연령대마다 각 항목의 0~100 분위수(101개)를 저장.
#    앱은 입력값이 이 분위수 배열의 어디에 놓이는지 보간해 백분위를 구한다.
#    항목별로 값이 있는 기록을 모두 쓴다(항목별 표본 수가 다를 수 있음).
# --------------------------------------------------------------------
def build_norms(s):
    norms = {}
    for sex in ["F", "M"]:
        norms[sex] = {}
        for _, _, b in AGE_BANDS:
            g = s[(s.sex == sex) & (s.band == b)]
            norms[sex][b] = {}
            for t in TESTS:
                v = g[t].dropna().values
                norms[sex][b][t] = {
                    "n": int(len(v)),
                    "q": [round(float(x), 2) for x in np.percentile(v, range(101))] if len(v) else [],
                }
    return norms


# --------------------------------------------------------------------
# 3. 체력나이 곡선 (기능: '체력나이')
#    성별 x 만나이(65~90)별 중앙값을 ±2세 이동창으로 구하고,
#    나이가 들수록 의자·제자리걷기는 줄고 3m 시간은 늘어나도록
#    등위회귀(isotonic)로 단조성을 보정한다.
#    앱은 입력값이 곡선상 어느 나이의 중앙값과 같은지 역보간해
#    항목별 체력나이를 구하고, 항목 평균을 종합 체력나이로 쓴다.
# --------------------------------------------------------------------
def build_age_curve(s):
    curve = {}
    for sex in ["F", "M"]:
        g = s[s.sex == sex].copy()
        g["age_c"] = g.age.clip(upper=90)
        curve[sex] = {"ages": CURVE_AGES}
        for t, higher_better in TESTS.items():
            med, wts = [], []
            for a in CURVE_AGES:
                w = 2
                while True:
                    v = g[(g.age_c >= a - w) & (g.age_c <= a + w)][t].dropna()
                    if len(v) >= 40 or w >= 6:
                        break
                    w += 1
                med.append(float(v.median()))
                wts.append(len(v))
            iso = IsotonicRegression(increasing=not higher_better)
            fit = iso.fit_transform(CURVE_AGES, med, sample_weight=wts)
            curve[sex][t] = [round(float(x), 2) for x in fit]
            curve[sex][t + "_n"] = wts
    return curve


# --------------------------------------------------------------------
# 4. 운동처방 파싱 (기능: '비슷한 어르신들이 실제 받은 운동')
#    MVM_PRSCRPTN_CN 형식: "준비운동:a,b / 본운동:c,d / 정리운동:e,f"
#    → 구간별 운동명 목록으로 분해, 공백 정리 후 운동명 사전(ID) 생성.
#    장소 분류는 운동명 키워드 규칙(아래)으로 개발자가 부여했다.
#      3 수영장  : '아쿠아','수영','물속'
#      2 체육시설: 고정식 기구·트레드밀·바벨·머신류 등 시설 기구가 필요한 운동
#      1 소도구  : 덤벨·탄력밴드·짐볼·공·폼롤러 등 작은 도구가 필요한 운동
#      0 집      : 그 외(스트레칭, 의자·물병·베개 이용, 걷기 등)
# --------------------------------------------------------------------
SECTIONS = {"준비운동": "prep", "본운동": "main", "정리운동": "cool"}
KW_POOL = ["아쿠아", "수영", "물속"]
KW_GYM = ["고정식", "트레드밀", "바벨", "머신", "스텝박스", "사다리", "앉아서 다리 펴기", "앉아서 다리 밀기",
          "엎드려서 다리 굽히기", "당겨 내리기", "앉아서 뒤로 당기기", "가슴 밀기", "가슴 모으기",
          "어깨 위로 밀기", "실내 자전거", "자전거타기", "플렉스 바", "목봉", "팔꿈치 굽히기", "팔꿈치 펴기",
          "몸통 움츠리기", "턱걸이", "빌리보", "딩딩링", "몸통 들어올리기"]
KW_TOOL = ["덤벨", "탄력밴드", "밴드", "짐볼", "폼롤러", "공 ", "공을", "볼 ", "볼을", "풍선", "줄넘기"]


def place_of(name):
    if any(k in name for k in KW_POOL):
        return 3
    if any(k in name for k in KW_GYM):
        return 2
    if any(k in name for k in KW_TOOL) or name.endswith("공") or name.endswith("볼"):
        return 1
    return 0


def parse_rx(text):
    out = {"prep": [], "main": [], "cool": []}
    for part in text.split(" / "):
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        key = SECTIONS.get(k.strip())
        if not key:
            continue
        for e in v.split(","):
            e = re.sub(r"\s+", " ", e).strip()
            if e and e not in out[key]:
                out[key].append(e)
    return out


# --------------------------------------------------------------------
# 5. 비슷한 어르신 찾기 + 인증등급 예측 (kNN)
#    핵심 3항목이 모두 있는 기록만 사용. 성별을 나눈 뒤
#    [나이, 의자, 제자리걷기, 3m] 를 표준화해 유클리드 거리로 이웃 K명을 찾는다.
#    앱에 싣는 레코드에는 회원ID·센터명·측정일을 넣지 않는다(비식별 축약).
#    홀드아웃 2,000건으로 '이웃 등급 중앙값'의 적중률을 측정해 meta에 기록.
# --------------------------------------------------------------------
KNN_FEATS = ["age", "chair", "step", "tug"]


def evaluate_knn(c):
    rng = np.random.default_rng(SEED)
    c = c.dropna(subset=["grade"]).reset_index(drop=True)
    idx = rng.permutation(len(c))
    te, tr = idx[:2000], idx[2000:]
    hits = within1 = total = 0
    for sex in ["F", "M"]:
        trs = [i for i in tr if c.sex[i] == sex]
        tes = [i for i in te if c.sex[i] == sex]
        mu = c.loc[trs, KNN_FEATS].mean()
        sd = c.loc[trs, KNN_FEATS].std()
        A = ((c.loc[trs, KNN_FEATS] - mu) / sd).values
        B = ((c.loc[tes, KNN_FEATS] - mu) / sd).values
        nb = NearestNeighbors(n_neighbors=KNN_K).fit(A).kneighbors(B, return_distance=False)
        g = c.grade.values[trs]
        for j, i in enumerate(tes):
            pred = int(np.floor(np.median(g[nb[j]]) + 0.5))
            hits += int(pred == c.grade[i])
            within1 += int(abs(pred - c.grade[i]) <= 1)
            total += 1
    return {"k": KNN_K, "n_test": total, "exact": round(hits / total, 3),
            "within1": round(within1 / total, 3), "seed": SEED,
            "method": "성별 분리, [나이·의자·제자리걷기·3m] 표준화 유클리드 거리, 이웃 등급 중앙값"}


GRADE6_FROM = "20250602"   # 인증등급 6등급 체계 시행일. 이전은 1~3등급+참가증이라 섞지 않음


def build_grade_knn(s):
    """예상 인증등급용 이웃 데이터: 6등급 체계 이후, 3항목이 모두 있는 기록."""
    c = s[(s.de >= GRADE6_FROM) & s.grade.between(1, 6)].dropna(subset=list(TESTS)).copy().reset_index(drop=True)
    out = {}
    for sex in ["F", "M"]:
        g = c[c.sex == sex]
        mu = g[KNN_FEATS].mean(); sd = g[KNN_FEATS].std()
        out[sex] = {"mu": [round(float(x), 3) for x in mu], "sd": [round(float(x), 3) for x in sd],
                    "feats": KNN_FEATS, "tug_scale": 10,
                    "rows": [[int(r.age), int(r.chair), int(r.step), int(round(r.tug * 10)), int(r.grade)]
                             for r in g.itertuples()]}
    return out, c


def evaluate_grade_knn(c, test_frac=0.1):
    """홀드아웃 검증: 기록의 10%를 숨기고 나머지로 등급을 예측해 실제와 비교.
    기준선: 계산 없이 한 등급만 찍었을 때(정확 일치는 최빈 등급, ±1은 가장 유리한 등급)."""
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(c)); n_te = int(len(c) * test_frac)
    te, tr = idx[:n_te], idx[n_te:]
    hits = within1 = total = 0
    for sex in ["F", "M"]:
        trs = [i for i in tr if c.sex[i] == sex]; tes = [i for i in te if c.sex[i] == sex]
        mu = c.loc[trs, KNN_FEATS].mean(); sd = c.loc[trs, KNN_FEATS].std()
        A = ((c.loc[trs, KNN_FEATS] - mu) / sd).values; B = ((c.loc[tes, KNN_FEATS] - mu) / sd).values
        nb = NearestNeighbors(n_neighbors=KNN_K).fit(A).kneighbors(B, return_distance=False)
        g = c.grade.values[trs]
        for j, i in enumerate(tes):
            pred = int(np.floor(np.median(g[nb[j]]) + 0.5))
            hits += int(pred == c.grade[i]); within1 += int(abs(pred - c.grade[i]) <= 1); total += 1
    gt, gtr = c.grade.values[te], c.grade.values[tr]
    mode = int(pd.Series(gtr).mode().iloc[0])
    best1 = max(range(1, 7), key=lambda k: float((abs(gt - k) <= 1).mean()))
    return {"k": KNN_K, "n_train": int(len(tr)), "n_test": int(total), "seed": SEED,
            "exact": round(hits / total, 3), "within1": round(within1 / total, 3),
            "baseline_exact": round(float((gt == mode).mean()), 3), "baseline_exact_grade": mode,
            "baseline_within1": round(float((abs(gt - best1) <= 1).mean()), 3), "baseline_within1_grade": best1,
            "period_from": GRADE6_FROM,
            "method": "성별 분리, [나이·의자·제자리걷기·3m] 표준화 유클리드 거리, 이웃 100명 등급 중앙값"}


def build_knn(s):
    c = s.dropna(subset=list(TESTS)).copy().reset_index(drop=True)
    ex_index, ex_count = {}, collections.Counter()
    parsed = []
    for t in c.rx:
        p = parse_rx(t)
        parsed.append(p)
        for sec in p:
            for e in p[sec]:
                ex_count[e] += 1
    for e, _ in ex_count.most_common():
        ex_index[e] = len(ex_index)
    exercises = [[e, place_of(e), ex_count[e]] for e in ex_index]

    knn = {}
    for sex in ["F", "M"]:
        g = c[c.sex == sex]
        mu = g[KNN_FEATS].mean()
        sd = g[KNN_FEATS].std()
        rows = []
        for i, r in g.iterrows():
            p = parsed[i]
            rows.append([int(r.age), int(r.chair), int(r.step), int(round(r.tug * 10)),
                         int(r.grade) if not np.isnan(r.grade) else 0,
                         [ex_index[e] for e in p["prep"]],
                         [ex_index[e] for e in p["main"]],
                         [ex_index[e] for e in p["cool"]]])
        knn[sex] = {"mu": [round(float(x), 3) for x in mu], "sd": [round(float(x), 3) for x in sd],
                    "feats": KNN_FEATS, "tug_scale": 10, "rows": rows}
    return knn, exercises, c


# --------------------------------------------------------------------
# 6. 국민체력100 센터 (기능: '가까운 국민체력100 센터')
#    원천에는 센터명(CNTER_NM)만 있어 시도/시군구는 센터명 기준으로
#    개발자가 수작업 매핑했다. 시군구가 불명확하면 None(시도 단위 표시).
#    시군구 표기는 [D2] 체육시설 데이터의 행정구역명과 맞췄다.
# --------------------------------------------------------------------
G = "전남광주통합특별시"
CENTER_REGION = {
    "원주": ("강원특별자치도", "원주시"), "시흥": ("경기도", "시흥시"), "동구(광주)": (G, "동구"),
    "전주": ("전북특별자치도", "전주시"), "미추홀": ("인천광역시", None), "세종": ("세종특별자치시", "세종시"),
    "북구(광주)": (G, "북구"), "사천": ("경상남도", "사천시"), "KSPO송파(출장A)": ("서울특별시", "송파구"),
    "KSPO광주(출장)": (G, None), "군산": ("전북특별자치도", "군산시"), "사하": ("부산광역시", "사하구"),
    "서구(광주)": (G, "서구"), "은평": ("서울특별시", "은평구"), "강릉": ("강원특별자치도", "강릉시"),
    "고양": ("경기도", "고양시"), "연제": ("부산광역시", "연제구"), "포항": ("경상북도", "포항시"),
    "경남": ("경상남도", None), "성동": ("서울특별시", "성동구"), "남구(부산)": ("부산광역시", "남구"),
    "수원": ("경기도", "수원시"), "포천": ("경기도", "포천시"), "청주": ("충청북도", "청주시"),
    "목포": (G, "목포시"), "화성": ("경기도", "화성시"), "제주": ("제주특별자치도", "제주시"),
    "서구(대전)": ("대전광역시", "서구"), "계룡": ("충청남도", "계룡시"), "동구(인천)": ("인천광역시", None),
    "스포원(금정)": ("부산광역시", "금정구"), "정읍": ("전북특별자치도", "정읍시"), "익산": ("전북특별자치도", "익산시"),
    "성남": ("경기도", "성남시"), "KSPO아산(출장)": ("충청남도", "아산시"), "안동": ("경상북도", "안동시"),
    "KSPO송파": ("서울특별시", "송파구"), "사상": ("부산광역시", "사상구"), "영암": (G, "영암군"),
    "동해": ("강원특별자치도", "동해시"), "창원": ("경상남도", "창원시"), "나주": (G, "나주시"),
    "곡성": (G, "곡성군"), "순천": (G, "순천시"), "중구(서울)": ("서울특별시", "중구"),
    "남원": ("전북특별자치도", "남원시"), "충주": ("충청북도", "충주시"), "광명": ("경기도", "광명시"),
    "양평": ("경기도", "양평군"), "천안": ("충청남도", "천안시"), "춘천": ("강원특별자치도", "춘천시"),
    "구미": ("경상북도", "구미시"), "증평": ("충청북도", "증평군"), "영동": ("충청북도", "영동군"),
    "신안": (G, "신안군"), "김천": ("경상북도", "김천시"), "도봉": ("서울특별시", "도봉구"),
    "삼척": ("강원특별자치도", "삼척시"), "의정부": ("경기도", "의정부시"), "KSPO대구(출장A)": ("대구광역시", None),
    "KSPO대구": ("대구광역시", None), "안산": ("경기도", "안산시"), "달서": ("대구광역시", "달서구"),
    "무안": (G, "무안군"), "서울시": ("서울특별시", None), "중구을지": ("서울특별시", "중구"),
    "KSPO아산": ("충청남도", "아산시"), "서대문구보건소": ("서울특별시", "서대문구"), "오산": ("경기도", "오산시"),
    "영등포": ("서울특별시", "영등포구"), "송파구보건소": ("서울특별시", "송파구"), "KSPO광주": (G, None),
    "강동": ("서울특별시", "강동구"), "KSPO대구(출장B)": ("대구광역시", None), "금천": ("서울특별시", "금천구"),
    "영주": ("경상북도", "영주시"), "구의": ("서울특별시", "광진구"), "마포망원": ("서울특별시", "마포구"),
    "진천": ("충청북도", "진천군"), "동작": ("서울특별시", "동작구"), "경산": ("경상북도", "경산시"),
    "보은": ("충청북도", "보은군"), "KSPO송파별관(어르신)": ("서울특별시", "송파구"), "중랑": ("서울특별시", "중랑구"),
    "서초": ("서울특별시", "서초구"), "마포서강": ("서울특별시", "마포구"), "관악": ("서울특별시", "관악구"),
    "구로": ("서울특별시", "구로구"), "태백": ("강원특별자치도", "태백시"), "군자": ("서울특별시", "광진구"),
    "진주(자체운영)": ("경상남도", "진주시"), "용산": ("서울특별시", "용산구"), "인천광역시": ("인천광역시", None),
}


def build_centers(df_all):
    total = df_all.CNTER_NM.value_counts()
    senior = df_all[df_all.AGRDE_FLAG_NM == "어르신"].CNTER_NM.value_counts()
    out, unmapped = [], []
    for name, n in total.items():
        sido, sgg = CENTER_REGION.get(name, (None, None))
        if sido is None:
            unmapped.append(name)
        out.append({"name": name, "sido": sido, "sgg": sgg,
                    "senior": int(senior.get(name, 0)), "total": int(n),
                    "mobile": "출장" in name})
    return out, unmapped


# --------------------------------------------------------------------
# 7. [D2] 공공 체육시설 (기능: '처방 운동을 할 수 있는 가까운 곳')
#    조건: 시설구분(FCLTY_FLAG_NM)='공공', 운영상태='정상운영', 삭제여부='N',
#          좌표(FCLTY_LA/LO)가 국내 범위, 시군구명 존재.
#    유형 분류(앱 표시용):
#      0 체육센터·생활체육관 : 시설유형 '생활체육관' 또는 이름에 '체육센터/체력단련/헬스' 포함
#      2 수영장              : 시설유형 '수영장'
#      3 동네 운동시설       : 시설유형 '간이운동장' (마을회관·경로당·공원 주민운동시설 등)
#    좌표는 소수 4자리(약 10m)로 줄여 용량을 낮추고, 이름+좌표 중복은 제거.
#    시설명에 '테스트'가 들어간 시험 입력 행은 제외.
# --------------------------------------------------------------------
def build_facilities(csv_path):
    # 원천 CSV에 따옴표가 깨진 행이 섞여 있어 python 엔진 + 불량행 콜백으로 읽는다.
    bad = []
    d = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, engine="python",
                    on_bad_lines=lambda line: bad.append(line) or None)
    n_all = len(d)
    p = d[(d.FCLTY_FLAG_NM == "공공") & (d.FCLTY_STATE_VALUE == "정상운영") & (d.DEL_AT == "N")].copy()
    n_public = len(p)
    gym_kw = p.FCLTY_NM.fillna("").str.contains("체육센터|체력단련|헬스")
    p["kind"] = np.select(
        [p.FCLTY_TY_NM.eq("생활체육관") | (p.FCLTY_TY_NM.eq("기타체육시설(체력단련장)") & gym_kw),
         p.FCLTY_TY_NM.eq("수영장"),
         p.FCLTY_TY_NM.eq("간이운동장")],
        [0, 2, 3], default=-1)
    p = p[p.kind >= 0]
    p["lat"] = pd.to_numeric(p.FCLTY_LA, errors="coerce").round(4)
    p["lon"] = pd.to_numeric(p.FCLTY_LO, errors="coerce").round(4)
    p = p[p.lat.between(33, 39) & p.lon.between(124, 132) & p.SIGNGU_NM.notna() & p.CTPRVN_NM.notna()]
    p["FCLTY_NM"] = p.FCLTY_NM.str.replace(r"\s+", " ", regex=True).str.strip()
    # 시험 입력으로 보이는 행 제외 (예: '간이운동장 테스트')
    p = p[~p.FCLTY_NM.str.contains("테스트|test|TEST", na=True)]
    p = p.drop_duplicates(["FCLTY_NM", "lat", "lon"])
    sidos = sorted(p.CTPRVN_NM.unique())
    regions = {sd: sorted(p[p.CTPRVN_NM == sd].SIGNGU_NM.unique()) for sd in sidos}
    rows = []
    for r in p.itertuples():
        si = sidos.index(r.CTPRVN_NM)
        gi = regions[r.CTPRVN_NM].index(r.SIGNGU_NM)
        tel = r.FCLTY_TEL_NO if isinstance(r.FCLTY_TEL_NO, str) else ""
        rows.append([r.FCLTY_NM, int(r.kind), si, gi, float(r.lat), float(r.lon), tel])
    info = {"rows_all": int(n_all), "bad_lines_skipped": len(bad), "rows_public_operating": int(n_public), "rows_used": len(rows),
            "kind_counts": {str(k): int(v) for k, v in p.kind.value_counts().items()}}
    return {"sidos": sidos, "regions": regions, "kinds": ["체육센터·생활체육관", "", "수영장", "동네 운동시설"],
            "rows": rows}, info


# --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df, load_info = load_fitness(a.raw)
    s_rx = clean_seniors(df)                       # [D1] 2026.05~07: 운동처방이 있는 기록
    covered = sorted(df.MESURE_DE.str[:6].unique())
    it, item_info = load_items(a.raw, covered)     # [D3] 2022.01~2026.04
    s_all = clean_seniors(pd.concat([df, it], ignore_index=True))   # 순위·체력나이·등급용 전체
    mapping_check = verify_mapping(s_all)
    norms = build_norms(s_all)
    curve = build_age_curve(s_all)
    knn, exercises, complete = build_knn(s_rx)     # 운동처방 이웃은 [D1]만
    knn_grade, grade_set = build_grade_knn(s_all)
    knn_eval = evaluate_grade_knn(grade_set)
    s = s_all
    centers, unmapped = build_centers(df)
    fac_csv = [f for f in os.listdir(a.raw) if f.startswith("KS_WNTY_PHSTRN_FCLTY_STTUS")][0]
    facilities, fac_info = build_facilities(os.path.join(a.raw, fac_csv))

    meta = {
        "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "fitness": load_info,
        "items": item_info,
        "all_ages_records": int(len(df) + len(it)),
        "period": {"min": str(s_all.de.min()), "max": str(s_all.de.max())},
        "grade_set": {"records": int(len(grade_set)), "from": GRADE6_FROM},
        "rx_set": {"records_complete": int(len(complete)), "with_prescription": int((complete.rx != "").sum()),
                   "period_min": str(s_rx.de.min()), "period_max": str(s_rx.de.max()),
                   "unique_members_D1": int(df[df.AGRDE_FLAG_NM == "어르신"].MBER_SEQ_NO_VALUE.nunique())},
        "seniors": {
            "records": int(len(s)), "outreach_share": round(float((s.place == "출장").mean()), 3),
            "by_sex": s.sex.value_counts().to_dict(),
            "by_band": s.band.value_counts().to_dict(),
            "valid_counts": {t: int(s[t].notna().sum()) for t in TESTS},
            "complete_3tests": int(len(complete)),
            "with_prescription": int((complete.rx != "").sum()),
        },
        "mapping_check": mapping_check,
        "knn_eval": knn_eval,
        "exercises": {"unique": len(exercises),
                      "by_place": collections.Counter(str(e[1]) for e in exercises)},
        "centers": {"count": len(centers), "unmapped": unmapped},
        "facilities": fac_info,
        "sources": [
            {"org": "서울올림픽기념국민체육진흥공단", "name": "체력측정 및 운동처방 종합 데이터",
             "platform": "문화빅데이터플랫폼", "files": load_info["files"]},
            {"org": "서울올림픽기념국민체육진흥공단", "name": "체력측정 항목별 측정 데이터",
             "platform": "문화빅데이터플랫폼", "files": item_info["files"]},
            {"org": "서울올림픽기념국민체육진흥공단", "name": "전국체육시설현황 데이터",
             "platform": "문화빅데이터플랫폼", "files": [fac_csv]},
        ],
    }
    out = {"norms": norms, "age_curve": curve, "knn": knn, "knn_grade": knn_grade, "exercises": exercises,
           "centers": centers, "facilities": facilities, "meta": meta}
    for k, v in out.items():
        with open(os.path.join(a.out, f"{k}.json"), "w", encoding="utf-8") as fp:
            json.dump(v, fp, ensure_ascii=False, separators=(",", ":"))
    print(json.dumps(meta, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
