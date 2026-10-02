# -*- coding: utf-8 -*-
"""
기대효과(정량) 산출 근거 스크립트
--------------------------------------------------------------------
보고서 2-5 기대효과 표의 데이터 기반 수치를 재현한다.
  ① 정밀측정 권고 대상 비율: 3항목(의자·제자리걷기·3m) 중 하나 이상이
     같은 성별·5세 단위 연령대에서 하위 10% 이하인 어르신 비율
  ② 체력나이가 실제 나이보다 2세 이상 많은 어르신 비율
     (앱과 같은 체력나이 계산식: data/age_curve.json 곡선 역보간, 항목 평균)
대상: preprocess.py와 같은 결합·정제 규칙을 거친, 3항목이 모두 있는 기록 전체
결과: data/effects.json
실행: python3 scripts/effects.py --raw raw --data data
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocess as P  # 같은 적재·정제 규칙 재사용


def fit_age(curve, test, v, higher_better):
    """앱(index.html)의 fitAge()와 같은 규칙: 곡선상 같은 값이 나오는 나이, 65~90세로 제한"""
    ages = curve["ages"]
    g = np.array(curve[test]) if higher_better else -np.array(curve[test])
    vv = v if higher_better else -v
    if vv >= g[0]:
        return float(ages[0])
    if vv <= g[-1]:
        return float(ages[-1])
    eq = np.where(g == vv)[0]
    if len(eq):
        return ages[0] + (eq[0] + eq[-1]) / 2
    for i in range(len(g) - 1):
        if g[i] > vv > g[i + 1]:
            return ages[i] + (g[i] - vv) / (g[i] - g[i + 1])
    return float(ages[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw")
    ap.add_argument("--data", default="data")
    a = ap.parse_args()

    df, _ = P.load_fitness(a.raw)
    covered = sorted(df.MESURE_DE.str[:6].unique())
    it, _ = P.load_items(a.raw, covered)
    s = P.clean_seniors(pd.concat([df, it], ignore_index=True))
    c = s.dropna(subset=list(P.TESTS)).copy()

    # ① 동년배 하위 10% (같은 성별·연령대 내 순위, 3m는 낮을수록 좋음)
    for t, hb in P.TESTS.items():
        c[t + "_better"] = c.groupby(["sex", "band"])[t].rank(pct=True, ascending=hb) * 100
    low10 = (c[[t + "_better" for t in P.TESTS]] <= 10).any(axis=1)

    # ② 체력나이: 값이 이산적이라 (성별, 항목, 값)별로 한 번씩 계산해 매핑
    curve = json.load(open(os.path.join(a.data, "age_curve.json"), encoding="utf-8"))
    ages = np.zeros(len(c))
    for t, hb in P.TESTS.items():
        col = np.zeros(len(c))
        for sex in ["F", "M"]:
            m = (c.sex == sex).values
            uniq = np.unique(c.loc[m, t].values)
            lut = {u: fit_age(curve[sex], t, float(u), hb) for u in uniq}
            col[m] = c.loc[m, t].map(lut).values
        ages += col
    fit = np.round(ages / len(P.TESTS))
    older2 = (fit - c.age.values) >= 2

    out = {
        "records": int(len(c)),
        "period": {"min": str(c.de.min()), "max": str(c.de.max())},
        "low10_any_share": round(float(low10.mean()), 4),
        "fitage_plus2_share": round(float(older2.mean()), 4),
        "per_1000": {"low10_any": int(round(low10.mean() * 1000)),
                     "fitage_plus2": int(round(older2.mean() * 1000))},
        "assumption_visit_rate": 0.10,
        "per_1000_center_visits": int(round(low10.mean() * 1000 * 0.10)),
        "note": "이용자 분포가 센터 측정 어르신과 같다는 가정. 방문 전환율 10%는 가정치",
    }
    with open(os.path.join(a.data, "effects.json"), "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
