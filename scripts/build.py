# -*- coding: utf-8 -*-
"""
data/*.json 을 src/index.template.html 에 넣어 하나의 index.html 을 만든다.
(서버·외부 API 없이 정적 호스팅 한 파일로 동작하게 하기 위함)
실행: python3 scripts/build.py
"""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(name):
    with open(os.path.join(ROOT, "data", name + ".json"), encoding="utf-8") as f:
        return json.load(f)
meta = load("meta")
f = meta["fitness"]
period = f"{f['measure_date_min'][:4]}.{int(f['measure_date_min'][4:6])}.{int(f['measure_date_min'][6:])}~" \
         f"{f['measure_date_max'][:4]}.{int(f['measure_date_max'][4:6])}.{int(f['measure_date_max'][6:])}"
payload = {
    "norms": load("norms"),
    "age_curve": load("age_curve"),
    "knn": load("knn"),
    "exercises": [[e[0], e[1]] for e in load("exercises")],   # [운동명, 장소분류]
    "centers": load("centers"),
    "facilities": load("facilities"),
    "meta": {"seniors": meta["seniors"]["records"], "complete": meta["seniors"]["complete_3tests"],
             "period": period, "knn_eval": meta["knn_eval"], "built_at": meta["built_at"]},
}
with open(os.path.join(ROOT, "src", "index.template.html"), encoding="utf-8") as fp:
    tpl = fp.read()
js = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
out = tpl.replace("/*__CHEONGCHUN_DATA__*/null", js)
with open(os.path.join(ROOT, "index.html"), "w", encoding="utf-8") as fp:
    fp.write(out)
print("index.html", round(len(out.encode("utf-8")) / 1024), "KB")
