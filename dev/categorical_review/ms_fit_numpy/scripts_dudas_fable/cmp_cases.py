import json, sys
labels = sys.argv[1:]
d = {l: json.load(open(f"out/cases_{l}.json", encoding="utf-8")) for l in labels}
keys = list(d[labels[0]].keys())
def cell(r):
    if r["status"] == "error":
        return "ERR " + r["error"][:110]
    o = r["out"]
    s = json.dumps(o) if not isinstance(o, str) else o
    return "ok " + s[:230]
for k in keys:
    print("##", k)
    for l in labels:
        r = d[l].get(k)
        print(f"   {l:8s}:", cell(r) if r else "MISSING")
        ws = [w for w in r["warnings"] if not w.startswith(("MissingValuesWarning", "UserWarning: X does not"))] if r else []
        if ws:
            print(f"   {'':8s}  W:", " || ".join(w[:100] for w in ws)[:330])
