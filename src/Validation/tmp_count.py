import json, collections

with open("data/base_data.json", "r", encoding="utf-8") as f:
    d = json.load(f)

ops = []

def collect(x):
    if isinstance(x, list):
        for a in x:
            collect(a)
    elif isinstance(x, dict):
        if x.get("objectType") == "OPERATION":
            ops.append(x)
        for v in x.values():
            collect(v)

collect(d)

c = collections.Counter(int(o.get("workCenterMachineId") or 0) for o in ops)
print("ops=", len(ops))
print("top10 workCenterMachineId:", c.most_common(10))
