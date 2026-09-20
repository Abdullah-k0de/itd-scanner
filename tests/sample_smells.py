import pandas as pd

# 1. Collection Choker + Math Looper
for idx, row in df.iterrows():
    total += row['value'] * 2.0

# 2. RAM Hog
res = sum([x * 2 for x in range(100)])

# 3. String Masher
s = ""
for item in items:
    s += str(item)

# 4. DIY Wheel
seen = []
for x in items:
    if x not in seen:
        seen.append(x)
