# Do NOT import pandas (to avoid the Python 3.12 astroid bug)
class DataFrame:
    pass

df = DataFrame()
for x in df:
    print(x)
