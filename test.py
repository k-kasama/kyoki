import MeCab
m = MeCab.Tagger('-Ochasen')
print(m.parse('半ライス餃子|ネギ塩から揚げ'))