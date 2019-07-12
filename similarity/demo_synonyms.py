import synonyms

print(synonyms.seg("中国南海"))
print(synonyms.seg("中南海"))

print("中国:{}".format(synonyms.nearby("人脸")))

print(synonyms.compare("西北","塑料",seg=False))
