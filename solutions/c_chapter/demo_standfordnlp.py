# https://blog.csdn.net/qq_35203425/article/details/80451243
import stanfordnlp
# stanfordnlp.download('zh')   # This downloads the Chinese models for the neural pipeline
nlp = stanfordnlp.Pipeline('zh') # This sets up a default neural pipeline in English
doc = nlp("奥巴马出生于夏威夷，他于2008年当选总统")
doc.sentences[0].print_dependencies()