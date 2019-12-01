# -*- coding:utf -*-
# 测试 jieba 分词的几种模式
import jieba


test_text = '电话号码查询'

# 精确模式
seg_list1 = jieba.cut(test_text, cut_all=False)
print("_".join(seg_list1))

# 全模式
seg_list2 = jieba.cut(test_text, cut_all=True)
print("_".join(seg_list2))

# 搜索引擎模式
seg_list3 = jieba.cut_for_search(test_text)
print("_".join(seg_list3))