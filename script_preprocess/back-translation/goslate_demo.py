#!/usr/bin/python3# -*- coding: utf-8 -*-# @Time    : 2019-03-14 10:43# @Author  : apollo2mars# @File    : goslate_demo.py# @Contact : apollo2mars@gmail.com# @Desc    :import goslateimport timefrom urllib import requestimport goslateimport randomimport sys, osabs_path = os.path.abspath(os.path.dirname(__file__))base_dir = abs_path[:abs_path.find("NLU-AI-SPEAKER") + len("NLU-AI-SPEAKER")]sys.path.append(base_dir)from utils.data_helper import open_fileinput_list = ['你好', '介绍一下故宫', '定一个早上八点的闹钟', '明天天气如何', '播放孙一的评书']lang_list = ['en', 'ko', 'ja', 'de', 'fr', 'ru', 'la']def read_proxy_list(input_file:str):	r_list = []	for line in open_file(input_file).readlines():		tmp_dict = {}		tmp_dict['http'] = line.strip()		r_list.append(tmp_dict)	return r_listdef _run():	proxy_list = read_proxy_list('Proxy List.txt')	# proxy_list = [	# 	{"http": "124.88.67.81:80"},	# 	{"http": "124.88.67.81:80"},	# 	{"http": "124.88.67.81:80"},	# 	{"http": "124.88.67.81:80"},	# 	{"http": "124.88.67.81:80"}	# ]	for item in input_list:		# 随机选择一个代理		proxy = random.choice(proxy_list)		print(proxy)		proxy_handler = request.ProxyHandler(proxy)		proxy_opener = request.build_opener(request.HTTPHandler(proxy_handler), request.HTTPSHandler(proxy_handler))		gs_with_proxy = goslate.Goslate(opener=proxy_opener, service_urls=['http://translate.google.com'])		for	lang in lang_list:			print(item)			try:				print(gs_with_proxy.translate(gs_with_proxy.translate(item, lang), 'zh'))				time.sleep(10)			except:				print("translate error")				time.sleep(60)def _run_simlpe():	gs = goslate.Goslate(service_urls=['http://translate.google.com'])	for item in input_list:		print(item)		for	lang in lang_list:			try:				tmp = gs.translate(item, lang)				time.sleep(10)				ch1 = gs.translate(tmp, 'zh')				print(ch1)				time.sleep(10)			except:				print("translate error")				time.sleep(60)_run_simlpe()# _run()"""如果出现error， 则保持当前句子，间隔一段时间后继续翻译"""