# # from sklearn import metrics as mr
# import pickle,math
# fq=open("Questionkeyword.txt","r")
# fa=open("Answerkeyword.txt","r")
# fqa=open("totalkeyword.txt","r")
# # mr.mutual_info_score(label,x)
# Q = []
# A = []
# QA=[]
# Q_dict = pickle.load(open("Question.pkl",'rb'))
# A_dict = pickle.load(open("Answer.pkl",'rb'))
# QA_dict = pickle.load(open("total.pkl",'rb'))
# #print A_dict
# for l in fqa:
#     QA.append(l.strip().split(" "))
# index = 0
# #print A
# fqa=open("totalkeyword.txt","r")
# co_dict={}
# for l in fqa:
#     if l.strip()=="":continue
#     for w1 in l.strip().split(" "):
#        # if a not in A_dict:print 1;continue
#         last_inp = 0
#         for w2 in QA[index][last_inp:len(QA[index])]:
#             #print w1+"&"+w2
#            # if w2 not in A_dict or w1 not in A_dict:continue
#             if w1==w2:
#                if w1+"&"+w2 not in co_dict:
#                 co_dict[w1+"&"+w2] = 0
#                co_dict[w1+"&"+w2] += 1
#                continue
#             list_sorted = list(set([w1,w2]))
#             list_sorted.sort()
#             if "&".join(list_sorted) not in co_dict:
#                 co_dict["&".join(list_sorted)] = 0
#             co_dict["&".join(list_sorted)] += 1
#             last_inp+=1
#     index += 1
#
#     #A.append(l.strip().split(" "))
#
# #print A_dict
# #print QA_dict
# def total_takings(yearly_record):
#   sum = 0
#   for key,value in yearly_record.items():
#       sum += value
#   return sum
# #num_Q_word = total_takings(Q_dict)
# num_A_word = total_takings(QA_dict)
# total_num = num_A_word
# MMI_dict = {}
# #import
# for cd in co_dict.items():
#     w1 = cd[0].split("&")[0]
#     w2 = cd[0].split("&")[1]
#    # print a,q
#     # print float(co_dict[cd[0]])/(float(Q_dict[q])*float(A_dict[a]))
# #    print w1
#     if w1 not in QA_dict or  w2 not in QA_dict or QA_dict[w1]==0 or QA_dict[w2]==0:continue
#    # MMI_dict[a+"|"+q] = math.log((float(co_dict[cd[0]])/len(A))/((float(A_dict[q])/total_num)*(float(A_dict[a])/total_num)))
#    # MMI_dict[w1+"|"+w2] = math.log((float(co_dict[cd[0]])/len(A))/(float(A_dict[w1])/total_num)/(float(A_dict[w2])/total_num))
#     MMI_dict[w1+"|"+w2] = (float(co_dict[cd[0]])/len(QA))/(float(QA_dict[w2])/total_num)
#     MMI_dict[w2+"|"+w1] = (float(co_dict[cd[0]])/len(QA))/(float(QA_dict[w1])/total_num)
# sorted_list = sorted(MMI_dict.items(), key=lambda x: x[1],reverse=True)
# import pickle
# PMI = pickle.dump(MMI_dict,open("PMI.pkl",'wb'))
# while True:
#     input = raw_input(":")
#     num=0
#     for i in sorted_list[0:-1]:
#      #   print i[0]
#         if input == i[0].split("|")[-1]:
#             num+=1
#             print i[0],i[1]
#             if num>100:break
# # print d
# # for md in MMI_dict.items():
# #     print md[0],md[1]
#
#
