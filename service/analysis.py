count_dict = {}
#count_dict = {'aaa':'111'}
count = 0

#with open('./absa_clf/train-term-category.txt', mode='r', encoding='utf-8') as f:
with open('result_all.txt', mode='r', encoding='utf-8') as f:
    lines = f.readlines()
    print(len(lines))

    while count < len(lines) - 3:
        term = lines[count+1].strip()
        category = lines[count+2].strip()
        if category in count_dict.keys():
            term_list = count_dict[category]
            term_list.append(term)
            count_dict[category] = term_list
        else:
            count_dict[category] = [term]
        count = count + 4 

for key, value in count_dict.items():
    print(key, len(value))


#print("run done")
#print(count_dict)
#print(count_dict.keys())

#new_dict = {}
#
#from collections import Counter
#for key, value in count_dict.items():
#    new_dict[key]=Counter(value).most_common()
#    
#print(new_dict)
#print(new_dict.keys())

#import pickle
#output=open('output.pkl', 'wb')
#pickle.dump(count_dict, output)
#output.close()
#
#input_1=open('output.pkl', 'rb')
#data = pickle.load(input_1)
#input_1.close()
#
#print(data)
