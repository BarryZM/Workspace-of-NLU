# coding:utf-8
# author:Apollo2Mars@gmail.com


def replace_str(inputs:str, start:int, end:int, word:str):
    """
    replace input[start:end] to 
    """ 
    print("\n\n###\n\n")

    print('inputs', inputs)
    print("start", start)
    print('end', end)
    if start != 0:
        pre_str = inputs[0:start]
        print("***", pre_str)
    else:
        print("start is 0")
        pre_str = ''

    if end != len(list(inputs))-1:
        post_str = inputs[end:len(list(inputs))-1]
        print("***", post_str)
    else:
        post_str = ''
        print('end is -1')

    print("pre", pre_str)
    print("word", word)
    print("post", post_str)
    all_str = pre_str + word + post_str
    print('all str is :', all_str)
 
    return all_str

def find_slots(labels:list, texts:list):
    slots_list = []
    tmp_dict = {}
    tmp_start = len(labels)-1
    tmp_end = 0
    for idx, item in enumerate(labels):
        if item.startswith("B"):
            tmp_start = idx
            idx += 1
            while idx < len(labels) and labels[idx].startswith("I"):
                idx += 1 
                tmp_end = idx
            tmp_dict['text'] = texts[tmp_start:tmp_end]
            tmp_dict['start'] = tmp_start
            tmp_dict['end'] = tmp_end
            slots_list.append(tmp_dict.copy())
    return slots_list

texts = []
labels = []
with open("/export/home/sunhongchao1/1-NLU/Workspace-of-NLU/corpus/sa/comment/air-purifier/label_100test/test.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    texts = []
    for item in lines:
        texts.append(item.split('\t')[0])
    

with open("outputs/entity_test_results.txt", 'r', encoding='utf-8') as f:
    labels = f.readlines()

sentence_text_list = []
sentence_label_list = []
sentence_slots_list = []

tmp_label_list = []
tmp_text_list = []
for label, text in zip(labels, texts):
    if label.strip() != "":
        tmp_label_list.append(label.strip('\n'))
        tmp_text_list.append(text.strip('\n'))
    else:
        sentence_label_list.append(tmp_label_list.copy())
        sentence_text_list.append(tmp_text_list.copy())
        tmp_label_list = []
        tmp_text_list = []

#print('sentence text:', sentence_text_list)  
#print('sentence label:', sentence_label_list)  

for text, label in zip(sentence_text_list, sentence_label_list):
    #print('text', text)
    slots = find_slots(label, text)
    #print("slots", slots) 
    sentence_slots_list.append(slots.copy())

print('sentence text is :', len(sentence_text_list))  
print('sentence label is :', len(sentence_label_list))  
print('sentence slots is :', len(sentence_slots_list))  

assert len(sentence_text_list) == len(sentence_label_list) == len(sentence_slots_list)

with open("result_absa_clf_training_data.txt", 'w', encoding='utf-8') as f:
    for line_item, slots_item in zip(sentence_text_list, sentence_slots_list):
        org_line = "".join(line_item)

        print("\n\n@@@@", org_line)

        for slots in slots_item:
            term = org_line[slots['start']:slots['end']] 
            if term is '':
                continue
            print("org_line", org_line)
            print("term", term)
            print('start', slots['start'])
            print('end', slots['end'])
            
            #line = org_line.replace(term, "$T")
            #print("new line", line)
            line = replace_str(org_line, int(slots['start']), int(slots['end']), '$T$')
            f.write(line+'\n'+term+'\n'+'味道'+'\n'+'0'+'\n')

