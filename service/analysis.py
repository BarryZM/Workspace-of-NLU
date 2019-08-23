count_dict = {}
count = 0

#with open('./absa_clf/train-term-category.txt', mode='r', encoding='utf-8') as f:
with open('bolang_600.txt', mode='r', encoding='utf-8') as f:
    lines = f.readlines()
    print(len(lines))

    while count < len(lines) - 3:
        term = lines[count+1].strip()
        category = lines[count+2].strip()
        polarity = lines[count+3].strip()
        #print(polarity)
        if polarity is not '0': # 0 is negative
            count = count + 4
            #print(">>> is not 0")
            continue
        if category in count_dict.keys():
            term_list = count_dict[category]
            term_list.append(term)
            count_dict[category] = term_list
        else:
            count_dict[category] = [term]
        count = count + 4 

#for key, value in count_dict.items():
#    print(key, len(value))


print("run done")
#print(count_dict)
#print(count_dict.keys())

new_dict = {}

from collections import Counter
for key, value in count_dict.items():
    new_dict[key]=Counter(value).most_common()
   
#print(new_dict)
for key, value in new_dict.items():
    print("##############2i###############")
    print('2-level category is :', key)
    print('term count as fellow', value)


import xlsxwriter
# Create an new Excel file and add a worksheet.
workbook = xlsxwriter.Workbook('bolang_600_negative.xlsx')
worksheet = workbook.add_worksheet()
# Widen the first column to make the text clearer.
#worksheet.set_column('A:B', 20000)
# Add a bold format to use to highlight cells.
bold = workbook.add_format({'bold': True})
# Write some simple text.
# worksheet.write('A1', 'Hello')
# Text with formatting.
# worksheet.write('A2', 'World', bold)
# Write some numbers, with row/column notation.
line_count = 0

for key, value in count_dict.items():
    worksheet.write(line_count, 0, key)
    worksheet.write(line_count, 1, len(value))
    line_count = line_count + 1

for item_key in new_dict.keys():
    worksheet.write(line_count, 0, ">>>>>>")
    line_count = line_count + 1
    worksheet.write(line_count, 0, item_key)
    line_count = line_count + 1
    for item_value in new_dict[item_key]:
        worksheet.write(line_count, 0, item_value[0])
        worksheet.write(line_count, 1, item_value[1])
        line_count = line_count + 1

# Insert an image.
#worksheet.insert_image('B5', 'logo.png')

workbook.close()

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
