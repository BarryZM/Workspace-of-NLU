# coding:utf-8
# get text infomation

def _get_length(line):
    return len(line)

def read_file_length_count(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        len_list = list(map(len, lines)) 
        #len_list = list(map(_get_length, lines)) 
        len_list.sort()
        len_list.reverse()
        print(len_list)


if __name__ == '__main__':
    read_file_length_count('origin.txt')
