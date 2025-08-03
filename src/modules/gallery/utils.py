import numpy as np


def choose_index(num_update, len_gal, len_que, max_no_embds):
    if len_gal + len_que <= max_no_embds:
        num_gal = len_gal
        num_que = len_que 
    else:
        num_que = max_no_embds // num_update if max_no_embds // num_update < len_que else len_que
        num_gal = max_no_embds - num_que
    gal_indexes = get_index(len_gal, num_gal)
    que_indexes = get_index(len_que, num_que)
    for index in que_indexes:
        gal_indexes.append(index + len_gal)
    return gal_indexes
    
def get_index(len, num_get):
    indexes, remove_indexes= [], []
    num_remove = len-num_get
    for i in range(1, num_remove + 1):
        k = i * (len - 1.0)/ (num_remove + 1)
        g = int(k)
        if k - g >=0.5: g += 1
        remove_indexes.append(g)
    for i in range(0, len):
        if i not in remove_indexes:
            indexes.append(i)
    return indexes
        