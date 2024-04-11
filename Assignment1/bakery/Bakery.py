import csv
from itertools import combinations
import tkinter as tk
from tkinter import ttk
def read_csv_file(file_path, percentage):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        total_rows = sum(1 for row in csv_reader)
        rows_to_read = int(percentage / 100 * total_rows)
        file.seek(0)  # Reset file pointer to read from the beginning
        csv_reader = csv.DictReader(file)  # Reinitialize CSV reader
        for index, row in enumerate(csv_reader):
            if index >= rows_to_read:
                break
            transaction_no = row['TransactionNo']
            items = row['Items']
            data.append((transaction_no, items))
    return data

def collect_items_by_transaction(csv_data):
    transaction_items = {}
    for transaction_no, items in csv_data:
        if transaction_no in transaction_items:
            transaction_items[transaction_no].add(items)
        # Use set to avoid duplicates
        else:
            transaction_items[transaction_no] = {items}  
    return transaction_items

def vertical_format(transaction_items):
    vertical_items = {}
    for transaction_no, items in transaction_items.items():
        for item in items:
            if item in vertical_items:
                vertical_items[item].add(transaction_no)
            else:
                vertical_items[item] = {transaction_no}
    return vertical_items

def vertical_data_format_algorithm_data(percent):
    file_path = 'bakery/Bakery.csv'
    csv_data = read_csv_file(file_path,percent)
    transaction_items = collect_items_by_transaction(csv_data)
    vertical_items = vertical_format(transaction_items)
    return vertical_items


def frequent_item_sets(vertical_items, min_support):
    frequent_items = {}
    for item, transaction_no in vertical_items.items():
        if len(transaction_no) >= min_support:
            frequent_items[item] = transaction_no
    # merge frequent items to get new items and set transaction_no common to both items
    new_items = {}
    # merge 2 items and set transaction_no common to both items without repetition
    frequent_items_list = list(frequent_items.items())
    for i in range(len(frequent_items_list)):
        for j in range(i + 1, len(frequent_items_list)):
            item1, transaction_no1 = frequent_items_list[i]
            item2, transaction_no2 = frequent_items_list[j]
            li = []
            it1 = item1.split(',')
            it2 = item2.split(',')
            for k in it1:
                li.append(k)
            for k in it2:
                li.append(k)
            li = list(set(li))
            new_item = ','.join(li)
            new_transaction_no = transaction_no1 & transaction_no2
            new_items[new_item] = new_transaction_no
    if all(len(transaction_no) < min_support for transaction_no in new_items.values()):
        return {}
    else:
        return new_items

def vertical_data_format_algorithm_frequent_item_sets(vertical_items, support):
    min_support = support
    frequent_items = vertical_items
    frequent_items_final = {}
    while True:
        new_frequent_items = frequent_item_sets(frequent_items, min_support)
        if new_frequent_items == {}:
            for item, transaction_no in frequent_items.items():
                if len(transaction_no) >= min_support:
                    frequent_items_final[item] = transaction_no
            break
        else:
            frequent_items = new_frequent_items
    # print('Frequent Items final++++++++++')
    # for item, transaction_no in list(frequent_items_final.items())[:]:
    #     print(f'{item}: {transaction_no}')
    # print('-----------------------')
    return frequent_items_final


def combinations_set(min_support_transaction_items):
    comb = {}
    for item, transaction_no in min_support_transaction_items.items():
        li = item.split(',')
        left_subsets = []
        for i in range(1, len(li)):
            left_subsets.extend(combinations(li, i))
        for left_subset in left_subsets:
            right_subset = set(li) - set(left_subset)
            if left_subset in comb:
                comb[left_subset].append(right_subset)
            else:
                comb[left_subset] = [right_subset]
    return comb

def strong_item_sets(vertical_items, left_subset, right_subset, min_confidence):
    strong_item_sets = {}
    if len(left_subset) == 1:
        left_subset = left_subset[0]
    else:
        left_subset = ','.join(left_subset)
    if len(right_subset) == 1:
        right_subset = next(iter(right_subset))
    else:
        right_subset = ','.join(right_subset)
    
    left_subset_transaction_no = {}
    for item, transaction_no in vertical_items.items():
        if item in left_subset:
            left_subset_transaction_no[item] = transaction_no
    # print('left_subset_transaction_no', left_subset_transaction_no)
    left_subset_transaction_no = set.intersection(*left_subset_transaction_no.values())
    # print('common_left_subset_transaction_no', left_subset_transaction_no)
    left_count = len(left_subset_transaction_no)

    right_subset_transaction_no = {}
    for item, transaction_no in vertical_items.items():
        if item in right_subset:
            right_subset_transaction_no[item] = transaction_no
    # Merge the transaction numbers from both subsets
    left_right_subset_transaction_no = set.intersection(*[set(txns) for txns in [left_subset_transaction_no] + list(right_subset_transaction_no.values())])
    left_right_cont = len(left_right_subset_transaction_no)
    
    confidence = left_right_cont / left_count
    # print('confidence', confidence)
    if confidence >= min_confidence:
        # print(f'{left_subset} => {right_subset} : {confidence}')
        strong_item_sets[f'{left_subset} => {right_subset}'] = confidence
    return strong_item_sets

def vertical_data_format_algorithm_strong_item_sets(vertical_items, frequent_items_final, confidence):
    combinations = combinations_set(frequent_items_final)
    # for left_subset, right_subset in list(combinations.items())[:]:
    #     print(f'{left_subset} => {right_subset}')
    strong_item = {}
    min_confidence = confidence / 100
    for i in combinations:
        left_subset = i
        for j in combinations[i]:
            right_subset = j
            strong_item = strong_item_sets(vertical_items, left_subset, right_subset, min_confidence)
    return strong_item        


vertical_items = vertical_data_format_algorithm_data(70)
print('Vertical Items:')
for item, transaction_no in list(vertical_items.items())[:]:
    print(f'{item}: {transaction_no}')
frequent_items_final = vertical_data_format_algorithm_frequent_item_sets(vertical_items, 2)
print('Frequent Items:')
for item, transaction_no in list(frequent_items_final.items())[:]:
    print(f'{item}: {transaction_no}')
strong_item = vertical_data_format_algorithm_strong_item_sets(vertical_items, frequent_items_final, 30)
print('Strong Association Rules:')
for item, confidence in list(strong_item.items())[:]:
    print(f'{item}: {confidence}')
