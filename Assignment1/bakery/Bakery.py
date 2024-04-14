import csv
from itertools import combinations
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

file_path = ''
verical_data = {}
frequent_data = {}

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
    global file_path
    if file_path == '':
        print('Please select a file')
        return
    # file_path = 'bakery/Bakery.csv'
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
    # for left_subset, right_subset in list(comb.items())[:]:
    #     print(f'{left_subset} => {right_subset}')
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
    print('confidence', confidence)
    if confidence >= min_confidence:
        print(f'{left_subset} => {right_subset} : {confidence}')
        strong_item_sets[f'{left_subset} => {right_subset}'] = confidence
    return strong_item_sets

def vertical_data_format_algorithm_strong_item_sets(vertical_items, frequent_items_final, confidence):
    combinations = combinations_set(frequent_items_final)
    # for left_subset, right_subset in list(combinations.items())[:]:
    #     print(f'{left_subset} => {right_subset}')
    strong_items = []
    min_confidence = confidence / 100
    for i in combinations:
        left_subset = i
        for j in combinations[i]:
            right_subset = j
            strong_item = strong_item_sets(vertical_items, left_subset, right_subset, min_confidence)
            strong_items.append(strong_item)
    # print strong_item
    print("lastttttt: ")
    for left_subset, right_subset in list(combinations.items())[:]:
        print(f'{left_subset} => {right_subset}') 
    return strong_items      


# vertical_items = vertical_data_format_algorithm_data(70)
# print('Vertical Items:')
# for item, transaction_no in list(vertical_items.items())[:]:
#     print(f'{item}: {transaction_no}')
# frequent_items_final = vertical_data_format_algorithm_frequent_item_sets(vertical_items, 2)
# print('Frequent Items:')
# for item, transaction_no in list(frequent_items_final.items())[:]:
#     print(f'{item}: {transaction_no}')
# strong_item = vertical_data_format_algorithm_strong_item_sets(vertical_items, frequent_items_final, 30)
# print('Strong Association Rules:')
# for item, confidence in list(strong_item.items())[:]:
#     print(f'{item}: {confidence}')

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=(("CSV files", "*.csv"), ("Excel files", "*.xls;*.xlsx"), ("Text files", "*.txt")))
    if file_path:
        file_entry.delete(0, tk.END)
        file_entry.insert(0, file_path)

def validate_confidence_input(P):
    # Validate the input for confidence percentage
    try:
        value = float(P)
        if 0 <= value <= 100:
            return True
        else:
            return False
    except ValueError:
        return False

def show_vertical_data():
    percentage = int(percentage_entry.get())
    global file_entry, file_path, verical_data
    file_entry = file_entry.get()
    file_path = file_entry
    verical_data = vertical_data_format_algorithm_data(percentage)
    result_text.delete(1.0, tk.END)
    for item, transaction_no in verical_data.items():
        result_text.insert(tk.END, f'{item}: {transaction_no}\n')

def show_frequent_item_sets():
    support = int(support_entry.get())
    global verical_data, frequent_data
    if verical_data == {}:
        print('Please get vertical data first')
        return
    frequent_data = vertical_data_format_algorithm_frequent_item_sets(verical_data, support)
    result_text.delete(1.0, tk.END)
    for item, transaction_no in frequent_data.items():
        result_text.insert(tk.END, f'{item}: {transaction_no}\n')

def show_strong_item_sets():
    confidence = int(confidence_entry.get())
    global verical_data, frequent_data
    if verical_data == {}:
        print('Please get vertical data first')
        return
    strong_items = vertical_data_format_algorithm_strong_item_sets(verical_data,frequent_data, confidence)
    print("show:")

    result_text.delete(1.0, tk.END)
    for item in strong_items:
        if item != {}:
            for item, confidence in item.items():
                result_text.insert(tk.END, f'{item}: {confidence}\n')

Windows = tk.Tk()
Windows.title('Bakery')

Windows.geometry('500x720')

Windows.configure(bg='#CDCDFB')  

browse_button = ttk.Button(Windows, text='Browse', command=browse_file, style='my.TButton')
browse_button.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

file_entry = ttk.Entry(Windows, font=('Arial', 18))
file_entry.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

title_label_percent = ttk.Label(Windows, text='Percentage', font=('Arial', 18), foreground='#1B0435')  # Set text color
title_label_percent.grid(row=2, column=0, padx=10, pady=10)
validate_precentage = Windows.register(validate_confidence_input)
percentage_entry = ttk.Entry(Windows, font=('Arial', 18), validate='key', validatecommand=(validate_precentage, '%P'))
percentage_entry.grid(row=2, column=1, padx=10, pady=10)

title_label_support = ttk.Label(Windows, text='Support', font=('Arial', 18), foreground='#1B0435')  # Set text color
title_label_support.grid(row=3, column=0, padx=10, pady=10)
support_entry = ttk.Entry(Windows, font=('Arial', 18))
support_entry.grid(row=3, column=1, padx=10, pady=10)

title_label_confidence = ttk.Label(Windows, text='Confidence', font=('Arial', 18), foreground='#1B0435')  # Set text color
title_label_confidence.grid(row=4, column=0, padx=10, pady=10)
confidence_entry = ttk.Entry(Windows, font=('Arial', 18))
confidence_entry.grid(row=4, column=1, padx=10, pady=10)

vertical_button = ttk.Button(Windows, text='Vertical Data', command=show_vertical_data, style='my.TButton')
vertical_button.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

frequent_item_sets_button = ttk.Button(Windows, text='Frequent Item Sets', command=show_frequent_item_sets, style='my.TButton')
frequent_item_sets_button.grid(row=6, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

strong_item_sets_button = ttk.Button(Windows, text='Strong Item Sets', command=show_strong_item_sets, style='my.TButton')
strong_item_sets_button.grid(row=7, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

result_text = tk.Text(Windows, wrap="word", height=18, width=60, foreground='#1B0435', background='#FFFFFF')  
result_text.grid(row=8, column=0, columnspan=2, padx=10, pady=10)

# style for buttons
s = ttk.Style()
s.configure('my.TButton', foreground='#1B0435', background='#E51093')  

Windows.mainloop()