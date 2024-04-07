import csv

def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
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


# function to check if number of Items in transaction_items > min_support and return only those transactions
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
            print("-------------------sssssssssssss-----------------")
            print(f'{new_item}: {new_transaction_no}')
    if all(len(transaction_no) < min_support for transaction_no in new_items.values()):
        return {}
    else:
        return new_items

# function find strong association rules between items in transactions by check if min confidend > number of transactions containing both items(x,y) / number of transactions containing item(x)
# def strong_item_sets(min_support_transaction_items, transaction_items, min_confidence):
    


# Example usage:
file_path ='bakery/BakeryData.csv'
csv_data = read_csv_file(file_path)
transaction_items = collect_items_by_transaction(csv_data)
vertical_items = vertical_format(transaction_items)
# print all transaction items from vertical_format
print('Vertical Items')
for item, transaction_no in list(vertical_items.items())[:]:
    print(f'{item}: {transaction_no}')


# # print first 5 frequent items from frequent_item_sets
# min_support = 2
# frequent_items = frequent_item_sets(vertical_items, min_support)
# print('Frequent Items final')
# for item, transaction_no in list(frequent_items.items())[:]:
#     print(f'{item}: {transaction_no}')
min_support = 1
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

print('Frequent Items final')
for item, transaction_no in list(frequent_items_final.items())[:]:
    print(f'{item}: {transaction_no}')
print('Strong Association Rules')
