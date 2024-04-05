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

# function to check if number of Items in transaction_items > min_support and return only those transactions
def min_support_transactions(transaction_items, min_support):
    min_support_transaction_items = {}
    for transaction_no, items in transaction_items.items():
        if len(items) >= min_support:
            min_support_transaction_items[transaction_no] = items
    return min_support_transaction_items

# function find strong association rules between items in transactions by check if min confidend > number of transactions containing both items(x,y) / number of transactions containing item(x)
def find_association_rules(min_support_transaction_items, transaction_items, min_confidence):
    association_rules = {}
    for transaction_no, items in min_support_transaction_items.items():
        for item in items:
            for other_item in items:
                if item != other_item:
                    print('------------------------------------')
                    print(f'Item: {item}, Other Item: {other_item}')
                    print('------------------------------------')

                    # item_support = len(transaction_items[transaction_no])
                    # both_items_support = len(min_support_transaction_items[transaction_no])
                    # confidence = both_items_support / item_support
                    item_support = 0
                    for transaction_no, items in transaction_items.items():
                        if item in items:
                            item_support += 1
                    both_items_support = 0
                    for transaction_no, items in transaction_items.items():
                        if item in items and other_item in items:
                            both_items_support += 1
                    confidence = both_items_support / item_support
                    if confidence >= min_confidence:
                        if item in association_rules:
                            association_rules[item].add(other_item)
                        else:
                            association_rules[item] = {other_item}
    return association_rules

# function aprori algorithm to print frequent itemsets and association rules with ther gonfidence
def apriori_algorithm(transaction_items, min_support, min_confidence):
    min_support_transaction_items = min_support_transactions(transaction_items, min_support)
    association_rules = find_association_rules(min_support_transaction_items, transaction_items, min_confidence)
    print('Frequent Itemsets:')
    for transaction_no, items in min_support_transaction_items.items():
        print(f'Transaction No: {transaction_no}, Items: {items}')
    print('Association Rules:')
    for item, other_items in association_rules.items():
        for other_item in other_items:
            print(f'{item} -> {other_item}')


# Example usage:
file_path ='bakery/BakeryData.csv'
csv_data = read_csv_file(file_path)
transaction_items = collect_items_by_transaction(csv_data)

apriori_algorithm(transaction_items, min_support=3, min_confidence=1)

# # print the first 5 transactions
# for transaction_no, items in list(transaction_items.items())[:9]:
#     print(f'Transaction No: {transaction_no}, Items: {items}')
# print('------------------------------------')
# min_support = 3
# min_support_transaction_items = min_support_transactions(transaction_items, min_support)
# # print the first 5 transactions with min_support
# for transaction_no, items in list(min_support_transaction_items.items())[:9]:
#     print(f'Transaction No: {transaction_no}, Items: {items}')
