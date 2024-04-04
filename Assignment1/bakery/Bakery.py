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

# write a function to check if number of Items in transaction_items > min_support and return only those transactions
def min_support_transactions(transaction_items, min_support):
    min_support_transaction_items = {}
    for transaction_no, items in transaction_items.items():
        if len(items) >= min_support:
            min_support_transaction_items[transaction_no] = items
    return min_support_transaction_items

# Example usage:
file_path ='bakery/BakeryData.csv'
csv_data = read_csv_file(file_path)
transaction_items = collect_items_by_transaction(csv_data)
# print the first 5 transactions
for transaction_no, items in list(transaction_items.items())[:9]:
    print(f'Transaction No: {transaction_no}, Items: {items}')
print('------------------------------------')
min_support = 3
min_support_transaction_items = min_support_transactions(transaction_items, min_support)
# print the first 5 transactions with min_support
for transaction_no, items in list(min_support_transaction_items.items())[:9]:
    print(f'Transaction No: {transaction_no}, Items: {items}')
