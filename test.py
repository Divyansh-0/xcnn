def min_operations_to_organize(items):
    operations = 0
    n = len(items)
    
    for i in range(n - 1):
        while items[i] % 2 == items[i + 1] % 2 and items[i + 1] != 0:
            items[i + 1] //= 2
            operations += 1
                
    return operations





# Example
items = [6, 5, 9, 7, 3]
result = min_operations_to_organize(items)
print("Minimum number of operations:", result)
