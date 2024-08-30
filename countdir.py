import os

def count_items_in_subdirs(parent_dir):
    lst = []
    for subdir, _, files in os.walk(parent_dir):
        if subdir != parent_dir:  # Exclude the parent directory itself
            num_items = len(files)
            print(f"Directory: {subdir} contains {num_items} items")
            lst.append(num_items)
    return lst

# Example usage:
parent_directory = "snippets"
print(sorted(count_items_in_subdirs(parent_directory)))
