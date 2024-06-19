import os
import sys

#load all files in this folder recursively
#change the word "goere" to "eereid"
#save the file again
#ignore the current file


def rename_all_files_in_folder(folder_path, old_word, new_word):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py") and file != os.path.basename(__file__):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                new_content = content.replace(old_word, new_word)
                with open(file_path, 'w') as f:
                    f.write(new_content)

def search_for_string(folder_path,key_word):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py") and file != os.path.basename(__file__):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                    if key_word in content:
                        print(file_path)

if __name__ == "__main__":
    folder_path = os.path.dirname(__file__)
    #old_word = "goere"
    #new_word = "eereid"
    #rename_all_files_in_folder(folder_path, old_word, new_word)
    key_word = "cache"
    search_for_string(folder_path,key_word)


