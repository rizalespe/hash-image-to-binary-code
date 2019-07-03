import torch
import numpy as np


binary_code = torch.load("output/output")
file_name =np.load("output/output_filename.npy")

print(file_name)
def hamdist(str1, str2):
    diffs = 0
    for ch1, ch2 in zip(str1, str2):
        if ch1 != ch2:
            diffs += 1
    return diffs

def list_to_string(list):
    string = ""
    for x in range(len(list)):
        string+=str(int(list[x].item()))

    return string



print(hamdist(list_to_string(binary_code[3]),list_to_string(binary_code[0])))
print(hamdist(list_to_string(binary_code[3]),list_to_string(binary_code[1])))
print(hamdist(list_to_string(binary_code[3]),list_to_string(binary_code[2])))
print(hamdist(list_to_string(binary_code[3]),list_to_string(binary_code[3])))
