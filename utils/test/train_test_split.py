from sklearn.model_selection import train_test_split
import os


data_name_list = os.listdir('/data/zxs/data/ecg_tiger/interval_json_done_1000')

print(data_name_list)

train_list, test_list = train_test_split(data_name_list, test_size=0.2, shuffle=True)

print(train_list)
print(test_list)

with open('/data/yhy/project/ecg_generation/dataset/tianchi_1000_reanno_train_jsons.txt', 'w') as f:
    for data_name in train_list:
        f.write(data_name + '\n')

with open('/data/yhy/project/ecg_generation/dataset/tianchi_1000_reanno_test_jsons.txt', 'w') as f:
    for data_name in test_list:
        f.write(data_name + '\n')
