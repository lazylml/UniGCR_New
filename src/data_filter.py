import torch
import os

min_len = 5

# 1. 读取原始数据
path = '../data/GRID/outputs/beauty_user_item_sequences'

path_train = os.path.join(path, "beauty_user_item_sequences_training.pt")
path_val = os.path.join(path, "beauty_user_item_sequences_evaluation.pt")
path_test = os.path.join(path, "beauty_user_item_sequences_testing.pt")

train_user2seq = torch.load(path_train, map_location="cpu")
val_user2seq = torch.load(path_val, map_location="cpu")
test_user2seq = torch.load(path_test, map_location="cpu")

print("original:")
print("train users:", len(train_user2seq))
print("val users:", len(val_user2seq))
print("test users:", len(test_user2seq))

# 2. 先过滤 train 中长度不足 min_len 的 user
filtered_train = {u: seq for u, seq in train_user2seq.items() if len(seq) >= min_len}

# 3. 保留 train 中出现的 user
kept_users = set(filtered_train.keys())

# 4. 用同样的 user 集合过滤 val / test
filtered_val = {u: seq for u, seq in val_user2seq.items() if u in kept_users}
filtered_test = {u: seq for u, seq in test_user2seq.items() if u in kept_users}

print("\nafter filtering:")
print("train users:", len(filtered_train))
print("val users:", len(filtered_val))
print("test users:", len(filtered_test))

# 5. 保存
path_out = '../data/GRID/outputs/beauty_user_item_sequences_core5'
path_train_out = os.path.join(path_out, "beauty_user_item_sequences_training_core5.pt")
path_val_out = os.path.join(path_out, "beauty_user_item_sequences_evaluation_core5.pt")
path_test_out = os.path.join(path_out, "beauty_user_item_sequences_testing_core5.pt")

torch.save(filtered_train, path_train_out)
torch.save(filtered_val, path_val_out)
torch.save(filtered_test, path_test_out)

print("\nSaved done.")
