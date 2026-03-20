import torch
import os

def get_item_count(dt):
    unique_items = set()
    for seq in dt.values():
        unique_items.update(seq)
    num_unique_items = len(unique_items)
    return num_unique_items

def get_data_len(dt):
    lens = [len(seq) for u, seq in dt.items()]
    return lens

def data_stats(data_train, data_val, data_test, desp):
    user_list_train, user_list_val, user_list_test = [u for u, seq in data_train.items()], [u for u, seq in data_val.items()], [u for u, seq in data_test.items()]
    user_set_train, user_set_val, user_set_test = set(user_list_train), set(user_list_val), set(user_list_test)
    users_train, users_val, users_test = len(data_train), len(data_val), len(data_test)
    items_train, items_val, items_test = get_item_count(data_train), get_item_count(data_val), get_item_count(data_test)
    lens_train, lens_val, lens_test = get_data_len(data_train), get_data_len(data_val), get_data_len(data_test)
    interactions_train, interactions_val, interactions_test = sum(lens_train), sum(lens_val), sum(lens_test)
    avg_len_train, avg_len_val, avg_len_test = round(interactions_train / users_train, 1), round(interactions_val / users_val, 1), round(interactions_test / users_test, 1)

    assert user_list_train == user_list_val == user_list_test, "user list not equal"
    assert user_set_train == user_set_val == user_set_test, "user set not equal"
    if user_list_train == user_list_val == user_list_test:
        print('train, val, test contain same users.')

    print(f"{desp}:\n"
          f"#Users, train: {users_train}, val: {users_val}, test: {users_test}\n"
          f"#Unique Users, train: {len(user_set_train)}, val: {len(user_set_val)}, test: {len(user_set_test)}\n"
          f"#Items, train: {items_train}, val: {items_val}, test: {items_test}\n"
          f"#Interactions, train: {interactions_train}, val: {interactions_val}, test: {interactions_test}\n"
          f"#Avg length, train: {avg_len_train}, val: {avg_len_val}, test: {avg_len_test}\n"
          f"#Min length, train: {min(lens_train)}, val: {min(lens_val)}, test: {min(lens_test)}\n"
          f"#Max length, train: {max(lens_train)}, val: {max(lens_val)}, test: {max(lens_test)}\n"+'-'*50)


# 1. 读取原始数据
print('toys data:')
path = '../data/GRID/outputs/toys_user_item_sequences'

path_train = os.path.join(path, "toys_user_item_sequences_training.pt")
path_val = os.path.join(path, "toys_user_item_sequences_evaluation.pt")
path_test = os.path.join(path, "toys_user_item_sequences_testing.pt")

user2seq_train = torch.load(path_train, map_location="cpu")
user2seq_val = torch.load(path_val, map_location="cpu")
user2seq_test = torch.load(path_test, map_location="cpu")


# 2. 统计信息 #users, #items, #interactions, avg length
data_stats(user2seq_train, user2seq_val, user2seq_test, 'original version')

# 3. 找出问题序列
question_users = []
for u, seq_train in user2seq_train.items():
    seq_val = user2seq_val[u]
    seq_test = user2seq_test[u]
    if not (seq_train == seq_test[:-2] and seq_val == seq_test[:-1]):
        question_users.append(u)

uid_example = question_users[10]
print(f'Number of question users (with incomplete sequences): {len(question_users)}')
print(f'Example of question user ID: {uid_example}\ntrain use sequence: {user2seq_train[uid_example]}\nval sequence: {user2seq_val[uid_example]}\ntest sequence: {user2seq_test[uid_example]}')

# 4. 恢复val, test中的不完整序列
full_user2seq_train = {u: seq for u, seq in user2seq_train.items()}
full_user2seq_val = {u: seq for u, seq in user2seq_val.items()}
full_user2seq_test = {u: seq for u, seq in user2seq_test.items()}
for uid in question_users:
    seq_train = user2seq_train[uid]
    seq_val = user2seq_val[uid]
    seq_test = user2seq_test[uid]

    full_seq_train = seq_train
    full_seq_val = seq_train + seq_val[-1:]
    full_seq_test = seq_train + seq_test[-2:]

    full_user2seq_train[uid] = full_seq_train
    full_user2seq_val[uid] = full_seq_val
    full_user2seq_test[uid] = full_seq_test

print('-'*10+'After sequence recovery:'+'-'*10)
print(f'Example of question user ID: {uid_example}\ntrain use sequence: {full_user2seq_train[uid_example]}\nval sequence: {full_user2seq_val[uid_example]}\ntest sequence: {full_user2seq_test[uid_example]}')
data_stats(full_user2seq_train, full_user2seq_val, full_user2seq_test, 'After sequence recovery')

# 5. 截断，适应unigcr的max_seq_len = 50
max_seq_len = 50-1
# 只截断val, test, 因为train不涉及beam search
truncate_user2seq_train = {u: seq for u, seq in full_user2seq_train.items()}
truncate_user2seq_val = {u: seq[-max_seq_len:] for u, seq in full_user2seq_val.items()}
truncate_user2seq_test = {u: seq[-max_seq_len:] for u, seq in full_user2seq_test.items()}
print('-'*10+'After truncation (max_seq_len = 49):'+'-'*10)
print(f'Example of question user ID: {uid_example}\ntrain use sequence: {truncate_user2seq_train[uid_example]}\nval sequence: {truncate_user2seq_val[uid_example]}\ntest sequence: {truncate_user2seq_test[uid_example]}')
data_stats(truncate_user2seq_train, truncate_user2seq_val, truncate_user2seq_test, 'After sequence recovery and truncation')

# 6. 保存
full_path_out = '../data/GRID/outputs/toys_user_item_sequences_full'
os.makedirs(full_path_out, exist_ok=True)
path_train_out = os.path.join(full_path_out, "toys_user_item_sequences_training_full.pt")
path_val_out = os.path.join(full_path_out, "toys_user_item_sequences_evaluation_full.pt")
path_test_out = os.path.join(full_path_out, "toys_user_item_sequences_testing_full.pt")
torch.save(full_user2seq_train, path_train_out)
torch.save(full_user2seq_val, path_val_out)
torch.save(full_user2seq_test, path_test_out)

truncate_path_out = '../data/GRID/outputs/toys_user_item_sequences_truncate49'
os.makedirs(truncate_path_out, exist_ok=True)
path_train_out = os.path.join(truncate_path_out, "toys_user_item_sequences_training_truncate49.pt")
path_val_out = os.path.join(truncate_path_out, "toys_user_item_sequences_evaluation_truncate49.pt")
path_test_out = os.path.join(truncate_path_out, "toys_user_item_sequences_testing_truncate49.pt")
torch.save(truncate_user2seq_train, path_train_out)
torch.save(truncate_user2seq_val, path_val_out)
torch.save(truncate_user2seq_test, path_test_out)

print("\nSaved done.")
