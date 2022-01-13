# concat all qa dataset to one file
import sys
import os
import json


def convert_key(init_key, new_key, obj):
    assert init_key in obj
    tmp = obj.pop(init_key)
    obj[new_key] = tmp
    return obj

if __name__=="__main__":
    qa_dir = sys.argv[1]
    qa_dirs = os.listdir(qa_dir)
    output_dir = os.path.join(qa_dir, 'unified')
    os.makedirs(output_dir, exist_ok=True)
    output_train_file = os.path.join(output_dir, 'train.jsonl')
    output_valid_file = os.path.join(output_dir, 'valid.jsonl')
    output_meta_file = os.path.join(output_dir, 'meta.txt')
    train_file = open(output_train_file, 'w')
    valid_file = open(output_valid_file, 'w')
    meta_file= open(output_meta_file, 'w')
    for dir_name in qa_dirs:
        if dir_name == 'unified':
            continue
        print("Copy {} to unified".format(dir_name))
        current_dataset_dir = os.path.join(qa_dir, dir_name)
        train_file_name = os.path.join(current_dataset_dir, 'train.jsonl')
        valid_file_name = os.path.join(current_dataset_dir, 'valid.jsonl')
        train_cnt = 0
        valid_cnt = 0
        with open(train_file_name, 'r') as f:
            for linenum, line in enumerate(f):
                line = line.strip()
                if line == '':
                    continue
                json_line = json.loads(line)
                if 'answer' in json_line:
                    json_line = convert_key('answer', 'answers', json_line)
                    line = json.dumps(json_line)
                train_file.write(line + '\n')
                train_cnt += 1
        with open(valid_file_name, 'r') as f:
            for linenum, line in enumerate(f):
                line = line.strip()
                if line == '':
                    continue
                json_line = json.loads(line)
                if 'answer' in json_line:
                    json_line = convert_key('answer', 'answers', json_line)
                    line = json.dumps(json_line)
                valid_file.write(line + '\n')
                valid_cnt += 1
        meta_file.write("{}: train {} | valid {}\n".format(dir_name, train_cnt, valid_cnt))
    train_file.close()
    valid_file.close()
    meta_file.close()
