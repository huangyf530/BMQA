import json
import datasets
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import bs4
from collections import OrderedDict
import time

def narrativeqa():
    dataset = datasets.load_dataset('narrativeqa')
    print(dataset)

def read_data_from_squad(path, split, dataset='squad'):
    with open(path, 'r') as f:
        data = json.load(f)['data']
    cnt = 0
    no_answer_cnt = 0
    new_data = []
    for d in tqdm(data, desc=split):
        for p in d['paragraphs']:
            title = d['title']
            passage = p['context']
            passage = title + ' </s> ' + passage
            passage = passage.replace('\n', ' ')  # replace \n
            for qa in p['qas']:
                question = qa['question']
                answers_set = set()
                for a in qa['answers']:
                    answers_set.add(a['text'])
                answers = list(answers_set)
                if len(answers) == 0:
                    assert qa['is_impossible'] is True
                    answers.append('no answer')
                    no_answer_cnt += 1
                idx = "{}.{:06d}".format(dataset, cnt)
                cnt += 1
                qap = {
                    'question': question,
                    'answers': answers,
                    'passage': passage,
                    'idx': idx,
                }
                new_data.append(qap)
    return new_data, cnt, no_answer_cnt

def squad():
    print(">>> Handle SQuAD2.0 ...")
    dir_path = "/data1/private/hyf/QADataset/SQuAD2.0"
    output_path = "/data1/private/hyf/QADataset/encoded/SQuAD2.0"
    os.makedirs(output_path, exist_ok=True)
    print("Load data from {:s}".format(dir_path))
    train_file = os.path.join(dir_path, 'train-v2.0.json')
    dev_file = os.path.join(dir_path, 'dev-v2.0.json')
    train_data, train_cnt, train_no_answer_cnt = read_data_from_squad(train_file, 'train')
    with open(os.path.join(output_path, 'train.jsonl'), 'w') as f:
        for d in train_data:
            f.write(json.dumps(d) + '\n')
    print("Train: total {:d} QA pairs. {:d} have no answer.".format(train_cnt, train_no_answer_cnt))
    dev_data, dev_cnt, dev_no_answer_cnt = read_data_from_squad(dev_file, 'valid')
    with open(os.path.join(output_path, 'valid.jsonl'), 'w') as f:
        for d in dev_data:
            f.write(json.dumps(d) + '\n')
    print("Valid: total {:d} QA pairs. {:d} have no answer.".format(dev_cnt, dev_no_answer_cnt))
    print("Write encoded SQuAD2.0 to \"{:s}\"".format(output_path))

def squad1_1():
    print(">>> Handle SQuAD1.1 ...")
    dir_path = "/data1/private/hyf/QADataset/SQuAD1.1"
    output_path = "/data1/private/hyf/QADataset/encoded/SQuAD1.1"
    os.makedirs(output_path, exist_ok=True)
    print("Load data from {:s}".format(dir_path))
    train_file = os.path.join(dir_path, 'train-v1.1.json')
    dev_file = os.path.join(dir_path, 'dev-v1.1.json')
    train_data, train_cnt, train_no_answer_cnt = read_data_from_squad(train_file, 'train')
    with open(os.path.join(output_path, 'train.jsonl'), 'w') as f:
        for d in train_data:
            f.write(json.dumps(d) + '\n')
    print("Train: total {:d} QA pairs. {:d} have no answer.".format(train_cnt, train_no_answer_cnt))
    dev_data, dev_cnt, dev_no_answer_cnt = read_data_from_squad(dev_file, 'valid')
    with open(os.path.join(output_path, 'valid.jsonl'), 'w') as f:
        for d in dev_data:
            f.write(json.dumps(d) + '\n')
    print("Valid: total {:d} QA pairs. {:d} have no answer.".format(dev_cnt, dev_no_answer_cnt))
    print("Write encoded SQuAD1.1 to \"{:s}\"".format(output_path))


def extract_span(tokens, annotation):
    start_index = annotation['start_token']
    end_index = annotation['end_token']
    return tokens[start_index:end_index]

def clean_text(text):
    text = text.replace(" ,", ",").replace(" .", ".").replace(" %", "%")
    text = text.replace(" - ", "-").replace(" : ", ":").replace(" / ", "/")
    text = text.replace("( ", "(").replace(" )", ")")
    text = text.replace("`` ", "\"").replace(" ''", "\"")
    text = text.replace(" 's", "'s").replace("s ' ", "s' ")
    return text.strip()

def convert_tokens_to_text(tokens: str):
    text = ' '.join(tokens)
    text = bs4.BeautifulSoup(text, "lxml").text
    return clean_text(text)

def nqans():
    print(">>> Handle NaturalQuestions ...")
    dir_path = "/data1/private/hyf/QADataset/NaturalQuestions"
    output_path = "/data1/private/hyf/QADataset/encoded/nqans"
    os.makedirs(output_path, exist_ok=True)
    print("Load data from {:s}".format(dir_path))
    train_file_name = "v1.0-simplified-simplified-nq-train.jsonl"
    dev_file_name = "v1.0-simplified-nq-dev-all.jsonl"
    for file_name, split in zip([train_file_name, dev_file_name], ['train', 'valid']):
        output_file_path = os.path.join(output_path, f"{split}.jsonl")
        output_file = open(output_file_path, 'w')
        candidates_cnt_list = []
        cnt = 0
        question_cnt = 0
        no_title_cnt = 0
        with open(os.path.join(dir_path, file_name)) as f:
            start_time = time.time()
            for linenum, line in enumerate(f):
                line = line.strip()
                if line == '':
                    continue
                line_json = json.loads(line)
                # Handle the two document formats in NQ (tokens or text).
                if "document_tokens" in line_json:
                    tokens = [t["token"] for t in line_json["document_tokens"]]
                elif "document_text" in line_json:
                    tokens = line_json["document_text"].split(" ")
                document_text = ' '.join(tokens)
                title = clean_text(bs4.BeautifulSoup(document_text, "lxml").h1.text)
                if title == '':
                    no_title_cnt += 1
                question = line_json['question_text'] + '?'
                long_answer_candidates = line_json['long_answer_candidates']
                annotations = line_json['annotations']
                answer_annotations = OrderedDict()
                for a in annotations:
                    if len(a['short_answers']) == 0:
                        # remove no short answer question
                        continue
                    long_answer_index = a['long_answer']['candidate_index']
                    if long_answer_index in answer_annotations:
                        # already in annotation
                        continue
                    long_answer_token = extract_span(tokens, a['long_answer'])
                    if long_answer_token[0] == '<Table>':  # remove table answer
                        continue
                    answers = []
                    for sa in a['short_answers']:
                        sa_tokens = extract_span(tokens, sa)
                        answers.append(convert_tokens_to_text(sa_tokens))
                    answer_annotations[long_answer_index] = answers
                if len(answer_annotations) == 0:
                    # no short answer
                    continue
                candidates_cnt = 0
                for index, c in enumerate(long_answer_candidates):
                    c_token = extract_span(tokens, c)
                    if c_token[0] == '<Table>':
                        # remove table cadidate
                        continue
                    if index not in answer_annotations and len(c_token) <= 20:
                        # remove too short candidate, 20 is a magic number
                        continue
                    if index not in answer_annotations:
                        # only context have answer will keep
                        continue
                    if title != '':
                        passage = title +' </s> ' + convert_tokens_to_text(c_token)
                    else:
                        passage = convert_tokens_to_text(c_token)
                    # print(f"candidate {cnt}:", extract_span(tokens, c))
                    if index in answer_annotations:
                        current_data = {
                            'question': question,
                            'answers': answer_annotations[index],
                            'passage': passage,
                            'question_idx': 'nq.{:06d}'.format(question_cnt),
                            'idx': 'nq.{:07d}'.format(cnt)
                        }
                    else:
                        current_data = {
                            'question': question,
                            'answers': ['no answer'],
                            'passage': passage,
                            'question_idx': 'nq.{:06d}'.format(question_cnt),
                            'idx': 'nq.{:07d}'.format(cnt)
                        }
                    candidates_cnt += 1
                    output_file.write(json.dumps(current_data) + '\n')
                    cnt += 1
                candidates_cnt_list.append(candidates_cnt)
                question_cnt += 1
                if question_cnt % 1000 == 0:
                    print("{}: Handle {:d} questions. No title cnt: {}.Use time {:.3f}s".format(linenum, question_cnt, no_title_cnt, time.time() - start_time))
        output_file.close()
        print("Write data to {:s}. Time: {:.3f}s".format(output_file_path, time.time() - start_time))
        print("Passage with no title: {}".format(no_title_cnt))
        print("{:s}: Question {:d} | Passages: {:d} | Average passage number: {:.2f}".format(split, question_cnt, cnt, np.mean(candidates_cnt_list)))

def quoref():
    print(">>> Handle QuoRef ...")
    dir_path = "/data1/private/hyf/QADataset/quoref-train-dev-v0.2"
    output_path = "/data1/private/hyf/QADataset/encoded/QuoRef"
    os.makedirs(output_path, exist_ok=True)
    print("Load data from {:s}".format(dir_path))
    train_file = os.path.join(dir_path, 'quoref-train-v0.2.json')
    dev_file = os.path.join(dir_path, 'quoref-dev-v0.2.json')
    train_data, train_cnt, train_no_answer_cnt = read_data_from_squad(train_file, 'train', 'quoref')
    with open(os.path.join(output_path, 'train.jsonl'), 'w') as f:
        for d in train_data:
            f.write(json.dumps(d) + '\n')
    print("Train: total {:d} QA pairs. {:d} have no answer.".format(train_cnt, train_no_answer_cnt))
    dev_data, dev_cnt, dev_no_answer_cnt = read_data_from_squad(dev_file, 'valid', 'quoref')
    with open(os.path.join(output_path, 'valid.jsonl'), 'w') as f:
        for d in dev_data:
            f.write(json.dumps(d) + '\n')
    print("Valid: total {:d} QA pairs. {:d} have no answer.".format(dev_cnt, dev_no_answer_cnt))
    print("Write encoded QuoRef to \"{:s}\"".format(output_path))
    
def newsqa():
    print(">>> Handle NewsQA ...")
    dir_path = "/data1/private/hyf/QADataset/NewsQA/newsqa"
    output_path = "/data1/private/hyf/QADataset/encoded/NewsQA"
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(dir_path, 'combined-newsqa-data-v1.json')) as f:
        data = json.load(f)['data']
    count = {'train': 0, 'dev': 0, 'test': 0}
    no_answer_count = {'train': 0, 'dev': 0, 'test': 0}
    new_datas = {'train': [], 'dev': [], 'test': []}
    for index, d in enumerate(tqdm(data, desc='newsqa')):
        passage = d['text']
        new_passage = ' '.join(filter(lambda p: p != '', passage.split('\n')))
        its_type = d['type']
        for q in d['questions']:
            # print(json.dumps(q, indent=4))
            question = q['q']
            if 'badQuestion' in q['consensus']:
                continue
            elif 'noAnswer' in q['consensus']:
                answer_text = 'no answer'
                no_answer_count[its_type] += 1
            else:
                answer_text = passage[q['consensus']['s']:q['consensus']['e']].strip()
            if answer_text[-1] == '.' or answer_text[-1] == ',':
                answer_text = answer_text[:-1]
            answers = [answer_text]
            new_data = {
                'passage': new_passage,
                'question': question,
                'answers': answers,
                'idx': "newsqa.{:06d}".format(count[its_type])
            }
            count[its_type] += 1
            new_datas[its_type].append(new_data)
    print("Data counts: Train {train} | Valid {dev} | Test {test}".format(**count))
    print("No answer data counts: Train {train} | Valid {dev} | Test {test}".format(**no_answer_count))
    with open(os.path.join(output_path, 'train.jsonl'), 'w') as f:
        for d in new_datas['train']:
            f.write(json.dumps(d) + '\n')
    with open(os.path.join(output_path, 'valid.jsonl'), 'w') as f:
        for d in new_datas['dev']:
            f.write(json.dumps(d) + '\n')
    with open(os.path.join(output_path, 'test.jsonl'), 'w') as f:
        for d in new_datas['test']:
            f.write(json.dumps(d) + '\n')

def drop():
    print(">>> Handle DROP ...")
    data_path = "/home/huangyufei/data1/QADataset/drop"
    train_file_name = os.path.join(data_path, 'drop_dataset_train.json')
    dev_file_name = os.path.join(data_path, 'drop_dataset_dev.json')
    output_path = "/data1/private/hyf/QADataset/encoded/drop"
    os.makedirs(output_path, exist_ok=True)
    print("Load data from {:s}".format(data_path))
    def get_answer_from_dict(answer_dict, question):
        number_ans = answer_dict['number']
        answers_set = set()
        if number_ans != '':
            answers_set.add(number_ans)
        if len(answer_dict['spans']) > 0:
            answers_set.update(answer_dict['spans'])
        date_ans = answer_dict['date']
        if len(date_ans) > 0:
            date_ans_list = []
            for key in ['day', 'month', 'year']:
                if date_ans[key] != '':
                    date_ans_list.append(date_ans[key])
            if len(date_ans_list) > 0:
                answers_set.add(' '.join(date_ans_list))
        return list(answers_set)
    for filename, split in zip([train_file_name, dev_file_name], ['train', 'valid']):
        cnt = 0
        no_answer_cnt = 0
        new_datas = []
        with open(filename) as f:
            data = json.load(f)
            for key in data:
                d = data[key]
                passage = d['passage']
                for qa in d['qa_pairs']:
                    question = qa['question']
                    if 'validated_answers' in qa and len(qa['validated_answers']) > 0:
                        answers_set = set()
                        for a in qa['validated_answers']:
                            answers = get_answer_from_dict(a, question)
                            answers_set.update(answers)
                        answers = list(answers_set)
                    else:
                        answers = get_answer_from_dict(qa['answer'], question)
                    if len(answers) == 0:
                        answers.append('no answer')
                        no_answer_cnt += 1
                        print(json.dumps(qa, indent=4))
                    new_data = {
                        'passage': passage,
                        'question': question,
                        'answers': answers,
                        'idx': 'drop.{:06d}'.format(cnt)
                    }
                    cnt += 1
                    new_datas.append(new_data)
        print("{}: total {:d} QA pairs. {:d} have no answer.".format(split, cnt, no_answer_cnt))
        with open(os.path.join(output_path, '{}.jsonl'.format(split)), 'w') as f:
            for d in new_datas:
                f.write(json.dumps(d) + '\n')
        print("Write {} data to {}.".format(split, output_path))
                    
                    

if __name__ == "__main__":
    squad()
    # nqans()
    quoref()
    # newsqa()
    # drop()
    squad1_1()
