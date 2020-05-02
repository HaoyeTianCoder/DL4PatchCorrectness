import glob, os
import io
import re
import numpy as np
import pandas as pd
import json
import datetime

import common
import pickle
import random

import subprocess

EXTRACTOR_JAR = '/Users/abdoulkader.kabore/snt/code2vec/JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar'
MAX_PATH_LENGTH = 8 
MAX_PATH_WIDTH = 2
TMP_FOLDER = 'tmp'

output_file_path = '../data/train_data5_frag_code2vec.txt'

def get_deep_files(dir_path, extensions=['.txt', '.patch']):
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            _, extension = os.path.splitext(f)
            if extension in extensions:
                yield root, os.path.join(root, f)

def create_train_data5_frag(path_patch_train):
    with open(output_file_path, 'w+') as f:
        data = ''
        for root, dirs, files in os.walk(path_patch_train):
            if files == ['.DS_Store']:
                continue
            for file in files:
                if file.endswith('txt') or file.endswith('patch'):
                    if file.endswith('.patch'):
                        bug_id = '_'.join([root.split('/')[-2],root.split('/')[-1], file])
                    else:
                        bug_id = '_'.join([root.split('/')[-1],file])
                    try:
                        buggy = get_diff_files(os.path.join(root,file),type='buggy')
                        patched = get_diff_files(os.path.join(root, file), type='patched')
                    except Exception as e:
                        print(e)
                        continue
                    label_temp = '1'
                    sample = label_temp + '<ml>' + bug_id + '<ml>' + buggy + '<ml>' + patched
                    data += sample + '\n'
                    
        f.write(data)

def get_diff_files(file_path, type, withFile=True):
    if withFile:
        file = open(file_path, 'r')
    else:
        file = file_path

    lines = ''
    p = r"([^\w_|@|$|:])"
    flag = True
        
    for line in file:
        line = line.strip()
        if '*/' in line:
            flag = True
            continue
        if flag == False:
            continue
        if line != '':
            if line.startswith('@@') or line.startswith('diff') or line.startswith('index') or line.startswith('---') or line.startswith('+++'):
                continue
            elif '/*' in line:
                flag = False
                continue
            elif type == 'buggy':
                if line.startswith('-'):
                    if line[1:].strip().startswith('//'):
                        continue
                    lines += line[1:] + ' '
                elif line.startswith('+'):
                    # do nothing
                    pass
                else:
                    lines += line + ' '
            elif type == 'patched':
                if line.startswith('+'):
                    if line[1:].strip().startswith('//'):
                        continue
                    lines += line[1:] + ' '
                elif line.startswith('-'):
                    # do nothing
                    pass
                else:
                    lines += line + ' '
   
    if withFile:
        file.close()

    return lines

def process_file(file_path, data_file_role, dataset_name, word_to_count, path_to_count, max_contexts):
    print(file_path)
    sum_total = 0
    sum_sampled = 0
    total = 0
    empty = 0
    max_unfiltered = 0
    output_path = '{}.{}.c2v'.format(dataset_name, data_file_role)
    with open(output_path, 'w') as outfile:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.rstrip('\n').split(' ')
                target_name = parts[0]
                contexts = parts[1:]

                if len(contexts) > max_unfiltered:
                    max_unfiltered = len(contexts)
                sum_total += len(contexts)

                if len(contexts) > max_contexts:
                    context_parts = [c.split(',') for c in contexts]
                    full_found_contexts = [c for i, c in enumerate(contexts)
                                           if context_full_found(context_parts[i], word_to_count, path_to_count)]
                    partial_found_contexts = [c for i, c in enumerate(contexts)
                                              if context_partial_found(context_parts[i], word_to_count, path_to_count)
                                              and not context_full_found(context_parts[i], word_to_count,
                                                                         path_to_count)]
                    if len(full_found_contexts) > max_contexts:
                        contexts = random.sample(full_found_contexts, max_contexts)
                    elif len(full_found_contexts) <= max_contexts \
                            and len(full_found_contexts) + len(partial_found_contexts) > max_contexts:
                        contexts = full_found_contexts + \
                                   random.sample(partial_found_contexts, max_contexts - len(full_found_contexts))
                    else:
                        contexts = full_found_contexts + partial_found_contexts

                if len(contexts) == 0:
                    empty += 1
                    continue

                sum_sampled += len(contexts)

                csv_padding = " " * (max_contexts - len(contexts))
                outfile.write(target_name + ' ' + " ".join(contexts) + csv_padding + '\n')
                total += 1

    print('File: ' + data_file_path)
    print('Average total contexts: ' + str(float(sum_total) / total))
    print('Average final (after sampling) contexts: ' + str(float(sum_sampled) / total))
    print('Total examples: ' + str(total))
    print('Empty examples: ' + str(empty))
    print('Max number of contexts per word: ' + str(max_unfiltered))
    return total

def context_full_found(context_parts, word_to_count, path_to_count):
    return context_parts[0] in word_to_count \
           and context_parts[1] in path_to_count and context_parts[2] in word_to_count


def context_partial_found(context_parts, word_to_count, path_to_count):
    return context_parts[0] in word_to_count \
           or context_parts[1] in path_to_count or context_parts[2] in word_to_count

def process(path, f, type, id_path):
    if not os.path.exists(id_path):
        changes = get_diff_files(f, type=type)
        # print('Source Code')
        # print(changes)
        # print('\n\n')
        file_path = path + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f") + '_PATCH.java'
        writer = io.open(file_path, 'w', encoding='utf-8')
        writer.write(changes)
        writer.close()

        command = 'java -cp ' + EXTRACTOR_JAR + ' JavaExtractor.App --max_path_length ' + \
                    str(MAX_PATH_LENGTH) + ' --max_path_width ' + str(MAX_PATH_WIDTH) + ' --file ' + file_path + ' > ' + id_path
        try:
            subprocess.check_call(command, shell=True, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError:
            return None
        
        os.remove(file_path)

    lines = []
    with open(id_path, 'r') as ff:
        for line in ff:
            lines.append(line)
    return lines

def retrieve_identifiers(input):
    data = []
    for line in input:
        parts = line.rstrip('\n').split(' ')
        target_name = parts[0]
        contexts = parts[1:]
        data.append(target_name)
        for c in contexts:
            tmp = c.split(',')
            del tmp[1]
            data.extend(tmp)

    return unique_list(data)

def unique_list(l):
  x = []
  for a in l:
    if a not in x:
      x.append(a)
  return x


if __name__ == '__main__':
    dir_path = '../data/code2vec_train_data/Patches_train'
    # dir_path = '.'
    n = -1
    i = 0
    j = 0
    data = ''
    for root, f in get_deep_files(dir_path=dir_path):
        head_tail = os.path.split(f) 
        tmp = head_tail[0].split('/')
        path = TMP_FOLDER + '/' + tmp[n]
        try:
            os.makedirs(path)
        except OSError:
            pass
        else:
            pass
            
        try:

            if f.endswith('.patch'):
                bug_id = '_'.join([root.split('/')[-2], root.split('/')[-1], head_tail[1]])
            else:
                bug_id = '_'.join([root.split('/')[-1], head_tail[1]])

            buggy_id_path = path + bug_id + '_BUG.id'
            patch_id_path = path + bug_id + '_PATCH.id'

            b_output = process(path, f, 'buggy', buggy_id_path)
            if b_output is not None and len(b_output) > 0:
                p_output = process(path, f, 'patched', patch_id_path)

                buggy_tokens = retrieve_identifiers(b_output)
                patch_tokens = retrieve_identifiers(p_output)

                if len(buggy_tokens) > 0 and len(patch_tokens) > 0:
                    label_temp = '1'
                    sample = label_temp + '<ml>' + bug_id + '<ml>' + ' '.join(buggy_tokens) + '<ml>' + ' '.join(patch_tokens)
                    data += sample + '\n'
                    i = i + 1
        except:
            j = j + 1

    out_writer = io.open(output_file_path, 'w', encoding='utf-8')
    out_writer.write(data)

    print(i)
    print(j)