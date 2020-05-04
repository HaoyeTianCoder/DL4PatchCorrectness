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

benchmarks = ["Bears", "Bugs.jar", "Defects4J", "IntroClassJava","QuixBugs"]
tools = ["Arja", "GenProg", "Kali", "RSRepair", "Cardumen", "jGenProg", "jKali", "jMutRepair", "Nopol", "DynaMoth", "NPEFix"]

def get_deep_files(dir_path, extensions=['.txt', '.patch'], paired=False):
    if paired:
        os.chdir(dir_path)
        buggy_files = glob.iglob(dir_path + '/**/*-buggyCode.txt', recursive=True)
        for buggy_file in buggy_files:
            tmp = buggy_file.partition("-")
            patch_file = tmp[0] + '-' + tmp[2].split('-')[0] + "-patchcode.txt"
            yield buggy_file, patch_file
    else:
        for root, dirs, files in os.walk(dir_path):
            for f in files:
                _, extension = os.path.splitext(f)
                if extension in extensions:
                    yield root, os.path.join(root, f)
    

def get_buggy_patch(path_patch_test):
    for benchmark in sorted(benchmarks):
        benchmark_path = os.path.join(path_patch_test, benchmark)
        for project in sorted(os.listdir(benchmark_path)):
            if project.startswith('.'):
                continue
            project_path = os.path.join(benchmark_path, project)
            folders = os.listdir(project_path)
            if benchmark == "QuixBugs":
                folders = [""]
            for id in sorted(folders):
                if id.startswith('.'):
                    continue
                bug_path = os.path.join(project_path, id)
                for repair_tool in sorted(os.listdir(bug_path)):
                    if repair_tool not in tools:
                        continue
                    tool_path = os.path.join(bug_path, repair_tool)
                    if not os.path.isdir(tool_path):
                        continue
                    for seed in sorted(os.listdir(tool_path)):
                        if type(seed).__name__ == 'list' and len(seed) > 1:
                            print('warning...')
                        seed_path = os.path.join(tool_path, seed)
                        results_path = os.path.join(seed_path, "result.json")
                        if os.path.exists(results_path):
                            with open(results_path, 'r') as f1:
                                patch_dict = json.load(f1)
                            patches = patch_dict['patches']
                            if patches != []:
                                for p in patches:
                                    patch = []
                                    if 'patch' in p:
                                        patch = p['patch']
                                        p_list = patch.split('\n')
                                    elif 'PATCH_DIFF_ORIG' in p:
                                        patch = p['PATCH_DIFF_ORIG']
                                        p_list = patch.split('\\n')
                                    
                                    buggy = get_diff_files(p_list, type='buggy', withFile=False)
                                    patched = get_diff_files(p_list, type='patched', withFile=False)
                                    yield results_path, buggy, patched, patch, seed, id, benchmark, repair_tool, project

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

def process(path, f, type, id_path, pathIsOk=False, fIsChanged=False):
    if not os.path.exists(id_path):
        if fIsChanged:
            changes = f
        else:
            changes = get_diff_files(f, type=type)
        # print('Source Code')
        # print(changes)
        # print('\n\n')

        print(path)


        if pathIsOk:
            file_path = f
        else:
            file_path = path + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f") + '_PATCH.java'
            writer = io.open(file_path, 'w', encoding='utf-8')
            writer.write(changes)
            writer.close()

        command = 'java -cp ' + EXTRACTOR_JAR + ' JavaExtractor.App --max_path_length ' + \
                    str(MAX_PATH_LENGTH) + ' --max_path_width ' + str(MAX_PATH_WIDTH) + ' --file ' + file_path + ' > ' + id_path
        try:
            subprocess.check_call(command, shell=True, stderr=subprocess.STDOUT)
        except:
            return None
        
        if not pathIsOk:
            os.remove(file_path)

    lines = []
    print(id_path)
    with open(id_path, 'r') as ff:
        for line in ff:
            lines.append(line)
    return lines

def retrieve_identifiers(input):
    data = []
    if input is None:
        return data
    for line in input:
        parts = line.rstrip('\n').split(' ')
        target_name = parts[0]
        contexts = parts[1:]
        # data.append(target_name)
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
    # output_file_path = '../data/train_data5_frag_code2vec.txt'
    # output_file_path = '../data/test_data_code2vec.txt'
    output_file_path = 'test.out'
    out_writer = io.open(output_file_path, 'w+', encoding='utf-8')

    # dir_path = '../data/code2vec_train_data/Patches_train'
    dir_path = '/Users/abdoulkader.kabore/snt/patch_prediction/data/PatchesData/test'
    # dir_path = '/Users/abdoulkader.kabore/snt/patch_prediction/data/code2vec_train_data/Patches_test'

    n = -1
    i = 0
    j = 0
    method = 'kuiData'

    try:
        os.makedirs(TMP_FOLDER)
    except OSError:
        pass

    if method == 'json':
        for f, buggy, patched, patch, seed, id, benchmark, repair_tool, project in get_buggy_patch(dir_path):
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
                bug_id = benchmark + '-' + project + '-' + id
                patch_id = repair_tool + '-' + seed

                buggy_id_path = path + bug_id + '_BUG.id'
                patch_id_path = path + patch_id + '_PATCH.id'

                b_output = process(path, buggy, 'buggy', buggy_id_path, fIsChanged=True)
                if b_output is not None and len(b_output) > 0:
                    p_output = process(path, patched, 'patched', patch_id_path, fIsChanged=True)

                    buggy_tokens = retrieve_identifiers(b_output)
                    patch_tokens = retrieve_identifiers(p_output)

                    if len(buggy_tokens) > 0 and len(patch_tokens) > 0:
                        print(buggy_tokens)
                        print(patch_tokens)
                        label_temp = '1'
                        sample = '<ml>'.join([label_temp, bug_id, patch_id, ' '.join(buggy_tokens), ' '.join(patch_tokens), '<dl>'.join(patch)])
                        out_writer.write(sample.strip('\n') + '\n')
                        i = i + 1
                        continue
                j = j + 1
            except:
                j = j + 1
    elif method == 'kuiData':
        for buggy_f, patch_f in get_deep_files(dir_path=dir_path, paired=True):
            head_tail = os.path.split(buggy_f)
            tmp = head_tail[0].split('/')
            path = TMP_FOLDER + '/' + tmp[n]
            try:
                os.makedirs(path)
            except OSError:
                pass
            else:
                pass
                
            try:
                bug_id = head_tail[1].split('buggyCode.txt')[0]

                buggy_id_path = path + bug_id + 'BUG.id'
                patch_id_path = path + bug_id + 'PATCH.id'

                b_output = process(path, buggy_f, 'buggy', buggy_id_path, pathIsOk=True)
                if b_output is not None and len(b_output) > 0:
                    p_output = process(path, patch_f, 'patched', patch_id_path, pathIsOk=True)

                    buggy_tokens = retrieve_identifiers(b_output)
                    patch_tokens = retrieve_identifiers(p_output)

                    if len(buggy_tokens) > 0 and len(patch_tokens) > 0:
                        label_temp = '1'
                        sample = label_temp + '<ml>' + bug_id + '<ml>' + ' '.join(buggy_tokens) + '<ml>' + ' '.join(patch_tokens)
                        out_writer.write(sample + '\n')
                        i = i + 1
                        continue
                j = j + 1
            except:
                j = j + 1
            
    else:
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
                        out_writer.write(sample + '\n')
                        i = i + 1
                        continue
                j = j + 1
            except:
                j = j + 1
    
    out_writer.close()

    print(i)
    print(j)