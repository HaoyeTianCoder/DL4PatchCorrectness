import os
import numpy as np
import pandas as pd
import re
from bert_serving.client import BertClient
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import word2vec,Doc2Vec, KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import json
from utils import my_split

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

path_patch_test = '/Users/haoye.tian/Documents/University/data/kui_patches/Patches_test'

root_new = '/Users/haoye.tian/Documents/University/data/experiment2'
benchmarks = ["Bears", "Bugs.jar", "Defects4J", "IntroClassJava","QuixBugs"]
# benchmarks = ['QuixBugs']
tools = ["Arja", "GenProg", "Kali", "RSRepair", "Cardumen", "jGenProg", "jKali", "jMutRepair", "Nopol", "DynaMoth", "NPEFix"]

# Begin code2vec
import io
import datetime
import subprocess

EXTRACTOR_JAR = '/Users/abdoulkader.kabore/snt/code2vec/JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar'
MAX_PATH_LENGTH = 8 
MAX_PATH_WIDTH = 2
TMP_FOLDER = 'tmp'

code2vec_path = '/Users/abdoulkader.kabore/snt/code2vec/models/code2vec/w2v_tokens_format.txt'
path_patch_test = '../data/code2vec_train_data/Patches_test'
root_new = '../data/experiment2_code2vec'

print('Loading code2vec pretrained model')
code2vec_model = KeyedVectors.load_word2vec_format(code2vec_path, binary=False)

try:
    os.makedirs(TMP_FOLDER)
except OSError:
    pass

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

def code2vec(df, code2vec_path, threshold):

    block = ''
    length =df.shape[0]
    bug_id = str(df['bugid'][0])
    block += '**************\n'
    block += 'Bugid: {}, patches: {} \n'.format(bug_id,length)

    # tokenize
    df['buggy'] = df['buggy'].map(lambda x: my_split(x))
    df['patched'] = df['patched'].map(lambda x: my_split(x))
    # result = cosine_similarity(bc.encode(list(np.array(df_quick['buggy']))),bc.encode(list(np.array(df_quick['patched']))))
    df['simi'] = None
    
    for index,row in df.iterrows():
        print('{}/{}'.format(index,length))
        if row['buggy'] == [] or row['patched'] == []:
            print('buggy or patched is []')
            continue
        try:
            buggy_tokens = []
            patch_tokens = []

            for i, b in enumerate(row['buggy']):
                if b in code2vec_model.vocab:
                    buggy_tokens.append(b)

            for i, b in enumerate(row['patched']):
                if b in code2vec_model.vocab:
                    patch_tokens.append(b)

            if len(buggy_tokens) == 0 or len(patch_tokens) == 0:
                continue

            bug_vec = code2vec_model[buggy_tokens]
            patched_vec = code2vec_model[patch_tokens]

            bug_vec = np.average(bug_vec, axis=0)
            patched_vec = np.average(patched_vec, axis=0)

        except Exception as e:
            print(e)
            continue
        result = cosine_similarity([bug_vec], [patched_vec])
        df.loc[index,'simi'] = float(result[0][0])
    df = df.sort_values(by='simi')
    df.index = range(len(df))

    return None, None, df

def process_id_extraction(path, changes, type, id_path):
    if not os.path.exists(id_path):
        
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
        
        os.remove(file_path)

    lines = []
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

# End

def get_diff_files_frag(patch,type):
    # with open(patch, 'r') as file:
        lines = ''
        p = r"([^\w_])"
        flag = True
        # try:
        for line in patch:
            line = line.strip()
            if '*/' in line:
                flag = True
                continue
            if flag == False:
                continue
            if line != '':
                if line.startswith('@@') or line.startswith('diff') or line.startswith('index'):
                    continue
                elif '/*' in line:
                    flag = False
                    continue
                elif type == 'buggy':
                    if line.startswith('---'):
                        line = re.split(pattern=p, string=line.split(' ')[1].strip())
                        lines += ' '.join(line) + ' '
                    elif line.startswith('-'):
                        if line[1:].strip().startswith('//'):
                            continue
                        line = re.split(pattern=p, string=line[1:].strip())
                        line = [x.strip() for x in line]
                        while '' in line:
                            line.remove('')
                        line = ' '.join(line)
                        lines += line.strip() + ' '
                    elif line.startswith('+'):
                        # do nothing
                        pass
                    else:
                        line = re.split(pattern=p, string=line.strip())
                        line = [x.strip() for x in line]
                        while '' in line:
                            line.remove('')
                        line = ' '.join(line)
                        lines += line.strip() + ' '
                elif type == 'patched':
                    if line.startswith('+++'):
                        line = re.split(pattern=p, string=line.split(' ')[1].strip())
                        lines += ' '.join(line) + ' '
                    elif line.startswith('+'):
                        if line[1:].strip().startswith('//'):
                            continue
                        line = re.split(pattern=p, string=line[1:].strip())
                        line = [x.strip() for x in line]
                        while '' in line:
                            line.remove('')
                        line = ' '.join(line)
                        lines += line.strip() + ' '
                    elif line.startswith('-'):
                        # do nothing
                        pass
                    else:
                        line = re.split(pattern=p, string=line.strip())
                        line = [x.strip() for x in line]
                        while '' in line:
                            line.remove('')
                        line = ' '.join(line)
                        lines += line.strip() + ' '
        # except Exception:
        #     print(Exception)
        #     return 'Error'
        return lines

def test_similarity_repair_tool(path_patch_test, model, threshold):
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
                data = ''
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
                            bug_id = benchmark + '-' + project + '-' + id
                            patch_id = repair_tool + '-' + seed
                            with open(results_path, 'r') as f1:
                                patch_dict = json.load(f1)
                            patches = patch_dict['patches']
                            if patches != []:
                                # data = np.array([])
                                path_result = os.path.join(root_new,repair_tool,bug_id)
                                # Doc_top10 = os.path.join(path_result,'Doc_top10')
                                # Bert_top10 = os.path.join(path_result,'Bert_top10')
                                # Doc_threshold = os.path.join(path_result,'Doc_threshold')
                                # Bert_threshold = os.path.join(path_result,'Bert_threshold')
                                cnt = 0
                                for p in patches:
                                    if 'patch' in p:
                                        patch = p['patch']
                                        p_list = patch.split('\n')
                                    elif 'PATCH_DIFF_ORIG' in p:
                                        patch = p['PATCH_DIFF_ORIG']
                                        p_list = patch.split('\\n')
                                    else:
                                        print('error...')

                                    if model == 'code2vec':
                                        buggy = get_diff_files(p_list, type='buggy', withFile=False)
                                        patched = get_diff_files(p_list, type='patched', withFile=False)
                                    else:
                                        buggy = get_diff_files_frag(p_list, type='buggy')
                                        patched = get_diff_files_frag(p_list, type='patched')

                                    buggy_id_path = TMP_FOLDER + '/' + bug_id + '_BUG.id'
                                    patch_id_path = TMP_FOLDER + '/' + patch_id + '_PATCH.id'

                                    b_output = process_id_extraction(TMP_FOLDER, buggy, 'buggy', buggy_id_path)
                                    if b_output is not None:
                                        p_output = process_id_extraction(TMP_FOLDER, patched, 'patched', patch_id_path)

                                        buggy_tokens = retrieve_identifiers(b_output)
                                        patch_tokens = retrieve_identifiers(p_output)
                                        
                                        if len(buggy_tokens) > 0 and len(patch_tokens) > 0:
                                            buggy = ' '.join(buggy_tokens)
                                            patched = ' '.join(patch_tokens)

                                            sample = np.array(['1', bug_id, patch_id, buggy, patched, patch]).reshape((-1, 6))
                                            if cnt == 0:
                                                data = sample
                                            else:
                                                data = np.concatenate((data, sample), axis=0)
                                            cnt += 1

                                if isinstance(data, str):
                                    continue

                                calulate_similarity(path_result,data,model,threshold)


def calulate_similarity(path_result,data, model, threshold):
    data = data.reshape((-1, 6))
    df = pd.DataFrame(data, dtype=str, columns=['label', 'bugid', 'patchid', 'buggy', 'patched', 'patch'])
    if model != 'code2vec':
        # tokenize
        df['buggy'] = df['buggy'].map(lambda x: word_tokenize(x))
        df['patched'] = df['patched'].map(lambda x: word_tokenize(x))
    df['simi'] = None

    if model == 'bert':
        df_threshold, df_top, df = bert(df, threshold)
    elif model == 'doc':
        df_threshold, df_top, df = doc(df, threshold)
    elif model == 'code2vec':
        df_threshold, df_top, df = code2vec(df, code2vec_path, threshold)
    else:
        print('wrong model')
        raise ('wrong model')

    #all
    path_rank = os.path.join(path_result, model + '_all')
    if not os.path.exists(path_rank):
        os.makedirs(path_rank)
    for index, row in df.iterrows():
        similarity = str(row['simi'])
        patch_id = str(row['patchid'])
        path_save = str(index) + '_' + similarity + '_' + patch_id + '.txt'
        patch = str(row['patch'])
        with open(os.path.join(path_rank, path_save), 'w+') as f:
            f.write(patch)

    # threshold version
    # path_rank = os.path.join(path_result, model+'_threshold')
    # if not os.path.exists(path_rank):
    #     os.makedirs(path_rank)
    # for index, row in df_threshold.iterrows():
    #     similarity = str(row['simi'])
    #     patch_id = str(row['patchid'])
    #     path_save = str(index) + '_' + similarity + '_' + patch_id + '.txt'
    #     patch = str(row['patch'])
    #     with open(os.path.join(path_rank, path_save),'w+') as f:
    #         f.write(patch)

    # top version
    # path_rank = os.path.join(path_result, model + '_top10')
    # if not os.path.exists(path_rank):
    #     os.mkdir(path_rank)
    # for index, row in df_top.iterrows():
    #     similarity = str(row['simi'])
    #     patch_id = str(row['patchid'])
    #     path_save = str(index) + '_' + similarity + '_' + patch_id + '.txt'
    #     patch = str(row['patch'])
    #     with open(os.path.join(path_rank, path_save), 'w+') as f:
    #         f.write(patch)

    # df[['bugid','patchid','simi','patch']].to_csv(os.path.join(path_result, model + '_all_patches.csv'), header=None, index=None, sep=' ', mode='a+')

def test_similarity(path_patch_test, model, threshold):
    # os.remove('../data/test_result_'+ model + '.txt' )
    flag = 0
    for root,dirs,files in os.walk(path_patch_test):
        for file in files:
            if file == 'test_data_bug_patches.txt':
                test_data = os.path.join(root,file)
                data = np.loadtxt(test_data, dtype=str, comments=None, delimiter='<ml>')
                data = data.reshape((-1,6))
                try:
                    df = pd.DataFrame(data, dtype=str, columns=['label', 'bugid', 'patchid','buggy', 'patched','patch'])
                except Exception as e:
                    print(e)
                # tokenize
                df['buggy'] = df['buggy'].map(lambda x: word_tokenize(x))
                df['patched'] = df['patched'].map(lambda x: word_tokenize(x))
                df['simi'] = None
                #
                # if str(df['bugid'][0]) == 'Defects4J-Chart-12':
                #     flag = 1
                #     continue
                # if flag == 0:
                #     continue

                if model == 'bert':
                    df_threshold, df_top = bert(df, threshold)
                elif model == 'doc':
                    df_threshold, df_top= doc(df, threshold)
                elif model == 'code2vec':
                    df_threshold, df_top, df = code2vec(df, code2vec_path, threshold)
                else:
                    print('wrong model')

                # threshold version
                path_rank = os.path.join(root,model+'_threshold_version')
                if not os.path.exists(path_rank):
                    os.mkdir(path_rank)

                for index, row in df_threshold.iterrows():
                    similarity = str(row['simi'])
                    patch_id = str(row['patchid'])
                    path_save = str(index) + '_' + similarity + '_' + patch_id + '.txt'
                    patch = str(row['patch'])
                    patch = patch.replace('<dl>','\n')
                    with open(os.path.join(path_rank, path_save),'w+') as f:
                        f.write(patch)

                # top version
                path_rank = os.path.join(root, model + '_top_version')
                if not os.path.exists(path_rank):
                    os.mkdir(path_rank)

                for index, row in df_top.iterrows():
                    similarity = str(row['simi'])
                    patch_id = str(row['patchid'])
                    path_save = str(index) + '_' + similarity + '_' + patch_id + '.txt'
                    patch = str(row['patch'])
                    patch = patch.replace('<dl>', '\n')
                    with open(os.path.join(path_rank, path_save), 'w+') as f:
                        f.write(patch)

            # df_ranked[['bugid','patchid','simi']].to_csv(os.path.join(root,'ranked_list.csv'), header=None, index=None, sep=' ',
            #                              mode='a')


def bert(df, threshold):
    block = ''
    length =df.shape[0]
    bug_id = str(df['bugid'][0])
    block += '**************\n'
    block += 'Bugid: {}, patches: {} \n'.format(bug_id,length)

    # to do: max_seq_len=360
    bc = BertClient(check_length=False)
    for index, row in df.iterrows():
        try:
            bug_vec = bc.encode([row['buggy']], is_tokenized=True)
            patched_vec = bc.encode([row['patched']], is_tokenized=True)
        except Exception as e:
            print(e)
            continue
        result = cosine_similarity(bug_vec, patched_vec)
        df.loc[index, 'simi'] = result[0][0]
    df = df.sort_values(by='simi',ascending=False)
    df.index = range(len(df))
    # threshold
    # df_threshold = df[df['simi'].values >= threshold]
    #
    # # top filter
    # top = 10
    # df_top = df[:top]
    #
    # block += 'Top: {}, post_patches: {}\n'.format(top, df_top.shape[0])
    # block += 'Threshold: {}, post_patches: {}\n'.format(threshold, df_threshold.shape[0])
    #
    # block += '{}\n'.format(df_top[['bugid', 'patchid', 'simi']])
    print(block)
    # with open('../data/test_result_bert_new_threshold_top.txt', 'a+') as f:
    #     f.write(block)

    # return df_threshold, df_top, df
    return None, None, df

def doc(df, threshold):
    block = ''
    length = df.shape[0]
    bug_id = str(df['bugid'][0])
    block += '**************\n'
    block += 'Bugid: {}, patches: {} \n'.format(bug_id, length)

    model = Doc2Vec.load('../data/doc_frag.model')

    for index, row in df.iterrows():
        bug_vec = model.infer_vector(row['buggy'] ,alpha=0.025,steps=300)
        patched_vec = model.infer_vector(row['patched'],alpha=0.025,steps=300)
        # similarity calculation
        result = cosine_similarity(bug_vec.reshape((1,-1)), patched_vec.reshape((1,-1)))
        df.loc[index, 'simi'] = result[0][0]
    df = df.sort_values(by='simi', ascending=False)
    df.index = range(len(df))
    # # threshold
    # df_threshold = df[df['simi'].values >= threshold]
    #
    # # top filter
    # top = 10
    # df_top = df[:top]
    #
    # block += 'Top: {}, post_patches: {}\n'.format(top, df_top.shape[0])
    # block += 'Threshold: {}, post_patches: {}\n'.format(threshold, df_threshold.shape[0])
    # block += '{}\n'.format(df_top[['bugid', 'patchid', 'simi']])

    print(block)
    # with open('../data/test_result_doc_new_threshold_top.txt', 'a+') as f:
    #     f.write(block)

    # return df_threshold, df_top, df
    return None, None, df

if __name__ == '__main__':

    # bert minumum, average, median
    model = 'bert'
    # threshold = [0.90844, 0.99825, 0.99894]

    # doc minumum, average, median, 1stqu
    # model = 'doc'
    # threshold = [0.28489, 0.91795, 0.93891, 0.8580]

    # test_similarity(path_patch_test,model=model,threshold=threshold[1])

    model = 'code2vec'
    test_similarity_repair_tool(path_patch_test,model=model,threshold=None)
