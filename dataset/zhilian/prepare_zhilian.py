# @Time   : 2022/4/10
# @Author : Chen Yang
# @Email  : flust@ruc.edu.cn

"""
data_round1
"""

import jieba
import re
import datetime
import tqdm
import os

class zhilian:
    def __init__(self):
        print('hello')
        self.get_inter()
        self.get_user_item()
        self.get_docs()
        pass

    def get_inter(self):
        f = open('table3_action.txt', 'r')
        f_target = open('zhilian.inter', 'w')
        f.readline()
        f_target.write('user_id:token\tjob_id:token\tdirect:token\tlabel:float\n')
        self.uid_set = set()
        self.jid_set = set()
        line_count = 0
        for line in f:
            uid, jid, b, d, s = line[:-1].split('\t')
            # if b == '0' and d == '0' and s == '0':
            if s == '0':
                continue
            self.uid_set.add(uid)
            self.jid_set.add(jid)
            import random
            if random.random() < 0.5:
                direct = '0'
            else:
                direct = '1'
            new_line = '\t'.join([uid, jid, direct, s]) + '\n'
            f_target.write(new_line)
            line_count += 1
        print("uid count is: ", len(self.uid_set))
        print("jid count is: ", len(self.jid_set))
        print("line count is: ", line_count)

    def get_user_item(self):
        u_index = [0, 1, 5, 8, 9, 10, 11]
        f = open('table1_user.txt', 'r')
        f_user = open('zhilian.user', 'w')
        head = f.readline()[:-1].split('\t')
        head = [head[i] for i in u_index]
        head = [i + ':token' for i in head]
        f_user.write('\t'.join(head) + '\n')
        user_line_count = 0
        for line in f:
            lines = line.split('\t')
            lines = [lines[i] for i in u_index]

            if lines[0] in self.uid_set:
                f_user.write('\t'.join(lines) + '\n')
                user_line_count += 1
        print("user line count:", user_line_count)
        f.close()
        f_user.close()

        j_index = [0, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16]
        f = open('table2_jd.txt', 'r')
        f_item = open('zhilian.item', 'w')
        head = f.readline()[:-1].split('\t')
        head = [head[i] for i in j_index]
        head[0] = 'job_id'
        head = [i + ':token' for i in head]
        f_item.write('\t'.join(head) + '\n')
        job_line_count = 0
        for line in f:
            lines = line.split('\t')
            lines = [lines[i] for i in j_index]
            if lines[0] in self.jid_set:
                f_item.write('\t'.join(lines) + '\n')
                job_line_count += 1
        print("job line count:", job_line_count)
        f.close()
        f_item.close()


    def get_docs(self):
        u_doc_index = [3, 4, 6, 7, -1]
        f = open('table1_user.txt', 'r')
        f.readline()
        f_udoc = open('zhilian.udoc', 'w')
        head = ['user_id:token', 'user_doc:token_seq']
        f_udoc.write('\t'.join(head) + '\n')
        user_doc_line_count = 0
        for line in f:
            lines = line[:-1].split('\t')
            # import pdb
            # pdb.set_trace()
            if lines[0] not in self.uid_set:
                continue
            user_doc_line_count += 1
            for i in u_doc_index:
                if lines[i] and lines[i] != '-':
                    sents = lines[i]
                    try:
                        sent_wds, sent_lens, _ = raw2token_seq(sents)
                        sent_wds = sent_wds.split(' ')
                        sent_lens = sent_lens.split(' ')
                        a = -(int)(sent_lens[-1])
                        for j in range(len(sent_lens)):
                            a += (int)(sent_lens[j - 1])
                            s_word_line = ' '.join(sent_wds[a:a + (int)(sent_lens[j])])
                            s_new_line = lines[0] + '\t' + s_word_line + '\n'
                            f_udoc.write(s_new_line)
                    except:
                        print(sents)
                        f_udoc.write(lines[0] + '\t' + sents + '\n')
        print("user_doc_line_count:", user_doc_line_count)
        f.close()
        f_udoc.close()

        # print('job begin')

        j_doc_index = [1, 2, 4, -1]
        f = open('table2_jd.txt', 'r')
        f.readline()
        f_idoc = open('zhilian.idoc', 'w')
        head = ['job_id:token', 'job_doc:token_seq']
        f_idoc.write('\t'.join(head) + '\n')
        # count = 0
        job_doc_line_count = 0
        for line in f:
            # count += 1
            lines = line[:-1].split('\t')
            if lines[0] not in self.jid_set:
                continue
            job_doc_line_count += 1
            for i in j_doc_index:
                if lines[i] and lines[i] != '-' and lines[i] != '\\N':
                    sents = lines[i]
                    try:
                        sent_wds, sent_lens, _ = raw2token_seq(sents)
                        sent_wds = sent_wds.split(' ')
                        sent_lens = sent_lens.split(' ')
                        a = -(int)(sent_lens[-1])
                        for i in range(len(sent_lens)):
                            a += (int)(sent_lens[i - 1])
                            s_word_line = ' '.join(sent_wds[a:a + (int)(sent_lens[i])])
                            s_new_line = lines[0] + '\t' + s_word_line + '\n'
                            f_idoc.write(s_new_line)
                    except:
                        print(sents)
                        f_idoc.write(lines[0] + '\t' + sents + '\n')

            # if count % 1000 == 0:
                # print(count)
        print("job doc line count:", job_doc_line_count)
        f.close()
        f_idoc.close()

# jieba.load_userdict(os.path.join(r'./origin_data_1/job_word.dct'))  # 加载自定义字典
# jobwd2jobtag = {}
# with open(os.path.join(r'./origin_data_1/basic-tag-data-online-v14.json'), 'r', encoding='utf-8') as file:
#     for line in file:
#         js = json.loads(line.strip())
#         jobtag = js['tag_word']
#         for wd in js['writing_words']:
#             jobwd2jobtag[wd] = jobtag


def clean_text(text):
    illegal_set = ',.;?!~[]\'"@#$%^&*()-_=+{}\\`～·！¥（）—「」【】|、“”《<》>？，。…：'   # 定义非法字符

    for c in illegal_set:
        text = text.replace(c, ' ')     # 非法字符 替换为 空格
    for pattern in ['岗位职责', '职位描述', '工作内容', '岗位描述', '岗位说明', '工作职责', '你的职责']:
        text = text.replace(pattern, '')   # 内容头部替换为空格
    text = ' '.join([_ for _ in text.split(' ') if len(_) > 0])
    return text    # 空格间隔


def cut_sent(text):
    wds = [_.strip() for _ in jieba.cut(text) if len(_.strip()) > 0]  # 分词，返回分词后的 list
    return wds


def split_sent(text):
    text = re.split('(?:[0-9][.;。：．•）\)])', text)  # 按照数字分割包括  1.  1;  1。  1：  1) 等
    ans = []
    for t in text:
        for tt in re.split('(?:[\ ][0-9][、，])', t):  #
            for ttt in re.split('(?:^1[、，])', tt):   # 1、
                for tttt in re.split('(?:\([0-9]\))', ttt):   # (1)
                    ans += re.split('(?:[。；…●])', tttt)

    return [_.strip() for _ in ans if len(_.strip()) > 0]


def raw2token_seq(s):
    sents = split_sent(s)
    sent_wds = []
    sent_lens = []
    for sent in sents:
        # 对于分段后每段文字：
        if len(sent) < 2:
            continue
        sent = clean_text(sent)
        if len(sent) < 1:
            continue
        wds = cut_sent(sent)
        # 切词
        # for wd in wds:
        #     if wd in jobwd2jobtag:
        #         wd = jobwd2jobtag[wd]
        #     # 词 对应的 tag
        # if len(wds) < 1: continue
        sent_wds.extend(wds)
        sent_lens.append(len(wds))
    if len(sent_wds) < 1:
        return None, None
    assert sum(sent_lens) == len(sent_wds)
    # 返回3个值，第一个是用空格连接的词，第二个是各句子长度，第三个是总词数
    return ' '.join(sent_wds), ' '.join(map(str, sent_lens)), len(sent_wds)


if __name__ == '__main__':
    zhilian()
    print('finished')