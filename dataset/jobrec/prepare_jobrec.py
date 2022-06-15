# .udoc

import re
def split_sent(text):
    text = re.split('(?:[0-9][.;。：．•）\)])', text)  # 按照数字分割包括  1.  1;  1。  1：  1) 等
    ans = []
    for t in text:
        for tt in re.split('(?:[\ ][0-9][、，])', t):  #
            for ttt in re.split('(?:^1[、，])', tt):   # 1、
                for tttt in re.split('(?:\([0-9]\))', ttt):   # (1)
                    ans += re.split('(?:[。；…●])', tttt)

    return [_.strip() for _ in ans if len(_.strip()) > 0]


def cut_sent(text):
    wds = [_.strip() for _ in text.split(' ') if len(_.strip()) > 0]  # 分词，返回分词后的 list
    return wds


def clean_text(text):
    illegal_set = ',.;?!~[]\'"@#$%^&*()-_=+{}`～·！¥（）—「」【】|、“”《<》>？，。…：'   # 定义非法字符

    for c in illegal_set:
        text = text.replace(c, ' ')     # 非法字符 替换为 空格
    text = ' '.join([_ for _ in text.split(' ') if len(_) > 0])
    
    return text    # 空格间隔


def raw2token_seq(s):
    sents = split_sent(s)
    sent_wds = []
    sent_lens = []
    for sent in sents:
        if len(sent) < 2:
            continue
        sent = clean_text(sent)
        if len(sent) < 1:
            continue
        wds = cut_sent(sent)
        sent_wds.extend(wds)
        sent_lens.append(len(wds))
    if len(sent_wds) < 1:
        return None, None, None
    assert sum(sent_lens) == len(sent_wds)
    # 返回3个值，第一个是用空格连接的词，第二个是各句子长度，第三个是总词数
    return ' '.join(sent_wds), ' '.join(map(str, sent_lens)), len(sent_wds)

u_doc_index = [3, 4, 5, 7, 8]
f = open('users.tsv', 'r')
f.readline()
f_his = open('user_history.tsv', 'r')
f_his.readline()

f_udoc = open('jobrec.udoc', 'w')
head = ['user_id:token', 'user_doc:token_seq']
f_udoc.write('\t'.join(head) + '\n')


for line in f_his:
    lines = line[:-1].split('\t')
    sents = lines[4]
    try:
        sent_wds, sent_lens, _ = raw2token_seq(sents)
        sent_wds = sent_wds.split(' ')
        sent_lens = sent_lens.split(' ')
        a = -(int)(sent_lens[-1])
        for j in range(len(sent_lens)):
            a += (int)(sent_lens[j - 1])
            s_word_line = ' '.join(sent_wds[a:a + (int)(sent_lens[j])])
            if s_word_line == '':
                continue
            s_new_line = lines[0] + '\t' + s_word_line + '\n'
            f_udoc.write(s_new_line)
    except:
        if sents == '':
            continue
        f_udoc.write(lines[0] + '\t' + sents + '\n')


for line in f:
    lines = line[:-1].split('\t')
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
                f_udoc.write(lines[0] + '\t' + sents + '\n')
                
f.close()
f_his.close()
f_udoc.close()


# .idoc
import re

i_doc_index = [2, 3, 4]
f = open('jobs.tsv', 'r')
f.readline()

f_idoc = open('jobrec.idoc', 'w')
head = ['item_id:token', 'item_doc:token_seq']
f_idoc.write('\t'.join(head) + '\n')


for line in f:
    lines = line[:-1].split('\t')
    for i in i_doc_index:
        if lines[i] and lines[i] != '-':
            sents = lines[i]
            sents = re.sub('\\\\r', ' ', sents)
            sents = re.sub('<[^>]+>', ' ', sents)
            sents = re.sub('&nbsp;', ' ', sents)
            sents = sents.split('   ')
            for sent in sents:
                sent = sent.strip()
                if len(sent) < 20:
                    continue
                s_new_line = lines[0] + '\t' + sent + '\n'
                f_idoc.write(s_new_line)
                
f.close()
f_idoc.close()

from collections import defaultdict
f = open('jobrec.udoc')
f.readline()
u_doc_num = defaultdict(int)
for l in f:
    lines = l.split('\t')
    u_doc_num[lines[0]] += 1
u_doc = sorted(u_doc_num.items(), key=lambda x: x[1])
count = 1
for i in u_doc:
    if i[1] < 10:
        count += 1
print(count)

from collections import defaultdict
f = open('jobrec.idoc')
f.readline()
i_doc_num = defaultdict(int)
for l in f:
    lines = l.split('\t')
    i_doc_num[lines[0]] += 1
i_doc = sorted(i_doc_num.items(), key=lambda x: x[1])
count = 1
for i in i_doc:
    if i[1] < 10:
        count += 1
print(count)

USER_THRESHOLD = 10
JOB_THRESHOLD = 20


# .inter
import time
def get_timestamp(timeStr):
    timeArray = time.strptime(timeStr, '%Y-%m-%d %H:%M:%S')
    timeStamp = int(time.mktime(timeArray))
    return timeStamp


f_inter = open('apps.tsv', 'r')
f_inter_target = open('jobrec.inter', 'w')
f_inter_target.write('user_id:token\titem_id:token\ttimestamp:float\n')
f_inter.readline()
user = set()
job = set()
l = f_inter.readline()
while l:
    lines = l.split('\t')
    uid = lines[0]
    iid = lines[-1][:-1]
    new_line = lines[0] + '\t' + lines[-1][:-1] + '\t' + str(get_timestamp(lines[-2][:19])) + '\n'
    if u_doc_num[lines[0]] > USER_THRESHOLD and i_doc_num[lines[0]] > JOB_THRESHOLD:
        user.add(uid)
        job.add(iid)
        f_inter_target.write(new_line)
    l = f_inter.readline()

f_inter.close()
f_inter_target.close()

# .user
u_index = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
f_user = open('users.tsv', 'r')
f_user_target = open('jobrec.user', 'w')
head = f_user.readline()[:-1].split('\t')
head = [head[i] for i in u_index]
head = [i + ':token' for i in head]
head[0] = 'user_id:token'
f_user_target.write('\t'.join(head) + '\n')
l = f_user.readline()

while l:
    lines = l[:-1].split('\t')
    lines = [lines[i] for i in u_index]
    new_line = '\t'.join(lines) + '\n'
    if lines[0] in user and u_doc_num[lines[0]] > USER_THRESHOLD:
        f_user_target.write(new_line)
    l = f_user.readline()

f_user.close()
f_user_target.close()

# .item
i_index = [0, 5, 6, 7, 8]
f_item = open('jobs.tsv', 'r')
f_item_target = open('jobrec.item', 'w')

head = f_item.readline()[:-1].split('\t')
head = [head[i] for i in i_index]
head = [i + ':token' for i in head]
head[0] = 'item_id:token'
f_item_target.write('\t'.join(head) + '\n')
l = f_item.readline()

while l:
    lines = l.split('\t')
    lines = [lines[i] for i in i_index]
    new_line = '\t'.join(lines) + '\n'
    if lines[0] in job and i_doc_num[lines[0]] > JOB_THRESHOLD:
        f_item_target.write(new_line)
    l = f_item.readline()

f_item.close()
f_item_target.close()


# .udoc
import re

def split_sent(text):
    text = re.split('(?:[0-9][.;。：．•）\)])', text)  # 按照数字分割包括  1.  1;  1。  1：  1) 等
    ans = []
    for t in text:
        for tt in re.split('(?:[\ ][0-9][、，])', t):  #
            for ttt in re.split('(?:^1[、，])', tt):   # 1、
                for tttt in re.split('(?:\([0-9]\))', ttt):   # (1)
                    ans += re.split('(?:[。；…●])', tttt)

    return [_.strip() for _ in ans if len(_.strip()) > 0]


def cut_sent(text):
    wds = [_.strip() for _ in text.split(' ') if len(_.strip()) > 0]  # 分词，返回分词后的 list
    return wds


def clean_text(text):
    illegal_set = ',.;?!~[]\'"@#$%^&*()-_=+{}`～·！¥（）—「」【】|、“”《<》>？，。…：'   # 定义非法字符

    for c in illegal_set:
        text = text.replace(c, ' ')     # 非法字符 替换为 空格
    text = ' '.join([_ for _ in text.split(' ') if len(_) > 0])
    
    return text    # 空格间隔


def raw2token_seq(s):
    sents = split_sent(s)
    sent_wds = []
    sent_lens = []
    for sent in sents:
        if len(sent) < 2:
            continue
        sent = clean_text(sent)
        if len(sent) < 1:
            continue
        wds = cut_sent(sent)
        sent_wds.extend(wds)
        sent_lens.append(len(wds))
    if len(sent_wds) < 1:
        return None, None, None
    assert sum(sent_lens) == len(sent_wds)
    # 返回3个值，第一个是用空格连接的词，第二个是各句子长度，第三个是总词数
    return ' '.join(sent_wds), ' '.join(map(str, sent_lens)), len(sent_wds)

u_doc_index = [3, 4, 5, 7, 8]
f = open('users.tsv', 'r')
f.readline()
f_his = open('user_history.tsv', 'r')
f_his.readline()

f_udoc = open('jobrec.udoc', 'w')
head = ['user_id:token', 'user_doc:token_seq']
f_udoc.write('\t'.join(head) + '\n')


for line in f_his:
    lines = line[:-1].split('\t')
    sents = lines[4]
    try:
        sent_wds, sent_lens, _ = raw2token_seq(sents)
        sent_wds = sent_wds.split(' ')
        sent_lens = sent_lens.split(' ')
        a = -(int)(sent_lens[-1])
        for j in range(len(sent_lens)):
            a += (int)(sent_lens[j - 1])
            s_word_line = ' '.join(sent_wds[a:a + (int)(sent_lens[j])])
            if s_word_line == '':
                continue
            s_new_line = lines[0] + '\t' + s_word_line + '\n'
            f_udoc.write(s_new_line)
    except:
        if sents == '':
            continue
        f_udoc.write(lines[0] + '\t' + sents + '\n')


for line in f:
    lines = line[:-1].split('\t')
    if u_doc_num[lines[0]] > USER_THRESHOLD:
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
                    f_udoc.write(lines[0] + '\t' + sents + '\n')
                
f.close()
f_his.close()
f_udoc.close()


# .idoc
import re

i_doc_index = [2, 3, 4]
f = open('jobs.tsv', 'r')
f.readline()

f_idoc = open('jobrec.idoc', 'w')
head = ['item_id:token', 'item_doc:token_seq']
f_idoc.write('\t'.join(head) + '\n')


for line in f:
    lines = line[:-1].split('\t')
    if i_doc_num[lines[0]] > JOB_THRESHOLD:
        for i in i_doc_index:
            if lines[i] and lines[i] != '-':
                sents = lines[i]
                sents = re.sub('\"', '', sents)
                sents = re.sub('\\\\r', ' ', sents)
                sents = re.sub('<[^>]+>', ' ', sents)
                sents = re.sub('&nbsp;', ' ', sents)
                sents = sents.split('   ')
                for sent in sents:
                    sent = sent.strip()
                    if len(sent) < 20:
                        continue
                    s_new_line = lines[0] + '\t' + sent + '\n'

                    f_idoc.write(s_new_line)
                
f.close()
f_idoc.close()
