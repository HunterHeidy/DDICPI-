
import os, sys, json, codecs, spacy, en_core_web_md

def tokenizing_and_tagging(sentence, model):
    toks, poses = [], []
    for word in model(sentence):
        toks.append(word.text.strip())
        poses.append(word.tag_)
    return toks, poses


def charidx2tokidx(chari, mapping, sentence_ori):
    toki = None
    try:
        toki = mapping[chari]
    except KeyError:
        try:
            toki = mapping[chari+1]
        except KeyError:
            print '!!! charidx2tokidx failed'
            print sentence_ori[:chari].encode('utf-8'), '|||', sentence_ori[chari].encode('utf-8'), '|||', sentence_ori[chari+1:].encode('utf-8')
    return toki


def process(path, data, fconllu, tokener_tagger):
    for n, line in enumerate(codecs.open(path, 'rU', 'utf-8')):
        if n == 0:
            continue # skip the headline

        line = line.strip().split('\t') # (file_id, sentence, G, PH, G-id, PH-id, G-st, G-ed, PH-st, PH-ed, rel)
        inst = {'id':line[0], 'ref':line[10], 'subj_id':line[4], 'obj_id':line[5], }

        sentence = sentence_ori = line[1]
        sentence = sentence.replace(':', ' : ', 100)
        sentence = sentence.replace('-', ' - ', 100)
        sentence = sentence.replace('+', ' + ', 100)
        sentence = sentence.replace('/', ' / ', 100)
        sentence = sentence.replace('(', ' ( ', 100)
        sentence = sentence.replace(')', ' ) ', 100)
        sentence = sentence.replace('=', ' = ', 100)
        sentence = sentence.replace('<sup>', ' <sup> ', 100)
        sentence = sentence.replace('</sup>', ' </sup> ', 100)
        sentence = sentence.replace('&quot;', ' &quot; ', 100)
        sentence = sentence.replace('&amp;', ' &amp; ', 100)
        sentence = sentence.replace('mdash;', ' mdash; ', 100)
        sentence = sentence.replace('VEGF-A', 'VEGF - A', 100)
        sentence = sentence.replace('Mfrprd6', ' Mfrp rd6', 100)
        sentence = sentence.replace('miRNA', ' miR NA', 100)
        sentence = sentence.replace('trm7D', 'trm7 D', 100)
        sentence = sentence.replace('KCNC3R423H', 'KCNC3 R423H', 100)
        sentence = sentence.replace('cancer', ' cancer ', 100)
        sentence = sentence.replace('tumor', ' tumor ', 100)
        sentence = sentence.replace('tumour', ' tumour ', 100)
        sentence = sentence.replace('pheno', ' pheno ', 100)
        sentence = sentence.replace('type', ' type ', 100)
        sentence = sentence.replace('spinocerebellar', 'spino cerebellar', 100)
        sentence = ' '.join(sentence.split())

        tokens, poses = tokenizing_and_tagging(sentence, tokener_tagger)
        inst['tokens'] = tokens
        inst['poses'] = poses

        mapping_c2st = {}
        mapping_c2ed = {}
        c = 0
        for i, word in enumerate(tokens):
            while sentence_ori[c].isspace():
                c += 1
            mapping_c2st[c] = i
            for j in range(len(word)):
                if word[j] != sentence_ori[c]:
                    print u'%s ||| %s' %(word[j:].decode('utf-8'), sentence_ori[c:].decode('utf-8'))
                    assert False
                c += 1
            mapping_c2ed[c] = i

        s_char_st, s_char_ed, e_char_st, e_char_ed = int(line[6]), int(line[7]), int(line[8]), int(line[9])
        inst['subj_start'] = charidx2tokidx(s_char_st, mapping_c2st, sentence_ori)
        inst['subj_end'] = charidx2tokidx(s_char_ed, mapping_c2ed, sentence_ori)
        inst['obj_start'] = charidx2tokidx(e_char_st, mapping_c2st, sentence_ori)
        inst['obj_end'] = charidx2tokidx(e_char_ed, mapping_c2ed, sentence_ori)

        if None in (inst['subj_start'], inst['subj_end'], inst['obj_start'], inst['obj_end'],):
            continue

        data.append(inst)
        for i, (t, p) in enumerate(zip(tokens, poses)):
            print >>fconllu, '\t'.join([str(i+1), t, t, p, p, '_', '0', 'root', '_', '_'])
        print >>fconllu, ''


#=================================

tokener_tagger = en_core_web_md.load()
train, test = [], []
train_fconllu = codecs.open('train.conllu', 'w', 'utf-8')
test_fconllu = codecs.open('test.conllu', 'w', 'utf-8')
process('train_1.tsv', train, train_fconllu, tokener_tagger)
process('train_2.tsv', train, train_fconllu, tokener_tagger)
process('test.tsv', test, test_fconllu, tokener_tagger)
train_fconllu.close()
test_fconllu.close()
json.dump(train, codecs.open('train.json', 'w', 'utf-8'))
json.dump(test, codecs.open('test.json', 'w', 'utf-8'))

