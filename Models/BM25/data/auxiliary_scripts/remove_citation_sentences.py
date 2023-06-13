'''
Removes sentences containing the <CITATION> tag.
'''

import numpy as np, pandas as pd, os, sys, json

folder_path = f'../corpus/ik_test'
destination_path = f'../corpus/sentence_removed/ik_test'

assert(os.path.isdir(folder_path))
os.makedirs(destination_path + f'/query/', exist_ok=True)
os.makedirs(destination_path + f'/candidate/', exist_ok=True)

# taken from stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
# Added Rs. to prefixes

import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Rs)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
digits = "([0-9])"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    # if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

# do on query cases
for f in (os.listdir(folder_path + f'/query/')):
    path = folder_path + f'/query/' + f
    outfile_path = destination_path + f'/query/' + f
    with open(path, 'r') as f_:
        content = f_.read()
    content = split_into_sentences(content)
    content = [i for i in content if not ('CITATION' in i)]   # remove tag containing sentences
    content = ' '.join(content)

    with open(outfile_path, 'w+') as f_out:
        f_out.write(content)

# do on candidate cases
for f in (os.listdir(folder_path + f'/candidate/')):
    path = folder_path + f'/candidate/' + f
    outfile_path = destination_path + f'/candidate/' + f
    with open(path, 'r') as f_:
        content = f_.read()
    content = split_into_sentences(content)
    content = [i for i in content if not ('CITATION' in i)]   # remove tag containing sentences
    content = ' '.join(content)

    with open(outfile_path, 'w+') as f_out:
        f_out.write(content)
