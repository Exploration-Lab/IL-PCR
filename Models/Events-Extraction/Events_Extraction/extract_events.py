#!/usr/bin/env python
# coding: utf-8

# In[48]:


import string
from tqdm import tqdm
import spacy
import os
import json
import re
import pickle
import time
# import concurrent.futures as cf
# from concurrent.futures import ProcessPoolExecutor

# import warnings
# warnings.filterwarnings("ignore")

# In[49]:
# spacy.prefer_gpu()
spacy.require_gpu()
global nlp
nlp = spacy.load("en_core_web_trf")


# In[50]:


alphabet_string = string.ascii_lowercase
alphabet_list = list(alphabet_string)
exclusion_list = alphabet_list + [
    "no",
    "nos",
    "sub-s",
    "subs",
    "ss",
    "cl",
    "dr",
    "mr",
    "mrs",
    "dr",
    "vs",
    "ch",
    "addl",
]
exclusion_list = [word + "." for word in exclusion_list]


# In[51]:


# def preprocess(content):
#     l_new = []
#     for token in content.split():
#         if token.find(".") != -1:
#             if token.count(".") < 2:
#                 if token.lower() in exclusion_list:
#                     # print(token)
#                     l_new.append(" ".join(token.split(".")))
#                 elif token.strip()[-1] != ".":
#                     l_new.append(" ".join(token.split(".")))
#                 else:
#                     l_new.append(token.strip())
#             else:
#                 l_new.append(" ".join(token.split(".")))
#         else:
#             l_new.append(token)
#     for i in range(len(l_new)):
#         l_new[i] = l_new[i].strip()
#     return " ".join(l_new)

def preprocess(content):
    raw_text = re.sub(r"\xa0", " ", content)
    raw_text = raw_text.split("\n")  # splitting using new line character
    text = raw_text.copy()
    text = [re.sub(r'[^a-zA-Z0-9.,<>)\-(/?\t ]', '', sentence)
            for sentence in text]
    # text1 = [re.sub(r'(?<=[^0-9])/(?=[^0-9])',' ',sentence) for sentence in text1]
    text = [re.sub("\t+", " ", sentence) for sentence in text]
    # converting multiple tabs and spaces ito a single tab or space
    text = [re.sub("\s+", " ", sentence) for sentence in text]
    text = [re.sub(" +", " ", sentence) for sentence in text]
    # these were the commmon noises in out data, depends on data
    text = [re.sub("\.\.+", "", sentence) for sentence in text]
    text = [re.sub("\A ?", "", sentence) for sentence in text]
    text = [sentence for sentence in text if(
        len(sentence) != 1 and not re.fullmatch("(\d|\d\d|\d\d\d)", sentence))]
    text = [sentence for sentence in text if len(sentence) != 0]
    text = [re.sub('\A\(?(\d|\d\d\d|\d\d|[a-zA-Z])(\.|\))\s?(?=[A-Z])', '\n', sentence)
            for sentence in text]  # dividing into para wrt to points
    text = [re.sub("\A\(([ivx]+)\)\s?(?=[a-zA-Z0-9])", '\n', sentence)
            for sentence in text]  # dividing into para wrt to roman points
    text = [re.sub(r"[()[\]\"$']", " ", sentence) for sentence in text]
    text = [re.sub(r" no.", " number ", sentence, flags=re.I)
            for sentence in text]
    text = [re.sub(r" nos.", " numbers ", sentence, flags=re.I)
            for sentence in text]
    text = [re.sub(r" co.", " company ", sentence) for sentence in text]
    text = [re.sub(r" ltd.", " limited ", sentence, flags=re.I)
            for sentence in text]
    text = [re.sub(r" pvt.", " private ", sentence, flags=re.I)
            for sentence in text]
    text = [re.sub(r" vs\.?", " versus ", sentence, flags=re.I)
            for sentence in text]
    text = [re.sub(r"ors\.?", "others", sentence, flags=re.I)
            for sentence in text]
    # text = [re.sub("\s+"," ",sentence) for sentence in text]
    text2 = []
    for index in range(len(text)):  # for removing multiple new-lines
        if(index > 0 and text[index] == '' and text[index-1] == ''):
            continue
        if(index < len(text)-1 and text[index+1] != '' and text[index+1][0] == '\n' and text[index] == ''):
            continue
        text2.append(text[index])
    text = text2

    text = "\n".join(text)
    lines = text.split("\n")
    text_new = " ".join(lines)
    text_new = re.sub(" +", " ", text_new)
    l_new = []
    for token in text_new.split():
        if token.lower() not in exclusion_list:
            l_new.append(token.strip())
    return " ".join(l_new)


# In[52]:


SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
OBJECTS = ["dobj", "dative", "attr", "oprd", "pobj"]
ADJECTIVES = [
    "acomp",
    "advcl",
    "advmod",
    "amod",
    "appos",
    "nn",
    "nmod",
    "ccomp",
    "complm",
    "hmod",
    "infmod",
    "xcomp",
    "rcmod",
    "poss",
    " possessive",
]
ADVERBS = ["advmod"]
COMPOUNDS = ["compound"]
PREPOSITIONS = ["prep"]


# In[53]:


def getSubsFromConjunctions(subs):
    moreSubs = []
    for sub in subs:
        # rights is a generator
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreSubs.extend(
                [tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"]
            )
            if len(moreSubs) > 0:
                moreSubs.extend(getSubsFromConjunctions(moreSubs))
    return moreSubs


# In[54]:


def getObjsFromConjunctions(objs):
    moreObjs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreObjs.extend(
                [tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"]
            )
            if len(moreObjs) > 0:
                moreObjs.extend(getObjsFromConjunctions(moreObjs))
    return moreObjs


# In[55]:


def getVerbsFromConjunctions(verbs):
    moreVerbs = []
    for verb in verbs:
        rightDeps = {tok.lower_ for tok in verb.rights}
        if "and" in rightDeps:
            moreVerbs.extend(
                [tok for tok in verb.rights if tok.pos_ == "VERB"])
            if len(moreVerbs) > 0:
                moreVerbs.extend(getVerbsFromConjunctions(moreVerbs))
    return moreVerbs


# In[56]:


def findSubs(tok):
    head = tok.head
    # print("head.head: ",head.head)
    while head.pos_ != "VERB" and head.pos_ != "NOUN" and head.head != head:
        head = head.head
    if head.pos_ == "VERB":
        subs = [tok for tok in head.lefts if tok.dep_ == "SUB"]
        if len(subs) > 0:
            verbNegated = isNegated(head)
            subs.extend(getSubsFromConjunctions(subs))
            return subs, verbNegated
        elif head.head != head:
            return findSubs(head)
    elif head.pos_ == "NOUN":
        return [head], isNegated(tok)
    return [], False


# In[57]:


def isNegated(tok):
    negations = {"no", "not", "n't", "never", "none"}
    for dep in list(tok.lefts) + list(tok.rights):
        if dep.lower_ in negations:
            return True
    return False


# In[58]:


def find_negation(tok):
    negations = {"no", "not", "n't", "never", "none"}
    for dep in list(tok.lefts):
        if dep.lower_ in negations:
            verb = dep.lower_ + " " + tok.lemma_
            verb_id = [dep.i, tok.i]
            return verb, verb_id
    #     for dep in list(tok.rights):
    #         if dep.lower_ in negations:
    #             return tok.lemma_ +" "+ dep.lower_
    verb = tok.lemma_
    verb_id = [tok.i]
    return verb, verb_id


# In[59]:


def findSVs(tokens):
    svs = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB"]
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        if len(subs) > 0:
            for sub in subs:
                svs.append(
                    (sub.orth_, "!" + v.orth_ if verbNegated else v.orth_))
    return svs


# In[60]:


def getObjsFromPrepositions(deps):
    objs = []
    # print("deps are: ", deps)
    for dep in deps:
        # print("For ",dep, "pos: ",dep.pos_," and dep: ",dep.dep_)
        # if
        if dep.pos_ == "ADP" and (dep.dep_ == "prep" or dep.dep_ == "agent"):
            # print("dep.rights are ",list(dep.rights))
            for tok in dep.rights:
                if (tok.pos_ == "NOUN" and tok.dep_ in OBJECTS) or (
                    tok.pos_ == "PRON" and tok.lower_ == "me"
                ):
                    objs.append(tok)
                elif tok.dep_ == "pcomp":
                    for t in tok.rights:
                        if (t.pos_ == "NOUN" and t.dep_ in OBJECTS) or (
                            t.pos_ == "PRON" and t.lower_ == "me"
                        ):
                            objs.append(t)
                else:
                    objs.extend(getObjsFromPrepositions(tok.rights))
            # objs.extend([tok for tok in dep.rights if tok.dep_  in OBJECTS or (tok.pos_ == "PRON" and tok.lower_ == "me")])
    return objs


# In[61]:


def getAdjectives(toks):
    toks_with_adjectives = []
    for tok in toks:
        adjs = [left for left in tok.lefts if left.dep_ in ADJECTIVES]
        adjs.append(tok)
        adjs.extend([right for right in tok.rights if tok.dep_ in ADJECTIVES])
        tok_with_adj = " ".join([adj.lower_ for adj in adjs])
        toks_with_adjectives.extend(adjs)
    return toks_with_adjectives


# In[62]:


def getObjsFromAttrs(deps):
    for dep in deps:
        if dep.pos_ == "NOUN" and dep.dep_ == "attr":
            verbs = [tok for tok in dep.rights if tok.pos_ == "VERB"]
            if len(verbs) > 0:
                for v in verbs:
                    rights = list(v.rights)
                    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
                    objs.extend(getObjsFromPrepositions(rights))
                    if len(objs) > 0:
                        return v, objs
    return None, None


# In[63]:


def getObjFromXComp(deps):
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(getObjsFromPrepositions(rights))
            if len(objs) > 0:
                return v, objs
    return None, None


# In[64]:


def getAllSubs(v):
    verbNegated = isNegated(v)
    # print("For ", v ," v.lefts are ", list(v.lefts))
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    # print("getAllSubs for ",v, subs )
    if len(subs) > 0:
        subs.extend(getSubsFromConjunctions(subs))
    else:
        foundSubs, verbNegated = findSubs(v)
        subs.extend(foundSubs)
    return subs, verbNegated


# In[65]:


def getAllObjs(v):
    # rights is a generator
    rights = list(v.rights)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
    # print("For ",v," rights are ",rights," and objs are ",objs)
    objs.extend(getObjsFromPrepositions(rights))

    potentialNewVerb, potentialNewObjs = getObjFromXComp(rights)
    if (
        potentialNewVerb is not None
        and potentialNewObjs is not None
        and len(potentialNewObjs) > 0
    ):
        objs.extend(potentialNewObjs)
        v = potentialNewVerb
    if len(objs) > 0:
        objs.extend(getObjsFromConjunctions(objs))
    else:
        # print("No OBJECTS")
        objs.extend(getObjsFromVerbConj(v))
    return v, objs


# In[66]:


def getAllObjsWithAdjectives(v):
    # rights is a generator
    rights = list(v.rights)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
    # print("For ",v," rights are ",rights," and objs are ",objs)
    if len(objs) == 0:
        objs = [tok for tok in rights if tok.dep_ in ADJECTIVES]
    objs.extend(getObjsFromPrepositions(rights))
    potentialNewVerb, potentialNewObjs = getObjFromXComp(rights)
    if (
        potentialNewVerb is not None
        and potentialNewObjs is not None
        and len(potentialNewObjs) > 0
    ):
        objs.extend(potentialNewObjs)
        v = potentialNewVerb
    if len(objs) > 0:
        objs.extend(getObjsFromConjunctions(objs))
    else:
        # print("No OBJECTS")
        objs.extend(getObjsFromVerbConj(v))
    return v, objs


# In[67]:


def getObjsFromVerbConj(v):
    objs = []
    rights = list(v.rights)
    # print("v.rights :", rights)
    for right in rights:
        if right.dep_ == "conj":
            subs, verbNegated = getAllSubs(right)
            objs.extend(subs)
        else:
            objs.extend(getObjsFromVerbConj(right))
    return objs


def findSVOs(tokens, len_doc):
    svos = []
    svo_token_ids = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB" and tok.dep_ != "aux"]
    # print("Verbs: ",verbs," size is: ",len(verbs))
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        verb, verb_id = find_negation(v)
        # print("For ",v," subs are ",subs)
        # if no subs, don't examine this verb any longer
        if len(subs) > 0:
            v, objs = getAllObjs(v)
            # print("For ",v," objs are ",objs)
            for sub in subs:
                for obj in objs:
                    sub_compound = generate_compound(sub)
                    obj_compound = generate_compound(obj)

                    sub_flag, sub_tag = check_tag(sub_compound)
                    obj_flag, obj_tag = check_tag(obj_compound)

                    if obj_flag and sub_flag:
                        event = (sub_tag, verb, obj_tag)
                    elif obj_flag:
                        event = (
                            " ".join(tok.lemma_ for tok in sub_compound),
                            verb,
                            obj_tag,
                        )
                    elif sub_flag:
                        event = (
                            sub_tag,
                            verb,
                            " ".join(tok.lemma_ for tok in obj_compound),
                        )
                    else:
                        event = (
                            " ".join(tok.lemma_ for tok in sub_compound),
                            verb,
                            " ".join(tok.lemma_ for tok in obj_compound),
                        )

                    svos.append(event)

    return svos, svo_token_ids


# In[23]:


def findSVAOs(tokens):
    svos = []
    verbs = [tok for tok in tokens if tok.pos_ == "VERB" and tok.dep_ != "aux"]
    # print("Verbs: ",verbs)
    for v in verbs:
        subs, verbNegated = getAllSubs(v)
        # if no subs, don't examine this verb any longer
        # print("For ",v," subs are ",subs)
        if len(subs) > 0:
            v, objs = getAllObjsWithAdjectives(v)
            # print("For ",v," objs are ",objs)
            for sub in subs:
                for obj in objs:
                    objNegated = isNegated(obj)
                    obj_desc_tokens = generate_left_right_adjectives(obj)
                    sub_compound = generate_compound(sub)
                    # verb_compound = generate_verb_advmod(v)
                    svos.append(
                        (
                            " ".join(tok.lower_ for tok in sub_compound),
                            v.lower_,
                            " ".join(tok.lower_ for tok in obj_desc_tokens),
                        )
                    )
    return svos


def check_tag(compound):
    flag = False
    res = ""
    for token in compound:
        if token.ent_type_ == "PERSON":
            flag = True
            res = "<NAME>"
            # print(token.text,"----",res)
            break
        elif token.ent_type_ == "ORG":
            flag = True
            res = "<ORG>"
            # print(token.text,"----",res)
            break
    return flag, res


def generate_compound(token):
    token_compunds = []
    for tok in token.lefts:
        if tok.dep_ in COMPOUNDS:
            token_compunds.extend(generate_compound(tok))
    token_compunds.append(token)
    for tok in token.rights:
        if tok.dep_ in COMPOUNDS:
            token_compunds.extend(generate_compound(tok))
    return token_compunds


def generate_verb_advmod(v):
    v_compunds = []
    for tok in v.lefts:
        if tok.dep_ in ADVERBS:
            v_compunds.extend(generate_verb_advmod(tok))
    v_compunds.append(v)
    for tok in v.rights:
        if tok.dep_ in ADVERBS:
            v_compunds.extend(generate_verb_advmod(tok))
    return v_compunds


# In[74]:


def generate_left_right_adjectives(obj):
    obj_desc_tokens = []
    for tok in obj.lefts:
        if tok.dep_ in ADJECTIVES:
            obj_desc_tokens.extend(generate_left_right_adjectives(tok))
    obj_desc_tokens.append(obj)
    for tok in obj.rights:
        if tok.dep_ in ADJECTIVES:
            obj_desc_tokens.extend(generate_left_right_adjectives(tok))
    return obj_desc_tokens


# In[75]:


single_words = ["a", "A", "<", ">", "i", "I"]


def remove_special_characters(text):
    regex = re.compile("[^a-zA-Z<>.\s]")
    text_returned = re.sub(regex, " ", text)
    tokens = text_returned.split()
    words = []
    for word in tokens:
        if len(word) > 1 or word in single_words:
            # stemming and removing stopwords from the tokens
            words.append(word)
    out = " ".join(words)
    # print("count of ! is : ", out.count("!"))
    return " ".join(words)


global_event_dict = dict()
global_event_line_dict = dict()


def insert_event_in_global_dict(file_id, i, line, SVO):
    for eve in SVO:
        # here eve is a tuple so converting into string
        eve = " ".join(eve)
        if eve not in global_event_line_dict:
            global_event_line_dict[eve] = {file_id: {i: line}}
        elif file_id not in global_event_line_dict[eve]:
            global_event_line_dict[eve][file_id] = {i: line}
        else:
            global_event_line_dict[eve][file_id][i] = line


def events_extraction(content, file_id):

    content = preprocess(content)
    # Define the pattern for sentence splitting
    pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'
    # Split the text into sentences using the pattern
    content_sents = re.split(pattern, content)
    # content_sents = content.split(".")
    file_svo = []
    file_svo_text = []
    len_doc = 0
    lines = []
    for i, line in enumerate(content_sents):
        line = line.strip()
        lines.append(remove_special_characters(line))

    for i, doc in enumerate(nlp.pipe(lines)):
        SVO, SVO_Token_IDs = findSVOs(doc, len_doc)
        if len(SVO) > 0:
            file_svo.append(SVO)
            insert_event_in_global_dict(file_id, i, lines[i], SVO)
            for eve in SVO:
                file_svo_text.append(" ".join(eve))

    return file_svo, file_svo_text, lines


if __name__ == "__main__":
    # load input_details.json
    with open('input_details.json') as f:
        input_details = json.load(f)
    # name of the dataset #train-test-dev #dir of files
    #"ilpcr", "train", "candidate"
    dataset, split_type, files_type = input_details[
        "dataset"], input_details["split_type"], input_details["files_type"]
    # path for data files
    input_root = input_details["input_root"]
    roots = input_root+"/"+dataset+"/"+split_type+"/"+files_type+"/"
    # path to store files with events
    output_root = input_details["output_root"]
    outpath = output_root+"/"+dataset+"/"+split_type+"/"+files_type+"/"
    os.makedirs(outpath, exist_ok=True)

    sent_data_path = output_root+"/"+dataset+"/"+split_type+"/" + \
        "sent_data_"+split_type+"_"+dataset+"_"+files_type+".sav"
    segment_dictionary_path = output_root+"/"+dataset+"/"+split_type+"/" + \
        "segment_dictionary_"+split_type+"_"+dataset+"_"+files_type+".sav"
    event_doc_line_path = output_root+"/"+dataset+"/"+split_type+"/" + \
        "event_doc_line_text_"+split_type+"_"+dataset+"_"+files_type+".pkl"

    file_lst = os.listdir(roots)
    print("Number of files: ", len(file_lst))

    contents = []
    for file in tqdm(file_lst):
        # print("Started for file: ", file)
        file_name = file.split(".txt")[0]
        file_id = int(file_name)
        file_path = os.path.join(roots, file)
        try:
            f = open(file_path, "r", encoding="utf-8")
            content = f.read()
            f.close()
        except Exception as e:
            print("Except 1 for file name: " + file_path + str(e) + "\n")

        contents.append(content)

    file_svo_lst = []
    file_sent_lst = []
    file_segment_dictionary = {"dict_"+files_type: {}}
    for i in tqdm(range(len(contents))):
        file_id = int(file_lst[i].split(".txt")[0])
        file = contents[i]
        file_svo, file_svo_text, lines = events_extraction(file, file_id)
        file_svo_lst.append(file_svo)
        file_segment_dictionary["dict_"+files_type][file_id] = file_svo_text
        file_sent_lst.append(lines)

    file_sent_dict = {files_type+"_data": {}}
    for i, file in enumerate(file_lst):
        file_sent_dict[files_type +
                       "_data"][int(file.split(".txt")[0])] = file_sent_lst[i]
        json_obj = json.dumps(file_svo_lst[i], indent=4)
        with open(outpath + file.split(".txt")[0] + ".json", "w", encoding="utf-8") as outfile:
            outfile.write(json_obj)

    with open(sent_data_path, 'wb') as f:
        pickle.dump(file_sent_dict, f)

    with open(segment_dictionary_path, 'wb') as f:
        pickle.dump(file_segment_dictionary, f)

    with open(event_doc_line_path, 'wb') as f:
        pickle.dump(global_event_line_dict, f)
