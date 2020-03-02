import nltk
from nltk.corpus import brown
from nltk.corpus import stopwords
import pandas as pd
import string
from collections import defaultdict
import copy

#######################################
# downloads
#######################################

# nltk.download('brown')
# print dir(brown)

# import nltk
# nltk.download('stopwords')
#

#######################################
# WordNet vbns for collecting examples
#######################################
# with open('our_vbns', 'rb') as fp:
#     wordnet_vbns = set(pickle.load(fp))


#######################################
# globals
#######################################
PUNCTUATION = set(string.punctuation).union({"\'\'", "``"})
STOPWORDS = set(stopwords.words('english')).union(PUNCTUATION)
wsj = brown.tagged_words(categories='fiction')
cfd_words = nltk.ConditionalFreqDist((tag, word) for (word, tag) in wsj)
vbns = list(set(cfd_words['VBN']))

all_nouns = list(set(cfd_words['NN'] + cfd_words['NNS']))
adjs = list(set(cfd_words['JJ']))
nouns = [noun for noun in all_nouns if noun not in STOPWORDS]
rows = vbns + nouns
cols = list(set(w for w in brown.words()))
per_rows = []
CORE_SENTS = brown.sents(categories='fiction')
PER_SENTS = brown.sents()[30000:46000]


def save_to_file(file_name, list):
    """ Saves a list of rows or cols to file, words are separated by \n"""
    with open(file_name, 'wb') as fp:
        for item in list:
            fp.write("%s\n" % item)


def cond_freq_dist(data, cols=[]):
    """ Takes a list of tuples and returns a conditional frequency distribution as a pandas dataframe. """
    def_cols_dict = {}
    for col in cols:
        def_cols_dict[col] = 0
    if cols:
        cfd = defaultdict(lambda: copy.deepcopy(def_cols_dict))
    else:
        cfd = defaultdict(lambda: defaultdict(int))
    for row, col in data:
        try:
            cfd[row][col] += 1
        except KeyError:
            try:
                cfd[row][col] = 1
            except KeyError:
                pass
                # if a word is not in the cols, ignore it
    return pd.DataFrame(cfd).fillna(0)


def create_per_rows():
    """ Concatenates vbns and nouns into pairs of vbn_noun separated by '_' and adds them to the global list per_rows"""
    for vbn in vbns:
        for noun in nouns:
            per_rows.append(vbn + "_" + noun)


def create_core_matrix():
    """ Splits the core sentences into bigram tuples and counts their frequencies (in a window of 1)"""
    bigrams = []
    for sent in CORE_SENTS:
        filtered_sent = [word for word in sent if word not in PUNCTUATION]
        bigrams.extend(list(nltk.bigrams(filtered_sent)))
    return cond_freq_dist(bigrams)


def create_per_matrix():
    """ Splits the peripheral sentences into trigram tuples, if the first two cells are vbn+noun or adjective+noun,
     it concatenates them into one phrase and returns a list of 2-elements-tuples to be sent for
     counting the phrases frequencies (in a window of 1)"""
    create_per_rows()
    trigrams = []
    for sent in PER_SENTS:
        filtered_sent = [word for word in sent if word not in PUNCTUATION]
        trigrams.extend(list(nltk.trigrams(filtered_sent)))
    filtered_trigrams = []
    for trigram in trigrams:
        if trigram[0] + "_" + trigram[1] in per_rows:
            filtered_trigrams.append((trigram[0] + "_" + trigram[1], trigram[2]))
        elif trigram[0] in adjs and trigram[1] in nouns:
            filtered_trigrams.append((trigram[0] + "_" + trigram[1], trigram[2]))
    return filtered_trigrams


def _repair_saved_matrix():
    header_core = pd.read_csv('header_core.dm', sep="\t")
    header_per = pd.read_csv('header_per.dm', sep="\t")
    per_cols = set(list(header_per.columns.values))
    core_cols = set(list(header_core.columns.values))
    cols_to_drop = per_cols.difference(core_cols)

    for name in cols_to_drop:
        header_per.drop(name, axis=1, inplace=True)

    print list(header_core.columns.values) == list(header_per.columns.values)
    header_per.to_csv('new_header_per.dm', sep="\t", index=False, header=True)
    header_per.to_csv('new_per.dm', sep="\t", index=False, header=False)


def main():
    # create core co-occurrence matrix
    core_mat = create_core_matrix()
    final_mat = core_mat.T
    final_mat_cols = list(final_mat.columns.values)
    final_mat_rows = list(final_mat.index.values)

    save_to_file("core.rows", final_mat_rows)
    save_to_file("core.cols", final_mat_cols)
    final_mat.to_csv('header_core.dm', sep="\t", index=True, header=True)
    final_mat.to_csv('core.dm', sep="\t", index=True, header=False)

    # create peripheral co-occurrence matrix with the core matrix columns
    per_mat = cond_freq_dist(create_per_matrix(), cols=final_mat_cols)
    final_per_mat = per_mat.T
    final_per_mat_cols = list(final_per_mat.columns.values)
    final_per_mat_rows = list(final_per_mat.index.values)

    # print final_per_mat_cols == final_mat_cols
    save_to_file("per.rows", final_per_mat_rows)
    save_to_file("per.cols", final_per_mat_cols)
    final_per_mat.to_csv('header_per.dm', sep="\t", index=True, header=True)
    final_per_mat.to_csv('per.dm', sep="\t", index=True, header=False)


if __name__ == "__main__":
    main()
