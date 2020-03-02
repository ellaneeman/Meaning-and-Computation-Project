from nltk.corpus import wordnet as wn
import pprint
from nltk.corpus import stopwords
import string

# DISSECT imports
from composes.semantic_space.space import Space
from composes.semantic_space.peripheral_space import PeripheralSpace
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from composes.transformation.dim_reduction.svd import Svd
from composes.transformation.feature_selection.top_feature_selection import TopFeatureSelection
from composes.utils import io_utils, log_utils
from composes.composition.lexical_function import LexicalFunction
from composes.utils import regression_learner
from composes.similarity.cos import CosSimilarity
import pickle

#######################################
# downloads
#######################################
# import nltk
# nltk.download('wordnet')


#######################################
# globals
#######################################
core_cooccurrence_file = "core.dm"
core_row_file = "core.rows"
core_col_file = "core.cols"
core_space_file = "core.pkl"
reduced_core_space_file = "reduced_core.pkl"
per_cooccurrence_file = "per.dm"
per_row_file = "per.rows"
per_col_file = "per.cols"
per_space_file = "per.pkl"
training_data_file = "train_data.txt"
PUNCTUATION = set(string.punctuation).union({"\'\'", "``"})
STOPWORDS = set(stopwords.words('english')).union(PUNCTUATION)
NUM_OF_EXAMPLES = 3
participle_verbs_to_examples = {}


def _extract_pair(vbn, example):
    """helper function for collect_examples function. """
    example_words = example.split(" ")
    index = example_words.index(vbn)
    noun = example_words[index + 1].strip("\"-").strip("(")
    composed_exp = vbn + "_" + noun
    return noun, composed_exp


def collect_examples(to_print=False):
    """ Finds verbs from WordNet thesaurus that appear also as adjectivs, but not as nouns or adverbs.
    collects WordNet examples (under the adjective role of the word) into the global dictionary
    participle_verbs_to_examples"""
    for word in wn.all_synsets():
        name = word.name()[:-5]
        synsets = wn.synsets(name)
        pos = set([word.pos() for word in synsets])
        if (('a' in pos) or ('s' in pos)) and ('v' in pos):
            if ('n' not in pos) and ('r' not in pos):
                adj_synsets = [s for s in synsets if s.name()[-4] in ['a', 's']]
                # collect examples for adjectives
                adj_examples = []
                for adj in adj_synsets:
                    if len(adj.examples()) > 0:
                        adj_examples.extend(adj.examples())
                examples = [example for example in adj_examples if " " + name + " " in example]
                if len(examples) > 1:
                    participle_verbs_to_examples[name] = examples

    # save the vbns list to be loaded in the corpus code file
    with open('our_vbns', 'wb') as fp_vbns:
        pickle.dump(participle_verbs_to_examples.keys(), fp_vbns)

    # save the vbns with their noun examples into a file in the train data format
    with open('train_data.txt', 'wb') as fp_data:
        for vbn in participle_verbs_to_examples.keys():
            if len(participle_verbs_to_examples[vbn]) > NUM_OF_EXAMPLES:
                for example in participle_verbs_to_examples[vbn]:
                    noun, composed_exp = _extract_pair(vbn, example)
                    if noun not in STOPWORDS:
                        fp_data.write(vbn + "_function\t" + noun + "\t" + composed_exp + "\n")
    if to_print:
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(participle_verbs_to_examples)
        print len(participle_verbs_to_examples.keys())


def main():

    # run this code for constructing the spaces from scratch

    # # building semantic space from co-occurrence counts
    # core_space = Space.build(data=core_cooccurrence_file, rows=core_row_file,
    #                          cols=core_col_file, format="dm")
    # # saving the semantic space
    # io_utils.save(core_space, core_space_file)
    # "applying ppmi weighting"
    # core_space = core_space.apply(PpmiWeighting())
    # # applying svd 500
    # reduced_core_space = core_space.apply(Svd(500))
    # # saving the semantic space
    # io_utils.save(reduced_core_space, reduced_core_space_file)
    #
    # # building peripheral space
    # per_space = PeripheralSpace.build(reduced_core_space, data=per_cooccurrence_file, cols=per_col_file,
    #                                   rows=per_row_file, format="dm")
    # # saving peripheral space
    # io_utils.save(per_space, per_space_file)

    # run this code for loading the saved spaces
    reduced_core_space = io_utils.load(reduced_core_space_file, Space)
    per_space = io_utils.load(per_space_file)

    # learning phase
    vbn_matrices_to_learn = ["broken", "changed", "closed", "colored", "detailed", "expressed", "given", "improved"]
    train_data = {}
    test_data = {}
    comp_models = {}
    composed_spaces = {}

    # read training and testing data
    for participle_verb in vbn_matrices_to_learn:
        train_data[participle_verb] = io_utils.read_tuple_list("train_data\\" + participle_verb + "_train_data.txt",
                                                               fields=[0, 1, 2])
        test_data[participle_verb] = io_utils.read_tuple_list("train_data\\" + participle_verb + "_test_data.txt",
                                                              fields=[0, 1, 2])

    # train each one of the matrices by the examples from the training data
    for participle_verb in vbn_matrices_to_learn:
        comp_models[participle_verb] = LexicalFunction(learner=regression_learner.RidgeRegressionLearner(param=2))
        comp_models[participle_verb].train(train_data[participle_verb], reduced_core_space, per_space)
        # create composed VNs in the "composed_space" for each of the nouns in the test data
        composed_spaces[participle_verb] = comp_models[participle_verb].compose(test_data[participle_verb],
                                                                                reduced_core_space)
    # evaluation phase - the printed output is saved in the results file
    for comp_space in composed_spaces.values():
        for pair in comp_space.get_row2id().keys():
            print pair + " neighbors:"
            print comp_space.get_neighbours(pair, 10, CosSimilarity(), per_space)
        print "\n"

if __name__ == "__main__":
    main()
