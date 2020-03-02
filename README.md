#Meaning and Computation Project

this project includes the files:
corpus.py - creates co-occurence matrices from Brown Corpus using nltk library.
project.py - creates semantic spaces from co-occurence matrices and trains models
			to learn a composed meaning, it uses the DISSECT toolkit.
results.txt - all neighbors of VNs and their similarity scores
train data directory - includes a train data file for each one of the vbn_functions
						with nouns that appear in the corpus after this vbn,
						as well as a test data file in the same format with our own 
						intuitive noun examples.

files for creating the spaces from matrices:
core.cols - list of columns of the core matrix
core.rows - list of rows of the core matrix
core.dm -  all core matrix saved in dense format, without the header (names of columns)
header_core.dm -  all core matrix saved in dense format, with header
per.cols - list of columns of the peripheral matrix
per.rows - list of rows of the peripheral matrix
per.dm -  all peripheral matrix saved in dense format, without the header (names of columns)
header_per.dm -  all peripheral matrix saved in dense format, with header

pkl files:
core.pkl - core semantic space saved in pickle format
reduced_core.pkl - reduced core semantic space (after ppmi and avd reduction) saved in pickle format
per.pkl - peripheral semantic space built under the reduced space, saved in pickle format
											
						
