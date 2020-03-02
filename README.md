# Meaning and Computation Project

this project includes the files: 
<ul>
<li>corpus.py - creates co-occurence matrices from Brown Corpus using nltk library.</li>
<li>project.py - creates semantic spaces from co-occurence matrices and trains models
			to learn a composed meaning, it uses the DISSECT toolkit.</li>
<li>results.txt - all neighbors of VNs and their similarity scores </li>
<li>train data directory - includes a train data file for each one of the vbn_functions
						with nouns that appear in the corpus after this vbn,
						as well as a test data file in the same format with our own 
						intuitive noun examples.</li>
</ul>
files for creating the spaces from matrices:
<ul>
<li>core.cols - list of columns of the core matrix</li>
<li>core.rows - list of rows of the core matrix</li>
<li>core.dm -  all core matrix saved in dense format, without the header (names of columns)</li>
<li>header_core.dm -  all core matrix saved in dense format, with header</li>
<li>per.cols - list of columns of the peripheral matrix</li>
<li>per.rows - list of rows of the peripheral matrix</li>
<li>per.dm -  all peripheral matrix saved in dense format, without the header (names of columns)</li>
<li>header_per.dm -  all peripheral matrix saved in dense format, with header</li>
</ul>
	
pkl files:
<ul>
<li>core.pkl - core semantic space saved in pickle format</li>
<li>reduced_core.pkl - reduced core semantic space (after ppmi and avd reduction) saved in pickle format</li>
	<li>per.pkl - peripheral semantic space built under the reduced space, saved in pickle format</li>
</ul>
						
