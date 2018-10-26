Test Data for SemEval-2 Task #8: Multi-Way Classification of Semantic Relations Between Pairs of Nominals

Iris Hendrickx, Su Nam Kim, Zornitsa Kozareva, Preslav Nakov, Diarmuid Ó Séaghdha, Sebastian Padó, Marco Pennacchiotti, Lorenza Romano and Stan Szpakowicz

The accompanying dataset is released under a Creative Commons Atrribution 3.0 Unported Licence (http://creativecommons.org/licenses/by/3.0/).

Version 1.0: March 18, 2010


SUMMARY

This test dataset consists of 2717 sentences that have been collected from the Web specifically for SemEval-2 Task #8.
The sentences do not overlap with the 8000 training sentences that have been released on March 5, 2010;
they also do not overlap with the sentences from SemEval-1 Task #4 (Classification of Semantic Relations between Nominals).


IMPORTANT

To use this test dataset, the participants should download from the Official SemEval-2010 website the following:

1. the training dataset (it also contains relation definitions and our annotation guidelines);
2. the official scorer and format checker.


INPUT DATA FORMAT

The format of the test data is illustrated by the following examples:

8001    "The most common <e1>audits</e1> were about <e2>waste</e2> and recycling."
8002    "The <e1>company</e1> fabricates plastic <e2>chairs</e2>."
8003    "The school <e1>master</e1> teaches the lesson with a <e2>stick</e2>."
8004    "The suspect dumped the dead <e1>body</e1> into a local <e2>reservoir</e2>."
....
10717    "A few days before the service, Tom Burris had thrown into Karen's <e1>casket</e1> his wedding <e2>ring</e2>."

Each line contains a sentence inside quotation marks, preceded by a numerical identifier. In each sentence, two entity mentions are tagged as e1 and e2 -- the numbering simply reflects the order of the mentions in the sentence. The span of the tag corresponds to the "base NP" which may be smaller than the full NP denoting the entity (see the annotation guidelines for details).


EVALUATION

The task is to predict, given a sentence and two tagged entities, which of the relation labels to apply. The predictions must be in the following format:

1 Content-Container(e2,e1)
2 Other
3 Entity-Destination(e1,e2)
...

There is a format checker released together with the scorer, which the participants should use to check their output before submitting their results.

The official evaluation measures are accuracy over all examples and macro-averaged F-score over the 9 relation labels apart from Other. To calculate the F-score, 9 individual F-scores -- one for each relation label -- are calculated in the standard way and the average of these scores is taken. See the README of the official scorer for more details.


TEST PROCEDURE

The participants can download the test dataset at any time up to the final results submission deadline (April 2, 2010). Once the data have been downloaded, participants will have 7 days to submit their results; they must also submit by the final deadline of April 2. Late submissions will not be counted. Participants should supply four sets of predictions for the test data, using four subsets of the training data:

TD1   training examples 1-1000
TD2   training examples 1-2000
TD3   training examples 1-4000
TD4   training examples 1-8000

For each training set, participants may use the data in that set for any purpose they wish (training, development, cross-validation and so forth). However, the training examples outside that set (e.g., 1001-8000 for TD1) may not be used in any way. The final 891 examples in the training release (examples 7110-8000) are taken from the SemEval-1 Task #4 datasets for relations 1-5 and hence their label distribution is skewed towards those relation classes.  Participants have the option of including or excluding these examples as appropriate for their chosen learning method. See the training data archive for details.

There is no restriction on the external resources that may be used.


SUBMISSION PROCEDURE

The participants should do a submission via the SemEval-2 website (shown below).

Each participating team should choose a short ID to be used in the official ranking. Ideally, the ID should be an abbreviation of the university, e.g., NUS for the National University of Singapore. Each team is allowed to submit multiple runs. In that case, a run ID should be augmented with run identification, e.g., NUS-WN, NUS-1, NUS-2, etc.

The submission should contain five files each starting with the above-described ID, e.g., if the ID is NUS, the files should be
- NUS_TD1.txt
- NUS_TD2.txt
- NUS_TD3.txt
- NUS_TD4.txt
- NUS_description.txt

The first four files should contain the classification decisions for test datasets TD1, TD2, TD3 and TD4, respectively.

The fifth file should contain the following information:
- ID of the team
- Names and affiliations of the participating team
- Contact person with email address: where we will send the results of the evaluation
- Short description of the approach for *each* run: one very short paragraph for each one that will help us in preparing the task overview paper
- Short description of the resources used for *each* run


USEFUL LINKS

Google group: http://groups.google.com.sg/group/semeval-2010-multi-way-classification-of-semantic-relations?hl=en
Task website: http://docs.google.com/View?docid=dfvxd49s_36c28v9pmw
SemEval-2 website: http://semeval2.fbk.eu/semeval2.php


TASK SCHEDULE

 * Test data release:                March 18, 2010
 * Result submission deadline:       7 days after downloading the *test* data, but no later than April 2
 * Organizers send the test results: April 10, 2010
 * Submission of description papers: April 17, 2010
 * Notification of acceptance:       May 6, 2010
 * SemEval-2 workshop (at ACL):      July 15-16, 2010
