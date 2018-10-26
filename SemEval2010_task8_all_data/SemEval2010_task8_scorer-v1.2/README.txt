Included are two tools for SemEval-2010 Task #8:
Multi-Way Classification of Semantic Relations Between Pairs of Nominals

The task is described on the following Web address:
	http://docs.google.com/View?id=dfvxd49s_36c28v9pmw


1. Output File Format Checker
-----------------------------

This is an official output file format checker for SemEval-2010 Task 8.

Use:
   semeval2010_task8_format_checker.pl <PROPOSED_ANSWERS>

Examples:
   semeval2010_task8_format_checker.pl proposed_answer1.txt
   semeval2010_task8_format_checker.pl proposed_answer2.txt
   semeval2010_task8_format_checker.pl proposed_answer3.txt
   semeval2010_task8_format_checker.pl proposed_answer4.txt
   semeval2010_task8_format_checker.pl proposed_answer5.txt

  In the examples above, the first three files are OK, while the last one contains four errors.
  And answer_key2.txt contains the true labels for the *training* dataset.

Description:
   The scorer takes as input a proposed classification file,
   which should contain one prediction per line in the format
   "<SENT_ID>	<RELATION>"
   with a TAB as a separator, e.g.,
         1	Component-Whole(e2,e1)
         2	Other
         3	Instrument-Agency(e2,e1)
             ...
   The file does not have to be sorted in any way.
   Repetitions of IDs are not allowed.

   In case of problems, the checker outputs the problemtic line and its number.
   Finally, the total number of problems found is reported
   or a message is output saying that the file format is OK.

   Participants are expected to check their output using this checker before submission.

Last modified: March 10, 2010



2. Scorer
---------

This is the official scorer for SemEval-2010 Task #8.

Last modified: March 22, 2010

Current version: 1.2

Revision history:
  - Version 1.2 (fixed a bug in the precision for the scoring of (iii))
  - Version 1.1 (fixed a bug in the calculation of accuracy)

Use:
   semeval2010_task8_scorer-v1.1.pl <PROPOSED_ANSWERS> <ANSWER_KEY>

Examples:
   semeval2010_task8_scorer-v1.2.pl proposed_answer1.txt answer_key1.txt > result_scores1.txt
   semeval2010_task8_scorer-v1.2.pl proposed_answer2.txt answer_key2.txt > result_scores2.txt
   semeval2010_task8_scorer-v1.2.pl proposed_answer3.txt answer_key3.txt > result_scores3.txt
   semeval2010_task8_scorer-v1.2.pl proposed_answer5.txt answer_key5.txt > result_scores5.txt

Description:
   The scorer takes as input a proposed classification file and an answer key file.
   Both files should contain one prediction per line in the format "<SENT_ID>	<RELATION>"
   with a TAB as a separator, e.g.,
         1	Component-Whole(e2,e1)
         2	Other
         3	Instrument-Agency(e2,e1)
             ...
   The files do not have to be sorted in any way and the first file can have predictions
   for a subset of the IDs in the second file only, e.g., because hard examples have been skipped.
   Repetitions of IDs are not allowed in either of the files.

   The scorer calculates and outputs the following statistics:
      (1) confusion matrix, which shows
         - the sums for each row/column: -SUM-
         - the number of skipped examples: skip
         - the number of examples with correct relation, but wrong directionality: xDIRx
         - the number of examples in the answer key file: ACTUAL ( = -SUM- + skip + xDIRx )
      (2) accuracy and coverage
      (3) precision (P), recall (R), and F1-score for each relation
      (4) micro-averaged P, R, F1, where the calculations ignore the Other category.
      (5) macro-averaged P, R, F1, where the calculations ignore the Other category.

   Note that in scores (4) and (5), skipped examples are equivalent to those classified as Other.
   So are examples classified as relations that do not exist in the key file (which is probably not optimal).

   The scoring is done three times:
     (i)   as a (2*9+1)-way classification
     (ii)  as a (9+1)-way classification, with directionality ignored
     (iii) as a (9+1)-way classification, with directionality taken into account.
   
   The official score is the macro-averaged F1-score for (iii).
