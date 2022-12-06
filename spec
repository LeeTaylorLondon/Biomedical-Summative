Use a variety of machine learning techniques that classify the images.
* Manually build and tweak a CNN model
* Perform transfer learning with VGGNet
* Perform transfer learning with ResNet
* Create images with GAN and investigate similar options
* Consider and experiment with other machine learning techniques


...


IMAGES - you will build deep learning classification models for MRI images

What to submit?
•The report (a single PDF file, max. 2000 words),
•Python scripts for data processing, visualisation, and classification ( tar.gz , or zip archive file).

The deadline for submission is 2022-12-16 16:30. You must submit the files on Canvas.
Late submissions (up to 1 week) will be penalised with −5points per day.

Before you start
Make sure to use the recommended project directory layout with data , scripts , and results directories.
All scripts should run from the project directory level, for example as ./scripts/task-1.py .
You don’t have to copy the data. Just use the relative paths directly in your scripts (e.g. data/visits.csv ).

When you are done, turn all the steps into a single preprocessing.py script, that generates a new
processed.pkl.gz file in the project data folder. Use this new data file in further analysis.

There are many possible approaches to each problem, and it is unclear which would bring the best results.
Therefore, you will have to experiment!

All scripts must be reproducible, so if randomness is involved, fix the seed of the random generator.

Did all the methods behave as expected or did the results surprise you? Was prediction as good on the test
folds as on the training folds? Please discuss in the report.

MARKING CRITERIA
Exploratory analysis 10%
Data pre-processing 10%
Methods 30%
Results 20%
Discussion/Conclusions 20%
Form (writing style, use of figures, etc.) 10%

COMMENTS SECTION
Add a final Comments section to your report. Use it to answer the following questions:
•If there are known problems with your code, please list them, and explain how you might have fixed
them if you had more time. We are likely to give you partial credit for it, if we see you understand the
problems well.
•Did you discuss the final report with other students? If so, let us know who you talked to and how you
helped each other.
•Which of the recommended additional readings you found to be most helpful throughout the course,
and in completing the report? Please make a list.
•Do you have any comments about this assignment that you would like to share? Was it too long or too
hard? Were all the requirements clear? Did you have fun working on it, or did you hate it? Do you
think you learned something while doing it, or was it a waste of time? Constructive feedback will be
highly appreciated! Be as open as you want, it will not affect your mark.

REPORT STRUCTURE
To format the report, use a structure shown below.
Focus on the core tasks of applying machine learning methods to analyse
the data. Describe your initial approach to the problem and all further
refinements. Provide justification for the choices made (e.g., how to handle
missing data, or which clustering algorithm to use). Finally, do not only
report the results but also interpret them, that is, describe what your
analysis has revealed about the data or methods.
The report should not be longer than 2k words (excluding figure/table
captions, references, and comments) with 10% tolerance.

REPORT STRUCTURE
1. Exploratory analysis
2. Pre-processing
3. Machine learning
3.1. Method A
3.2. Method B
...
4. Conclusions
5. Comments