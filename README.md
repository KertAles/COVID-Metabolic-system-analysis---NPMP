# Analysis of metabolic models of cells infected with COVID-19

The goal of this project was to analyse the metabolic models of healthy and COVID-19 infected cells. The analysis was approached as a classification problem. Using machine learning methods, namely the Gaussian naive Bayes classifiers and decision tree classifiers, 5 types of cells were classified into their classes, based cell reaction fluxes, acquired using 4 different models. We tried a number of experimental setups - the first where all reactions were known, the second where only one reaction is know, and the third, similar to the second, but the reactions are grouped by subsystems. The first setup achieved a 100\% accuracy score on all cell/model combinations. The second and third setup were evaluated with regards to the proportion of significantly changed reactions (SCRs), which are reaction capable of producing a classifier with a 100\% accuracy score on the test set. These proportions were grouped by different cells and models, and plotted using boxes with whiskers.
