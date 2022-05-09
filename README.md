# cs5293sp22-project3

Name: Sai Prakash Davuluri

Goal: The aim of the project is to train the pre-defined machine learning model on training data and predict using testing data. Finally, we have to print the precision_score, recall_score, f1_score for the dataset.

### How to run the application

#### steps:
* First clone the project using : git clone '_github_project_url_'
* check to see if there is a directory named as 'cs5293sp22-project3' using ls command
* open the directory using cd cs5293sp22-project3
* Run the below command to start the project:

                        
                     pipenv run python project3.py

### WEB OR EXTERNAL LIBRARIES:
* pandas
* sklearn
* joblib
* nltk

### INSTALLATION OF ABOVE LIBRARIES:
* pipenv install pandas
* pipenv install scikit-learn
* pipenv install joblib
* pipenv install nltk

### Assumptions or Bugs

* I have defined the url to '_unredactor.tsv_' file in the program. So, if the location of that file change or access to that repository becomes private then it is not possible to train and test the model.


### Functions and approach

predictor()

* This function acts as a base layer for my program where it makes function calls to perform operations and to produce results.

* I have pre-defined the url in this method to retrieve the raw data from the GitHub repository and passed it as input parameter to read_tsv() method.

* After making different function calls to perform operations like data extraction from tsv file, performing normalization on training and testing data, and using pre-defined machine learning model to predict the output.

* Finally, I am printing the output to the console for the user to view.

read_tsv(url)

* This method takes the pre-defined url and reads the file in csv format. After reading the data into dataframe, I am assigning column names to the dataframe.

* I have deleted the column 'person' as I don't need that information and dropping any rows with null values.

* To overcome the error  i.e., pre-defined model throws when it encounters an unseen label. I am label encoding all the names in the dataframe and adding it as the column.

extract_sentences(dataFrame)

* This method accepts the dataframe as input parameter. Using for loop, I am segregating the data storing in two different list.

* The data labeled as 'training' and 'validation' will be stored in temp_redacted_sentences and data labeled as 'testing' will be stored in temp_test_redacted_sentences.

* The names labeled as 'training' and 'validation' will be stored in names list and names labeled as 'testing' will be stored in test_names.

normalize_sentences(temp_redacted_sentences, temp_test_redacted_sentences)

* In this method, I am performing normalization on both types of data.

* I have used pre-defined stopwords module from nltk.corpus library and stored them in a list. Also, added few more words to that list to further clean the data.

* Then I am converting every word in a sentence into lower case and added them to the list.

* I am also considering the redaction part(unicode block) as unwanted word/data in a sentence and removing it.


model_definition(names, test_names, _sentences, _test_sentences)

* I am using make_union module from sklearn.pipeline library to unionize two different vectorizers called CountVectorizer and TfidfVectorizer.

* I am using joblib library to save the machine learning model because there is no need to dynamically add the features to model at the time of processing.

* I am loading the ML model which was saved using joblib.load(fileName) method and calculating the precision_score, recall_score, and f1_score.

* Finally, returning the these parameters as output to the predictor method to print and display them to user.


### Test cases

* Test case to test above functionalities are available in tests folder.

* Command to run test case: pipenv run python -m pytest

* I have taken sample tsv file called unredactor_test.tsv which is a smaller version of unredactor.tsv to reduce the burden on the machine.

test_model_definition():

* To test every functionality in the project, I have written one test case that effectively checks to see if every function is satisfied. 

* I have pre-defined the filename and passed it as input to read_tsv(file_name) method. The output that I am getting from the read_tsv(file_name) method, is passed as input parameter to extract_sentences(dataFrame) method.

* I am following the same process for normalize_sentences(temp_redacted_sentences, temp_test_redacted_sentences) and model_definition(names, test_names, _sentences, _test_sentences) methods.

* Finally, I am using assert statements to test the output of each functionality like if type of the variable matched or if the output variable is not None, etc.

### GitHub:
The above-mentioned files need to be added, committed, and pushed to GitHub repository by using the following commands.

git add file-name;

git commit -m "commit message"

git push origin main



