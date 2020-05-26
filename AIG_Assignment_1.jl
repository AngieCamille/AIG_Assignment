# 217058345 - ANGELINE MUKAMANA

#Importing package for CSV files
#Importing package for manipulating DataFrames
using CSV, DataFrames

# Reading bank marketing campaign from local CSV file
bank = CSV.read("C:\\Users\\angie\\Documents\\School\\AIG\\bank-additional-full.csv")
# Reading bank marketing campaign from AngieCamille github
bank = CSV.read(download("https://raw.githubusercontent.com/AngieCamille/AIG_Assignment/master/bank-additional-full.csv"))

# Data cleaning

# Selecting banking details (age and duration) to work with
bank[[Symbol("age"),Symbol("duration")]]
#Converting the table to float
function convert_str_floated64(x)
    parse(Float64, x)
end
#Converting the dataset to a matrix
x = convert(Matrix,bank[[Symbol("age"),Symbol("duration")]])
# Selecting the clients that have and haven't placed a term deposit (identifying: y)
bank = CSV.read(download("https://raw.githubusercontent.com/AngieCamille/AIG_Assignment/master/bank-additional-full.csv"))
bank[:21]
# Clean the dataset and save it into another dataset containing only the features that you wish to use for the classication
# Reading cleaned dataset (CSV file from AngieCamille github)
bank = CSV.read(download("https://raw.githubusercontent.com/AngieCamille/AIG_Assignment/master/bank-additional-clean.csv"))

# Building the classifier

# Importing package for regularised logistic regression
# Importing package for calculations
using ScikitLearn
using Statistics

# Read the data and convert to array
bank = CSV.read(download("https://raw.githubusercontent.com/AngieCamille/AIG_Assignment/master/bank-additional-clean.csv"))
x = convert(Array,bank[:,[1,2]])
y = convert(Array,bank[:,[3]])
#Splitting the data into training and testing data
function partitionTrainTest(bank, y, train_perc = 0.7) # Training percentage is 70%
       n = size(bank,1) # n is size of dataset
       data = shuffle(1:n) # rearrange the data
       train_data = view(data, 1:floor(Int, at*n)) # train_data is the Training data
       test_data = view(data, (floor(Int, at*n)+1):n) # test_data is the Testing data
       bank[train_data, 1:end .!=y], bank[train_data, y], bank[test_data, 1:end .!=y], bank[test_data, y]
end
# Importing Logistic Regression
@sk_import linear_model: LogisticRegression
# Applying Logistic Regression to a variable
log_reg_model = LogisticRegression()
# fitting the model with Logistic Regression, dependent variables and target variables
fit!(log_reg_model, x, y)
# Finding the prediction
prediction = predict(log_reg_model, x)
# Importing accuracy method
@sk_import metrics: accuracy_score
# Checking accuracy on target variables
accuracy = accuracy_score(prediction, y)
# Calculating confusion matrix (for performance matrix)
# Importing confusion method
@sk_import metrics: confusion_matrix
confusion_matrix(test_data, train_data)
# Importing classification method
@sk_import metrics: classification_report
print(classification_report(test_data, train_data))
# Importing precision method
@sk_import metrics: precision_recall_curve
precision, recall, threshold = precision_recall_curve(test_data, train_data)
#Implementing confusion matrix formulas
# ac : accuracy
# pr : precision
# re : recall
# (True Positive : tp) (True Negative : tn) (False Positive : fp) (False Negative : fn)
ac = [tp + tn]/[tp + tn + fp + fn]
pr = [tp]/[tp + fp]
re = [tp]/[tp + fn]
