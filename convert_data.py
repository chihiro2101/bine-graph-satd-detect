import pandas as pd
from sklearn.model_selection import train_test_split
import pdb
from sklearn.model_selection import StratifiedKFold
import os
import subprocess
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# df = pd.read_csv("data/satd-java-raw/satd-dataset-r-comment-without-nondebt.csv")
# print(len(df))
# df = df.dropna()
# print(len(df))
# df.to_csv("data/satd-java-raw/satd-dataset-r-comment-without-nondebt.csv", index= False, header= True)

df = pd.read_csv("data/satd-java-raw/satd-dataset-r-comment-without-nondebt.csv")
print(len(df))
df = df.dropna()
print(len(df))
# import pdb;pdb.set_trace()

# #binary-classification
# df['classification'] = df['classification'].apply(lambda x : x if x == 'non_debt' else "debt")

#multi-class classification
# df = df[df.classification != 'non_debt']
# df.to_csv("data/satd-java-raw/satd-dataset-r-comment-without-nondebt.csv", index= False, header= True)


# satd_type = ['code_debt', 'documentation_debt', 'test_debt', 'requirement_debt' ]
# df['classification'] = df['classification'].apply(lambda x : "code_debt" if x == 'design_debt' else x)
# df['classification'] = df['classification'].apply(lambda x : "non_debt" if x not in satd_type else x)
# df = df[df.classification != 'non_debt']
# df.to_csv("data/satd-java-raw/satd-dataset-r-comment-without-nondebt.csv", index= False, header= True)
# print(len(df))

df = df.rename(columns={"classification": "label"})
# df = df.rename(columns={"debt": "label"})

# number_of_folds = 5 #for code comment, issue
# number_of_folds = 3 #for commit message
# number_of_folds = 3 #for pull--request
number_of_folds = 5 #for r--comment
kf = KFold(n_splits = number_of_folds, shuffle = True, random_state = 2)
fold = kf.split(df)


def get_dataset():
    result = next(fold, None)
    train = df.iloc[result[0]]
    test =  df.iloc[result[1]]
    # import pdb;pdb.set_trace()
    train.to_csv("train.csv")
    test.to_csv("test.csv")
    train['subset'] = 'train'
    test['subset'] = 'test'
    frames = [train, test]
    dataset = pd.concat(frames)
    # import pdb;pdb.set_trace()

    #create file dataset.txt
    df_label = dataset[['subset', 'label']].copy()
    df_content = dataset[['text']].copy()
    df_label.to_csv("data/SATD.txt", sep='\t', index=True, header=False)
    df_content.to_csv("data/corpus/SATD.clean.txt", sep=' ', index=False, header=False)
    

get_dataset()
# get_dataset()
print("DONE")

# Assuming your features are in X and labels in y
# X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# # Create an instance of SMOTE
# smote = SMOTE(random_state=42)

# # Fit the SMOTE model on the training data
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


# import pdb;pdb.set_trace()
# # Now, X_resampled and y_resampled contain the resampled data
# print(pd.Series(y_resampled).value_counts())
