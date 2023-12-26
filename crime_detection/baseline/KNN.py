import pandas as pd
from sklearn.model_selection import train_test_split
from cuml.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


dul_data = pd.read_csv('./data/Pre-processing_Crime_Data_Top10.csv')

data = dul_data

features = ['AREA', 'Rpt Dist No', 'Part 1-2', 'Vict Age', 'Vict Sex', 'Vict Descent',
            'Premis Cd', 'Weapon Used Cd', 'Status Desc', 'Cross Street',  'Month_Rptd', 'Day_Rptd', 'Month_OCC', 'Day_OCC', 'Date Difference',
            "Year_Rptd", "Year_OCC", 'Hour', 'Minute'
           ]

features_19 = ["AREA NAME", "Rpt Dist No", "Part 1-2", "Vict Age", "Vict Sex", "Vict Descent", "Premis Desc","Weapon Desc","Status", 
                      "Cross Street", "Month_Rptd", "Day_Rptd", "Year_Rptd", "Month_OCC", "Day_OCC", "Year_OCC", "Date Difference", "Hour", "Minute"]

features_16 = ["AREA NAME", "Rpt Dist No", "Part 1-2", "Vict Age", "Vict Sex", "Vict Descent","Weapon Desc","Status", 
                      "Cross Street", "Month_Rptd", "Day_Rptd", "Year_Rptd", "Month_OCC", "Day_OCC", "Year_OCC", "Date Difference"]

features_13 = ["AREA NAME", "Rpt Dist No",  "Vict Age", "Vict Sex", "Vict Descent","Weapon Desc","Status", 
                      "Cross Street", "Month_Rptd", "Day_Rptd", "Year_Rptd", "Month_OCC",  "Year_OCC"]

features_10 = ["AREA NAME", "Rpt Dist No",  "Vict Age", "Vict Sex", "Vict Descent",
                       "Month_Rptd", "Day_Rptd", "Year_Rptd", "Month_OCC",  "Year_OCC"]

features_7 = ["AREA NAME",   "Vict Age", "Vict Sex", 
                       "Month_Rptd", "Day_Rptd",  "Month_OCC",  "Year_OCC"]

list_feature = [features_7, features_10, features_13, features_16, features_19]


for features in list_feature:
    all_reports = ""
    X = data[features]  
    y = data['Label']
    numeric_features = ['Date Difference']
    categorical_features = list(set(X.columns) - set(numeric_features))
    
    if numeric_features in features:
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
    else :
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
    for i in range(10):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)
        knn = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
        knn.fit(X_train_preprocessed, y_train)
        y_pred = knn.predict(X_test_preprocessed)
        report = classification_report(y_test, y_pred, digits=5,zero_division="warn")
        print("KNN_'+str(len(features))+'d_classification_report:\n", report)
        all_reports += f"KNN_{len(features)}d_classification_reports {i}:\n{report}\n\n"
        
    with open('./report/KNN_'+str(len(features))+'d_classification_reports.txt', "w") as report_file:
        report_file.write(all_reports)

    print(all_reports)
