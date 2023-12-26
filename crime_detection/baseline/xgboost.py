import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier


dul_data = pd.read_csv('../data/Pre-processing_Crime_Data_Top10.csv')

features = ['AREA', 'Rpt Dist No', 'Part 1-2', 'Vict Age', 'Vict Sex', 'Vict Descent',
            'Premis Cd', 'Weapon Used Cd', 'Status Desc', 'Cross Street',  'Month_Rptd', 'Day_Rptd', 'Month_OCC', 'Day_OCC', 'Date Difference',
            "Year_Rptd", "Year_OCC", 'Hour', 'Minute'
           ]

numeric_features = ['Date Difference']
categorical_features = list(set(features) - set(numeric_features))

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

all_reports = ""
data = dul_data
for i in range(10):
    
    X = data[features]  
    y = data['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    xgb_model = XGBClassifier(
        objective='multi:softmax',
        learning_rate=0.1,
        n_estimators=5,
        max_depth=None,
        subsample=1,
        colsample_bytree=1,
        random_state=None,
        verbosity=0,
    )

    xgb_model.fit(X_train_preprocessed, y_train)

    y_pred = xgb_model.predict(X_test_preprocessed)

    report = classification_report(y_test, y_pred, digits=5, zero_division="warn")
    print("Classification Report:\n", report)
    all_reports = f"XGBoost_Classification Report {i}:\n{report}\n\n"
    with open("../report/XGBoost_classification_reports.txt", "a") as report_file:
        report_file.write(all_reports)

    print(all_reports)