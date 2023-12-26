import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

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

for i in range(10):
    data = dul_data.sample(n=100000, random_state=i).reset_index(drop=True)
    X = data[features]  
    y = data['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    gbdt = GradientBoostingClassifier(loss='log_loss', learning_rate=0.1, n_estimators=5, subsample=1,
                                      min_samples_split=2, min_samples_leaf=1, max_depth=None,
                                      init=None, random_state=None, max_features=None,
                                      verbose=0, max_leaf_nodes=None, warm_start=False)
    gbdt.fit(X_train_preprocessed, y_train)
    y_pred = gbdt.predict(X_test_preprocessed)
    report = classification_report(y_test, y_pred, digits=5, zero_division="warn")
    print("Classification Report:\n", report)
    all_reports = f"GBDT_Classification Report {i}:\n{report}\n\n"
    with open("../report/GBDT_classification_reports.txt", "a") as report_file:
        report_file.write(all_reports)

    print(all_reports)
