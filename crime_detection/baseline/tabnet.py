import pandas as pd
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import classification_report
import torch
from sklearn.preprocessing import LabelEncoder

def train_and_evaluate_tabnet(X, y, device, i, num_epochs=10, patience=10, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
    model = TabNetClassifier(device_name=device)
    model.fit(X_train.values, y_train.values, max_epochs=num_epochs, patience=patience)
    y_pred = model.predict(X_test.values)
    report = classification_report(y_test, y_pred, digits=5)
    return report

data = pd.read_csv('../data/Pre-processing_Crime_Data_Top10.csv')

target = 'Label'

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
    X = data[features]
    y = data[target]
    element_to_remove = "Date Difference"
    cat_features = [item for item in features if item != element_to_remove]

    label_encoder = LabelEncoder()
    for feature in cat_features:
        X[feature] = label_encoder.fit_transform(X[feature])
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    all_reports = ""
    num_runs = 10
    for i in range(num_runs):
        print(f"Run {i+1}:")
        report = train_and_evaluate_tabnet(X, y, device, i)
        all_reports += f"Tabnet_Classification Report {i}:\n{report}\n\n"
        print(report)
        print("-" * 80)

    with open('../report/Tabnet_'+str(len(features))+'d_classification_reports.txt', "a") as report_file:
        report_file.write(all_reports)

    print(all_reports)
