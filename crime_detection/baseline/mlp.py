import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report

dul_data = pd.read_csv('../data/Pre-processing_Crime_Data_Top10.csv')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MLPModel(nn.Module):
    def __init__(self, input_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def train_and_evaluate_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    model = model.to(device)
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
        
    model.eval()  
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(batch_labels.cpu().numpy())
    
    report = classification_report(y_true, y_pred, digits=5)
    return report

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

list_feature = [features_19, features_7, features_10, features_13, features_16, features_19]

all_reports = ""
num_epochs = 50

for feat in list_feature:
    for i in range(1,10):
        
        data = dul_data
        numerical_cols = ['Date Difference']
        element_to_remove = "Date Difference"
        categorical_cols = [item for item in feat if item != element_to_remove]
        

        
        data[categorical_cols] = data[categorical_cols].astype(str)
        x_categorical = torch.tensor(pd.get_dummies(data[categorical_cols]).values, dtype=torch.float)

        if element_to_remove in feat:
            x_numerical = torch.tensor(data[numerical_cols].values, dtype=torch.float)
            features = torch.cat((x_numerical, x_categorical), dim=1)
        else:
            features = x_categorical

        labels = torch.tensor(data['Label'].values, dtype=torch.long)
        num_classes = 10

        input_dim = features.shape[1]
        mlp_model = MLPModel(input_dim).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
        
        print(f"Experiment {i+1}:")
        
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=i)
        batch_size = 64
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = TensorDataset(X_test.unsqueeze(1), y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        mlp_model = MLPModel(input_dim).to(device)
        optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
        report = train_and_evaluate_model(mlp_model, train_loader, test_loader, criterion, optimizer, num_epochs)
        all_reports = f"MLP_Classification Report {i}:\n{report}\n\n"
        print(report)
        print("===================================")
        # Save all_reports string to a txt file
        with open('../report/MLP_'+str(len(feat))+'d_classification_reports.txt', "a") as report_file:
            report_file.write(all_reports)
