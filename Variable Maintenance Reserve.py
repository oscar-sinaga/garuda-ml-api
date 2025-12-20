import pandas as pd
df = pd.read_excel("C:/Users/hp/Downloads/a/05. Database RP May 2025 - AC REGISTER.xlsx", sheet_name="Raw", skiprows=1)
df.head()


df = df[df['MAINTENANCE RESERVE'] >= 0].copy()

df1 = df[
    (df['FUEL BURN (IN LITER)'] != 0) &
    (df['FLIGHT HOURS'] != 0) &
    (df['MAINTENANCE RESERVE'] != 0)
].copy()

## Finding Corelation
df_num = df.select_dtypes(include=["number"])

corr = df_num.corrwith(df_num['MAINTENANCE RESERVE']).sort_values(ascending=False).dropna()

print(corr.head(30))

## Group VM per PK Register per Periode
df_group = df1.groupby(["AC REG", "PERIODE"]).agg({
    "MAINTENANCE RESERVE": "sum",
    "FLIGHT HOURS": "sum",
    "NUMBER OF LANDING": "sum",
    "FUEL BURN (IN LITER)": "sum",
    "ATK (000)": "sum",
    "LEASE AIRCRAFT": "mean",
    "SERVICE TYPE": "first", 
    "AIRCRAFT TYPE": "first",
}).reset_index()
print(df_group)
df_group['FH_per_Cycle'] = df_group['FLIGHT HOURS'] / df_group['NUMBER OF LANDING']
df_group = df_group.fillna(0)

## Input Selected Features
selected_features = [
    'FLIGHT HOURS',
    'FUEL BURN (IN LITER)',
    'NUMBER OF LANDING',
    'ATK (000)',
    'LEASE AIRCRAFT',
    'AIRCRAFT TYPE',
    'AC REG',
    'PERIODE',
    'FH_per_Cycle'
]

## SPlit Data
X = df_group[selected_features].copy()
y = df_group['MAINTENANCE RESERVE'].copy()
print (X)


from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(df_group, groups=df_group["AC REG"]))

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

categorical_cols = ['AC REG','PERIODE', 'AIRCRAFT TYPE']
numeric_cols = list(set(selected_features) - set(categorical_cols))
print(numeric_cols)
## Encoder
from sklearn.preprocessing import OneHotEncoder


encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder.fit(X_train[categorical_cols])
X_train_encoded = encoder.transform(X_train[categorical_cols])
X_test_encoded = encoder.transform(X_test[categorical_cols])
encoded_cols = encoder.get_feature_names_out(categorical_cols)

X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoded_cols, index=X_train.index)
X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoded_cols, index=X_test.index)
X_train_final = pd.concat([X_train[numeric_cols], X_train_encoded], axis=1)
X_test_final = pd.concat([X_test[numeric_cols], X_test_encoded], axis=1)
X_test_final
## PREDIKSI
from xgboost import XGBRegressor

bst = XGBRegressor(
    n_estimators=500, 
    max_depth=6, 
    learning_rate=0.05, 
    gamma=0.1, 
    objective="reg:squarederror"
)


bst.fit(X_train_final, y_train)
y_pred = bst.predict(X_test_final)
from sklearn.metrics import mean_absolute_percentage_error


mean_absolute_percentage_error(y_test, y_pred)
