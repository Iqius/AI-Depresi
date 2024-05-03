import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Langkah 1: Memuat dan memahami data dari file "sleep.csv"
sleep_data = pd.read_csv("sleep.csv")
print("Data Sleep:")
print(sleep_data.head())

# Mengubah nilai "Sleep Apnea" menjadi 1 dan nilai NaN menjadi 0 dalam kolom "Sleep Disorder"
sleep_data['Sleep Disorder'] = sleep_data['Sleep Disorder'].apply(lambda x: 1 if x == 'Sleep Apnea' else 0 if pd.notna(x) else x)

# Mengubah kolom "Gender" menjadi representasi numerik
label_encoder = LabelEncoder()
sleep_data['Gender'] = label_encoder.fit_transform(sleep_data['Gender'])

# Langkah 2: Membuat fitur "risiko_depresi" berdasarkan kriteria yang telah ditentukan
sleep_data['risiko_depresi'] = ((sleep_data['Sleep Duration'] < 6) & 
                                (sleep_data['Quality of Sleep'] < 7) & 
                                (sleep_data['Physical Activity Level'] < 60) & 
                                (sleep_data['Stress Level'] > 6) & 
                                (sleep_data['Sleep Disorder'].notna())).astype(int)

# Memisahkan fitur dan target
X = sleep_data[['Age', 'Gender', 'Sleep Duration', 'Quality of Sleep', 
                'Physical Activity Level', 'Stress Level', 'Sleep Disorder']]
y = sleep_data['risiko_depresi']

# Membuat dan melatih model Decision Tree
model = DecisionTreeClassifier()
model.fit(X, y)

# Prediksi menggunakan model pada data latih
train_predictions = model.predict(X)

# Hitung akurasi pada data latih
accuracy = accuracy_score(y, train_predictions)
print("Akurasi model pada data latih: {:.2f}%".format(accuracy * 100))

# Langkah 3: Memuat data uji dari file "test.csv"
test_data = pd.read_csv("test.csv")
print("\nData Test:")
print(test_data.head())

# Mengubah kolom "Gender" menjadi representasi numerik pada data uji
test_data['Gender'] = label_encoder.transform(test_data['Gender'])

# Mengubah nilai "Sleep Apnea" menjadi 1 dan nilai NaN menjadi 0 dalam kolom "Sleep Disorder" pada data uji
test_data['Sleep Disorder'] = test_data['Sleep Disorder'].apply(lambda x: 1 if x == 'Sleep Apnea' else 0 if pd.notna(x) else x)

# Menggunakan model yang telah dilatih untuk memprediksi risiko depresi pada data uji
X_test = test_data[['Age', 'Gender', 'Sleep Duration', 'Quality of Sleep', 
                    'Physical Activity Level', 'Stress Level', 'Sleep Disorder']]
predictions = model.predict(X_test)

# Menambahkan prediksi ke dalam data uji
test_data['risiko_depresi'] = predictions

# Menyimpan hasil prediksi ke dalam file CSV
test_data.to_csv('test_predictions.csv', index=False)

print("\nHasil prediksi risiko depresi pada data uji disimpan dalam file 'test_predictions.csv'.")
