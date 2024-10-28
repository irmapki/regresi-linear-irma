# Import library yang diperlukan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Membuat dataset buatan
# Usia (X) dan Jumlah (y)
usia = np.array([[1], [2], [3], [4], [5]])
jumlah = np.array([2, 3, 4, 1, 5])

# Membagi dataset menjadi data pelatihan dan pengujian
usia_train, usia_test, jumlah_train, jumlah_test = train_test_split(usia, jumlah, test_size=0.2, random_state=42)

# Membuat model regresi linear
model = LinearRegression()

# Melatih model
model.fit(usia_train, jumlah_train)

# Memprediksi nilai
jumlah_pred = model.predict(usia_test)

# Menampilkan hasil
print("Koefisien:", model.coef_)
print("Intercept:", model.intercept_)
print("Nilai Prediksi:", jumlah_pred)

# Visualisasi hasil
plt.scatter(usia, jumlah, color='blue', label='Data Asli')
plt.plot(usia, model.predict(usia), color='red', label='Regresi Linear')
plt.scatter(usia_test, jumlah_pred, color='green', marker='x', label='Prediksi')
plt.xlabel('Usia')
plt.ylabel('Jumlah')
plt.legend()
plt.title('Regresi Linear Sederhana')
plt.show()
