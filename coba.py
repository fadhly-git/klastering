# Import library
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("daftar_hadir.csv", delimiter=";")
    return df

# Sidebar untuk memilih jenis aktivitas dan jumlah klaster
st.sidebar.title("Parameter Klastering")

# Load data
df = load_data()  # <-- Pastikan variabel df sudah didefinisikan di sini
df['jenis_aktivitas'] = df['status'].apply(lambda x: 'keberangkatan' if x == 0 else 'kepulangan')

# Filter data berdasarkan jenis aktivitas yang dipilih
jenis_aktivitas = st.sidebar.selectbox("Pilih Jenis Aktivitas", df['jenis_aktivitas'].unique())
filtered_df = df[df['jenis_aktivitas'] == jenis_aktivitas]
df['waktu_numerik'] = pd.to_datetime(df['waktu']).dt.hour * 60 + pd.to_datetime(df['waktu']).dt.minute
# Pilih fitur yang akan digunakan
features = st.sidebar.multiselect("Pilih fitur untuk klastering", ['waktu_numerik', 'id_pegawai'])

st.write(filtered_df[features])


# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(filtered_df[features])

# Proses klastering dengan KMeans
num_clusters = st.sidebar.slider("Jumlah Klaster", min_value=2, max_value=10, value=3)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
filtered_df["Cluster"] = kmeans.fit_predict(X_scaled)

# Visualisasi hasil klastering
st.subheader("Visualisasi Hasil Klastering")
st.write(filtered_df)

# Visualisasi hasil klastering
st.subheader("Visualisasi Hasil Klastering")
plt.figure(figsize=(10, 6))
for cluster in range(num_clusters):
    cluster_data = filtered_df[filtered_df["Cluster"] == cluster]
    plt.scatter([cluster] * len(cluster_data), cluster_data['id_pegawai'], label=f"Cluster {cluster}")

plt.title(f"Hasil Klastering {jenis_aktivitas.capitalize()} menggunakan KMeans")
plt.xlabel("Cluster")
plt.ylabel("ID Pegawai")
plt.xticks(range(num_clusters))
plt.legend()
st.pyplot(plt)
