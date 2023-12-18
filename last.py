# Import library
import streamlit as st
import numpy as np 
import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
import matplotlib.transforms as transforms
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# Load dataset
file_path = "daftar_hadir.csv"
df = pd.read_csv(file_path, delimiter=';')

# Konversi kolom waktu ke dalam bentuk numerik
df['waktu_numerik'] = pd.to_datetime(df['waktu']).dt.hour * 60 + pd.to_datetime(df['waktu']).dt.minute

# Tambahkan kolom jenis_aktivitas
df['jenis_aktivitas'] = df['status'].apply(lambda x: 'keberangkatan' if x == 0 else 'kepulangan')

# Sidebar
st.sidebar.header("Parameter Klastering")
jenis_aktivitas = st.sidebar.radio("Pilih jenis aktivitas", ['keberangkatan', 'kepulangan'])
features = st.sidebar.multiselect("Pilih fitur untuk klastering", ['waktu_numerik'])
num_samples = st.sidebar.slider("Jumlah data untuk klastering", min_value=100, max_value=len(df), value=1000)
num_clusters = st.sidebar.slider("Jumlah klaster", min_value=2, max_value=10, value=3)

# Filter data berdasarkan jenis aktivitas
filtered_df = df[df['jenis_aktivitas'] == jenis_aktivitas].sample(n=num_samples, random_state=42)

# Klastering
if st.sidebar.button("Mulai Klastering"):
    if not features:
        st.warning("Pilih setidaknya satu fitur untuk klastering.")
    elif len(filtered_df) == 0:
        st.warning(f"Tidak ada data untuk jenis aktivitas {jenis_aktivitas}. Pilih jenis aktivitas lain.")
    else:
        # Pilih fitur yang akan digunakan
        X = filtered_df[features]

        # Normalisasi data
        scaler = StandardScaler()
        
        # Pengecekan apakah ada sampel yang cukup untuk diproses
        if len(X) > 1:
            X_scaled = scaler.fit_transform(X)

            # Proses klastering dengan KMeans
            kmeans = KMeans(n_clusters=num_clusters, random_state=2023)
            filtered_df["Cluster"] = kmeans.fit_predict(X_scaled)

            # Tampilkan hasil klastering
            st.success("Klastering selesai!")
            st.write(filtered_df[['id_pegawai', 'waktu_numerik', 'jenis_aktivitas', 'Cluster']])

            # Visualisasi hasil klastering
            st.subheader("Visualisasi Hasil Klastering")
            fig, ax = plt.subplots()

            for cluster in range(num_clusters):
                cluster_data = filtered_df[filtered_df["Cluster"] == cluster]
                ax.scatter(cluster_data['waktu_numerik'], cluster_data['id_pegawai'], label=f"Cluster {cluster}")

                confidence_ellipse(
                    x=cluster_data['waktu_numerik'],
                    y=cluster_data['id_pegawai'],
                    ax=ax,
                    edgecolor="black",
                    facecolor=f"C{cluster}",
                    alpha=0.2,
                    n_std=2  # You can adjust the number of standard deviations here
                )

            plt.title(f"Hasil Klastering {jenis_aktivitas.capitalize()} menggunakan KMeans")
            plt.xlabel("Waktu Numerik")
            plt.ylabel("ID Pegawai")
            plt.legend()
            st.pyplot(fig)

        else:
            st.warning("Tidak cukup sampel untuk diproses. Pilih jenis aktivitas lain.")

# Tampilkan informasi dataset
st.header(f"Informasi Dataset Daftar Hadir - {jenis_aktivitas.capitalize()}")
st.write(filtered_df[['id_pegawai', 'tanggal', 'waktu', 'jenis_aktivitas', 'Cluster']] if 'Cluster' in filtered_df.columns else filtered_df[['id_pegawai', 'tanggal', 'waktu', 'jenis_aktivitas']])

# Tampilkan deskripsi statistik dataset
st.header("Deskripsi Statistik Dataset")
st.write(filtered_df.describe())