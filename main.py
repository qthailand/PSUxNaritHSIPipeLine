# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:18:35 2026

@author: acer
"""

from hsi_loader import *
from hsi_plots import *
import pandas as pd
import numpy as np
from hsi_utilities import *

#%% Step 1: Load spectra from cubes assigned by ground truth file

#1.1 get data folder
folder_path = browse_data_folder()  

#1.2 get data folder
spectral_data, classes, wavelength, cube_names, image_id, annotation_id = bip_spectra_loading_by_gt(
    folder_path,                                                                                          
    max_spectra_per_id=100,
    random_state=42,
)

get_spectra_per_class(classes)

#1.3 create sample id array: cube_name + annotation_id (unique for each spectrum)
sample_id = np.char.add(
    np.char.add(cube_names.astype(str), "_"),
    annotation_id.astype(str),
)

# create dataframe of metadata
bundle_metadata = pd.DataFrame({
    "classes": classes,
    "cube_names": cube_names,
    "image_id": image_id,
    "annotation_id": annotation_id,
    "sample_id": sample_id,
})

#plot 1
plot_all_spectra_by_classes(spectral_data,classes,wavelength, max_spectra_per_class=500)

#%% Step2: Prepocessing
#2.1 sor
data_sor, classes_sor, keep_mask = spectral_sor(spectral_data, classes,std_weight=1.0)

cube_names = cube_names[keep_mask] # filter out metadata
image_id = image_id[keep_mask]
annotation_id = annotation_id[keep_mask]
sample_id = sample_id[keep_mask]

plot_all_spectra_by_classes(data_sor,classes_sor,wavelength,max_spectra_per_class=500)

#2.2 Spectral smoothening
data_sg, classes_sg = spectral_SG_smooth(data_sor,classes_sor,window_length=11,polyorder=2)
plot_all_spectra_by_classes(data_sg,classes_sg,wavelength,max_spectra_per_class=500)

#2.3 Spectral 1st Derivative 

data_1stDeriv, classes_1stDeriv = spectral_SG_1stDeriv(
    data_sg,
    classes_sg,
    wavelength,
    window_length=11,
    polyorder=2,
    max_spacing_variation=5.0,
    )

plot_all_spectra_by_classes(data_1stDeriv,classes_1stDeriv, wavelength,max_spectra_per_class=500)


#%%

# 3.1 PCA

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Standardize spectral bands
scaler = StandardScaler()

data_scaled = scaler.fit_transform(
    data_sg
)

# PCA
# Keep enough PCs to explain 99% of the total variance
pca = PCA(
    n_components=0.99,
    svd_solver="full",
)

data_pca = pca.fit_transform(
    data_scaled
)

classes_pca = classes_sg.copy()


# PCA information
print("\nPCA results")
print("-----------")

print(
    f"Original dimensions : {data_sg.shape[1]}"
)

print(
    f"Number of PCs       : {data_pca.shape[1]}"
)

print(
    f"Explained variance  : "
    f"{np.sum(pca.explained_variance_ratio_) * 100:.2f}%"
)

print(
    "\nExplained variance by PC:"
)

for i, variance in enumerate(
    pca.explained_variance_ratio_,
    start=1,
):
    print(
        f"PC{i:3d}: {variance * 100:7.3f}%"
    )

#%% Step 3.2: 3D PCA plot by class

import matplotlib.pyplot as plt
import numpy as np

# Check that PCA has at least 3 components
if data_pca.shape[1] < 3:
    raise ValueError(
        "PCA must contain at least 3 principal components "
        "for a 3D plot."
    )

unique_classes = np.unique(
    classes_pca
)

# Create figure
fig = plt.figure(
    figsize=(10,5)
)

ax = fig.add_subplot(
    111,
    projection="3d",
)

# Plot each class separately
for cls in unique_classes:

    class_mask = (
        classes_pca == cls
    )

    ax.scatter(
        data_pca[class_mask, 0],
        data_pca[class_mask, 1],
        data_pca[class_mask, 2],
        s=12,
        alpha=0.6,
        label=str(cls),
    )

# Explained variance for axis labels
pc1_var = (
    pca.explained_variance_ratio_[0]
    * 100
)

pc2_var = (
    pca.explained_variance_ratio_[1]
    * 100
)

pc3_var = (
    pca.explained_variance_ratio_[2]
    * 100
)

# Axis labels
ax.set_xlabel(
    f"PC1 ({pc1_var:.1f}%)",
    fontsize=11,
)

ax.set_ylabel(
    f"PC2 ({pc2_var:.1f}%)",
    fontsize=11,
)

ax.set_zlabel(
    f"PC3 ({pc3_var:.1f}%)",
    fontsize=11,
)

ax.set_title(
    "3D PCA Score Plot",
    fontsize=14,
)

# Class legend
ax.legend(
    title="Classes",
    loc="best",
)

plt.tight_layout()
plt.show()
#%% Step 3: Feature Extraction






