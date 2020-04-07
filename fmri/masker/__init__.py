import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np

from nilearn import datasets
from nilearn.image import concat_imgs
from nilearn.input_data import NiftiLabelsMasker
from nilearn.image import high_variance_confounds

class Masker():

    def __init__(self):
        # img = concat_imgs(temp_path+filename, auto_resample=True, verbose=0)
        fmri_data = datasets.fetch_adhd(n_subjects=1)
        fmri_filename = fmri_data.func[0]
        self.fmri_filename = fmri_filename

    def apply_mask(self, atlas="AAL"):

        if atlas=="AAL":
            # load atlas
            atlas_filename = datasets.fetch_atlas_aal(version="SPM12", verbose=0).maps
        elif atlas=="multiscale":
            raise NotImplementedError()
        else:
            raise ValueError("Altas should be 'AAL' or 'multiscale'")

        # set mask
        masker = NiftiLabelsMasker(
            labels_img=atlas_filename,
            standardize=True, detrend=True,
            low_pass=0.08, high_pass=0.01, t_r=3.7,
            memory="nilearn_cache", verbose=0)

        # apply mask to data
        confounds = high_variance_confounds(self.fmri_filename, n_confounds=1, detrend=True)
        ts_hvar = masker.fit_transform(self.fmri_filename, confounds=confounds)

        return ts_hvar
