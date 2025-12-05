
"""
CMPE 295A - Intracranial Aneurysm Detection
Preprocessing Pipeline Module

Author: Venkat Gowtham Bhupalam
Date: December 2024
"""

import os
import numpy as np
import pydicom
import nibabel as nib
from scipy import ndimage
import json

class AneurysmPreprocessor:
    """
    Preprocessing pipeline for intracranial aneurysm detection.
    
    Steps:
    1. Load DICOM series
    2. Intensity normalization (percentile clipping)
    3. Resample to isotropic spacing
    4. Export to NIfTI format for nnU-Net
    """
    
    def __init__(self, target_spacing=(1.0, 1.0, 1.0)):
        self.target_spacing = target_spacing
    
    def load_dicom_series(self, series_folder):
        """Load DICOM series and return 3D volume with metadata."""
        dcm_files = sorted([f for f in os.listdir(series_folder) if f.endswith('.dcm')])
        
        slices = []
        for f in dcm_files:
            dcm = pydicom.dcmread(os.path.join(series_folder, f))
            slices.append(dcm)
        
        # Sort by slice location
        slices.sort(key=lambda x: float(getattr(x, 'SliceLocation', 0)))
        
        # Stack into 3D volume
        volume = np.stack([s.pixel_array for s in slices], axis=0)
        
        # Get spacing
        pixel_spacing = slices[0].PixelSpacing
        slice_thickness = float(getattr(slices[0], 'SliceThickness', 1.0))
        spacing = (slice_thickness, float(pixel_spacing[0]), float(pixel_spacing[1]))
        
        metadata = {
            'modality': slices[0].Modality,
            'series_uid': getattr(slices[0], 'SeriesInstanceUID', 'unknown'),
            'original_shape': volume.shape,
            'original_spacing': spacing
        }
        
        return volume, spacing, metadata
    
    def normalize_intensity(self, volume, percentile_lower=1, percentile_upper=99):
        """Percentile-based intensity normalization."""
        p_low = np.percentile(volume, percentile_lower)
        p_high = np.percentile(volume, percentile_upper)
        
        volume_clipped = np.clip(volume, p_low, p_high)
        volume_norm = (volume_clipped - p_low) / (p_high - p_low + 1e-8)
        
        return volume_norm.astype(np.float32)
    
    def resample_volume(self, volume, current_spacing):
        """Resample volume to target isotropic spacing."""
        resize_factor = np.array(current_spacing) / np.array(self.target_spacing)
        resampled = ndimage.zoom(volume, resize_factor, order=1)
        return resampled
    
    def preprocess(self, series_folder):
        """Run full preprocessing pipeline."""
        volume, spacing, metadata = self.load_dicom_series(series_folder)
        volume_norm = self.normalize_intensity(volume)
        volume_resampled = self.resample_volume(volume_norm, spacing)
        
        metadata['processed_shape'] = volume_resampled.shape
        metadata['target_spacing'] = self.target_spacing
        
        return volume_resampled, metadata


def prepare_nnunet_dataset(preprocessor, series_list, data_path, output_dir, max_cases=None):
    """
    Prepare dataset in nnU-Net format.
    
    Creates:
    - imagesTr/case_XXXX_0000.nii.gz
    - labelsTr/case_XXXX.nii.gz
    - dataset.json
    """
    os.makedirs(f"{output_dir}/imagesTr", exist_ok=True)
    os.makedirs(f"{output_dir}/labelsTr", exist_ok=True)
    
    series_path = f"{data_path}/series"
    seg_path = f"{data_path}/segmentations"
    
    if max_cases is None:
        max_cases = len(series_list)
    
    processed_cases = []
    
    for i, series_id in enumerate(series_list[:max_cases]):
        print(f"Processing {i+1}/{max_cases}: {series_id[:30]}...")
        
        try:
            volume, metadata = preprocessor.preprocess(os.path.join(series_path, series_id))
            
            seg_file = f"{seg_path}/{series_id}.nii"
            if os.path.exists(seg_file):
                seg_nii = nib.load(seg_file)
                seg_data = seg_nii.get_fdata()
                seg_binary = (seg_data > 0).astype(np.uint8)
            else:
                seg_binary = np.zeros(volume.shape, dtype=np.uint8)
            
            case_id = f"case_{i:04d}"
            
            img_nii = nib.Nifti1Image(volume, np.eye(4))
            nib.save(img_nii, f"{output_dir}/imagesTr/{case_id}_0000.nii.gz")
            
            if seg_binary.shape != volume.shape:
                zoom_factors = np.array(volume.shape) / np.array(seg_binary.shape)
                seg_resampled = ndimage.zoom(seg_binary, zoom_factors, order=0)
            else:
                seg_resampled = seg_binary
            
            lbl_nii = nib.Nifti1Image(seg_resampled.astype(np.uint8), np.eye(4))
            nib.save(lbl_nii, f"{output_dir}/labelsTr/{case_id}.nii.gz")
            
            processed_cases.append({'case_id': case_id, 'series_id': series_id})
            print(f"  ✓ {case_id}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    dataset_json = {
        "name": "IntracranialAneurysm",
        "description": "RSNA Intracranial Aneurysm Detection",
        "tensorImageSize": "3D",
        "channel_names": {"0": "MRA"},
        "labels": {"background": 0, "vessel": 1},
        "numTraining": len(processed_cases),
        "file_ending": ".nii.gz"
    }
    
    with open(f"{output_dir}/dataset.json", 'w') as f:
        json.dump(dataset_json, f, indent=2)
    
    return processed_cases


if __name__ == "__main__":
    # Example usage
    preprocessor = AneurysmPreprocessor(target_spacing=(1.0, 1.0, 1.0))
    print("Preprocessor initialized")
