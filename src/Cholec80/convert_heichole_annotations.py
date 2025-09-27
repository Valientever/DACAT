#!/usr/bin/env python3
"""
Script to convert HeiChole phase annotation files from integer format to string format
matching the Cholec80 dataset format.
"""

import os
import glob
import pandas as pd

# Phase mapping from integer to string (standard laparoscopic cholecystectomy phases)
PHASE_MAPPING = {
    0: "Preparation",
    1: "CalotTriangleDissection", 
    2: "ClippingCutting",
    3: "GallbladderDissection",
    4: "GallbladderPackaging",
    5: "CleaningCoagulation",
    6: "GallbladderRetraction"
}

def convert_annotation_file(input_file, output_file):
    """Convert a single HeiChole annotation file to Cholec80 format."""
    print(f"Converting {os.path.basename(input_file)}...")
    
    # Read the CSV file
    df = pd.read_csv(input_file, header=None, names=['frame_number', 'phase_int'])
    
    # Convert integer phases to string phases
    df['phase_string'] = df['phase_int'].map(PHASE_MAPPING)
    
    # Check for unmapped phases
    unmapped = df[df['phase_string'].isna()]
    if not unmapped.empty:
        print(f"Warning: Found unmapped phases in {input_file}:")
        print(unmapped['phase_int'].unique())
        # Fill unmapped phases with "Unknown"
        df['phase_string'] = df['phase_string'].fillna("Unknown")
    
    # Create output dataframe in Cholec80 format
    output_df = pd.DataFrame({
        'Frame': df['frame_number'],
        'Phase': df['phase_string']
    })
    
    # Write to output file with tab separator (matching Cholec80 format)
    output_df.to_csv(output_file, sep='\t', index=False)
    
    print(f"✓ Converted {len(df)} frames. Output: {os.path.basename(output_file)}")
    
    # Print phase distribution
    phase_counts = df['phase_string'].value_counts()
    print("Phase distribution:")
    for phase, count in phase_counts.items():
        print(f"  {phase}: {count} frames")
    print()

def main():
    # Define paths
    input_dir = "/home/santhi/Documents/DACAT/src/Cholec80/data/HeiChole/phase_annotations"
    output_dir = "/home/santhi/Documents/DACAT/src/Cholec80/data/HeiChole/phase_annotations_converted"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Converting HeiChole phase annotation files to Cholec80 format...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Phase mapping: {PHASE_MAPPING}")
    print("-" * 60)
    
    # Find all HeiChole annotation files
    input_pattern = os.path.join(input_dir, "Hei-Chole*_Annotation_Phase.csv")
    input_files = glob.glob(input_pattern)
    
    if not input_files:
        print("No HeiChole annotation files found!")
        return
    
    print(f"Found {len(input_files)} files to convert.")
    print()
    
    # Convert each file
    for input_file in sorted(input_files):
        # Generate output filename (change extension to .txt to match Cholec80)
        basename = os.path.basename(input_file)
        video_name = basename.replace("_Annotation_Phase.csv", "")
        output_filename = f"{video_name}-phase.txt"
        output_file = os.path.join(output_dir, output_filename)
        
        try:
            convert_annotation_file(input_file, output_file)
        except Exception as e:
            print(f"✗ Error converting {basename}: {str(e)}")
            print()
    
    print("Conversion completed!")
    print(f"Converted files are saved in: {output_dir}")

if __name__ == "__main__":
    main()