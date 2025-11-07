"""
Final Project Reorganization
=============================
Creates clean structure and consolidates all documentation
"""

import os
import shutil
import glob

print("="*80)
print("REORGANIZING PROJECT - FINAL CLEAN STRUCTURE")
print("="*80)

# Define clean structure
structure = {
    '01_data': ['Original and preprocessed datasets'],
    '02_models': {
        'simple_baseline': ['Simple ML models'],
        'advanced_baseline': ['Advanced ML models'],
        'transformer': ['Deep learning model']
    },
    '03_notebooks': ['Jupyter notebooks (if any)'],
    '04_results': {
        'visualizations': ['All plots'],
        'tables': ['All CSV results']
    },
    '05_app': ['Streamlit application'],
    '06_docs': ['Single README only']
}

# Create directories
print("\n[1/4] Creating clean directory structure...")
os.makedirs('01_data/raw', exist_ok=True)
os.makedirs('01_data/processed', exist_ok=True)
os.makedirs('02_models/simple_baseline', exist_ok=True)
os.makedirs('02_models/advanced_baseline', exist_ok=True)
os.makedirs('02_models/transformer', exist_ok=True)
os.makedirs('03_notebooks', exist_ok=True)
os.makedirs('04_results/visualizations', exist_ok=True)
os.makedirs('04_results/tables', exist_ok=True)
os.makedirs('05_app', exist_ok=True)
os.makedirs('06_docs', exist_ok=True)
print("[OK] Directory structure created")

# Move data
print("\n[2/4] Moving data files...")
if os.path.exists('data/structured_crystallization_data.xlsx'):
    shutil.copy2('data/structured_crystallization_data.xlsx', '01_data/raw/')
if os.path.exists('data/data.csv'):
    shutil.copy2('data/data.csv', '01_data/raw/')
    
# Copy preprocessed data
if os.path.exists('baseline_simple/data_preprocessed.csv'):
    shutil.copy2('baseline_simple/data_preprocessed.csv', '01_data/processed/simple_baseline_preprocessed.csv')
if os.path.exists('advanced_baseline/data_advanced_preprocessed.csv'):
    shutil.copy2('advanced_baseline/data_advanced_preprocessed.csv', '01_data/processed/advanced_baseline_preprocessed.csv')
if os.path.exists('transformer_approach/data_transformer_preprocessed.csv'):
    shutil.copy2('transformer_approach/data_transformer_preprocessed.csv', '01_data/processed/transformer_preprocessed.csv')
    
print("[OK] Data files organized")

# Move models
print("\n[3/4] Moving models...")
# Simple
simple_models = glob.glob('baseline_simple/*.pkl')
for model in simple_models:
    shutil.copy2(model, '02_models/simple_baseline/')
if os.path.exists('baseline_simple/training_results.json'):
    shutil.copy2('baseline_simple/training_results.json', '02_models/simple_baseline/')

# Advanced  
advanced_models = glob.glob('advanced_baseline/*.pkl')
for model in advanced_models:
    shutil.copy2(model, '02_models/advanced_baseline/')
if os.path.exists('advanced_baseline/training_results.json'):
    shutil.copy2('advanced_baseline/training_results.json', '02_models/advanced_baseline/')

# Transformer
transformer_files = glob.glob('transformer_approach/*.pkl') + glob.glob('transformer_approach/*.pth') + glob.glob('transformer_approach/*.json')
for f in transformer_files:
    shutil.copy2(f, '02_models/transformer/')
if os.path.exists('transformer_approach/transformer_results.csv'):
    shutil.copy2('transformer_approach/transformer_results.csv', '02_models/transformer/')
    
print("[OK] Models organized")

# Move results
print("\n[4/4] Moving results...")
# Visualizations
if os.path.exists('baseline_simple/eda_plots'):
    for plot in glob.glob('baseline_simple/eda_plots/*.png'):
        filename = os.path.basename(plot)
        shutil.copy2(plot, f'04_results/visualizations/eda_{filename}')
    for csv in glob.glob('baseline_simple/eda_plots/*.csv'):
        filename = os.path.basename(csv)
        shutil.copy2(csv, f'04_results/tables/eda_{filename}')

if os.path.exists('results/plots'):
    for plot in glob.glob('results/plots/*.png'):
        shutil.copy2(plot, '04_results/visualizations/')

# Tables
if os.path.exists('results'):
    for csv in glob.glob('results/*.csv'):
        shutil.copy2(csv, '04_results/tables/')

print("[OK] Results organized")

# Move app
print("\n[5/5] Moving application...")
if os.path.exists('app.py'):
    shutil.copy2('app.py', '05_app/app.py')
if os.path.exists('RUN_APP.bat'):
    shutil.copy2('RUN_APP.bat', '05_app/RUN_APP.bat')
print("[OK] Application organized")

print("\n" + "="*80)
print("REORGANIZATION COMPLETE!")
print("="*80)
print("\nNew structure created:")
print("  01_data/           # All data files")
print("  02_models/         # All trained models")
print("  03_notebooks/      # For future notebooks")
print("  04_results/        # All visualizations and tables")
print("  05_app/            # Streamlit application")
print("  06_docs/           # Documentation (README will be here)")
print("="*80)

