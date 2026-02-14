"""
Skrypt do analizy i prezentacji wyników eksperymentów
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from IPython.display import display, Image

# Ścieżka do folderu z wynikami
OUTPUT_DIR = "outputs_studentA_mwilk"

def load_all_summaries():
    """Załaduj wszystkie pliki summary CSV"""
    summary_files = glob(os.path.join(OUTPUT_DIR, "A_summary_*.csv"))
    
    all_summaries = []
    for file in sorted(summary_files):
        df = pd.read_csv(file)
        # Dodaj informację o pliku źródłowym
        filename = os.path.basename(file)
        df['source_file'] = filename
        all_summaries.append(df)
    
    if all_summaries:
        return pd.concat(all_summaries, ignore_index=True)
    return None

def load_latest_summary():
    """Załaduj najnowszy plik summary"""
    summary_files = glob(os.path.join(OUTPUT_DIR, "A_summary_*.csv"))
    if not summary_files:
        return None
    
    latest_file = sorted(summary_files)[-1]
    print(f"📊 Ładuję najnowszy plik: {os.path.basename(latest_file)}\n")
    return pd.read_csv(latest_file), latest_file

def show_latest_plot():
    """Pokaż najnowszy wykres"""
    plot_files = glob(os.path.join(OUTPUT_DIR, "A_plot_*.png"))
    if not plot_files:
        print("❌ Brak plików wykresów")
        return
    
    latest_plot = sorted(plot_files)[-1]
    print(f"📈 Najnowszy wykres: {os.path.basename(latest_plot)}\n")
    display(Image(filename=latest_plot))

def show_all_line_plots():
    """Pokaż wszystkie wykresy liniowe (porównanie różnych konfiguracji)"""
    line_plots = glob(os.path.join(OUTPUT_DIR, "A_line_*.png"))
    
    if not line_plots:
        print("❌ Brak wykresów liniowych")
        return
    
    print(f"📊 Znaleziono {len(line_plots)} wykresów liniowych:\n")
    
    for plot_file in sorted(line_plots):
        filename = os.path.basename(plot_file)
        print(f"🔹 {filename}")
        display(Image(filename=plot_file))
        print("\n" + "="*80 + "\n")

def analyze_hidden_size_impact():
    """Analiza wpływu rozmiaru warstwy ukrytej (HIDDEN)"""
    all_data = load_all_summaries()
    
    if all_data is None:
        print("❌ Brak danych do analizy")
        return
    
    print("🔍 ANALIZA WPŁYWU HIDDEN SIZE\n")
    print("="*80)
    
    # Grupuj po hidden size
    grouped = all_data.groupby('hidden').agg({
        'mean_best_acc': 'mean',
        'P_success': 'mean',
        'seq_len': ['min', 'max', 'count']
    }).round(3)
    
    print("\nPodsumowanie według rozmiaru warstwy ukrytej (HIDDEN):")
    print(grouped)
    
    return all_data

def compare_experiments():
    """Porównaj różne eksperymenty"""
    summary_files = glob(os.path.join(OUTPUT_DIR, "A_summary_*.csv"))
    
    print("📋 LISTA WSZYSTKICH EKSPERYMENTÓW\n")
    print("="*80)
    
    experiments = []
    for file in sorted(summary_files):
        df = pd.read_csv(file)
        filename = os.path.basename(file)
        
        # Wyciągnij parametry z nazwy pliku
        if df.shape[0] > 0:
            hidden = df['hidden'].iloc[0]
            pooling = df['pooling'].iloc[0]
            seq_range = f"{df['seq_len'].min()}-{df['seq_len'].max()}"
            avg_acc = df['mean_best_acc'].mean()
            success_rate = df['P_success'].mean()
            
            experiments.append({
                'Plik': filename,
                'HIDDEN': hidden,
                'Pooling': pooling,
                'Seq Range': seq_range,
                'Avg Accuracy': f"{avg_acc:.3f}",
                'Success Rate': f"{success_rate:.1%}"
            })
    
    exp_df = pd.DataFrame(experiments)
    print(exp_df.to_string(index=False))
    print("\n")
    
    return exp_df

# ============================================================================
# GŁÓWNA FUNKCJA DO PREZENTACJI WYNIKÓW
# ============================================================================

def show_results(mode='latest'):
    """
    Główna funkcja do prezentacji wyników
    
    Parametry:
    - mode: 'latest' - tylko najnowsze wyniki
            'all' - wszystkie wykresy
            'compare' - porównanie eksperymentów
            'analyze' - szczegółowa analiza
    """
    
    print("\n" + "="*80)
    print("🎯 PREZENTACJA WYNIKÓW EKSPERYMENTÓW")
    print("="*80 + "\n")
    
    if mode == 'latest':
        # Pokaż najnowsze wyniki
        df, filepath = load_latest_summary()
        if df is not None:
            print("📊 NAJNOWSZE WYNIKI:\n")
            display(df)
            print("\n")
            show_latest_plot()
    
    elif mode == 'all':
        # Pokaż wszystkie wykresy liniowe
        show_all_line_plots()
    
    elif mode == 'compare':
        # Porównaj wszystkie eksperymenty
        compare_experiments()
    
    elif mode == 'analyze':
        # Szczegółowa analiza
        compare_experiments()
        print("\n")
        analyze_hidden_size_impact()
        print("\n")
        show_all_line_plots()
    
    print("\n" + "="*80)
    print("✅ Analiza zakończona!")
    print("="*80)


# ============================================================================
# PRZYKŁADY UŻYCIA:
# ============================================================================

if __name__ == "__main__":
    # Odkomentuj wybraną opcję:
    
    # 1. Pokaż tylko najnowsze wyniki
    show_results('latest')
    
    # 2. Pokaż wszystkie wykresy
    # show_results('all')
    
    # 3. Porównaj wszystkie eksperymenty
    # show_results('compare')
    
    # 4. Pełna analiza
    # show_results('analyze')
