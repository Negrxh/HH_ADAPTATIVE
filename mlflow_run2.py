import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURACIÓN ---
# 1. Configura aquí dónde está tu CSV con los datos de la HH
PATH_CSV_HH = "logs/results.csv"  # <--- REVISA QUE ESTA RUTA SEA CORRECTA
# 2. Configura el dataset que quieres graficar
DATASET_FOCUS = "mnist" 
# 3. Configura el nombre del experimento Baseline en MLflow
BASELINE_EXP_NAME = "optuna_baseline_isolated_v2"


def get_hh_data_from_csv(csv_path, dataset_target):
    """Lee el CSV y lo transforma al formato que necesitamos para graficar"""
    print(f"--> Leyendo datos de HH desde CSV: {csv_path}...")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("❌ Error: No encuentro el archivo CSV. Revisa la variable PATH_CSV_HH.")
        return pd.DataFrame(), pd.DataFrame()

    # Filtramos por dataset
    df = df[df['dataset'] == dataset_target].copy()
    
    if df.empty:
        print(f"⚠️ El CSV no tiene datos para el dataset '{dataset_target}'")
        return pd.DataFrame(), pd.DataFrame()

    # 1. PREPARAR HISTORIAL (Para curva de convergencia)
    # Seleccionamos las columnas equivalentes
    df_hist = df[['round', 'best_so_far', 'run_id']].copy()
    df_hist.columns = ['Iteration', 'F1-Score', 'RunID']
    df_hist['Strategy'] = "HH (Adaptive)" # Etiqueta para el gráfico
    
    # 2. PREPARAR FINAL (Para Boxplot)
    # Agrupamos por run_id y tomamos el valor máximo de best_so_far
    # (Asumiendo que best_so_far es monotónico, el último es el mejor)
    df_final = df.groupby('run_id')['best_so_far'].max().reset_index()
    df_final.columns = ['RunID', 'F1-Score']
    df_final['Strategy'] = "HH (Adaptive)"

    print(f"   ✅ HH cargada: {len(df_final)} ejecuciones encontradas.")
    return df_hist, df_final


def get_baseline_data_from_mlflow(exp_name, dataset_target):
    """Lee el Baseline desde MLflow (que ya sabemos que funciona)"""
    print(f"--> Leyendo datos de Baseline desde MLflow: {exp_name}...")
    
    mlflow.set_tracking_uri("file:./mlruns")
    client = MlflowClient()
    
    exp = mlflow.get_experiment_by_name(exp_name)
    if not exp:
        print(f"❌ Error: No encuentro el experimento '{exp_name}' en ./mlruns")
        return pd.DataFrame(), pd.DataFrame()
        
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    
    history_data = []
    final_data = []
    
    for _, run in runs.iterrows():
        if run.status != 'FINISHED': continue
        
        # Filtro de dataset
        ds_val = str(run.get('params.dataset', '')).lower()
        if ds_val != dataset_target: continue

        run_id = run.run_id
        
        # A. Datos Finales
        best_val = run.get('metrics.best_so_far', 0)
        if best_val == 0:
            best_val = run.get('metrics.final_f1_macro', run.get('metrics.score', 0))
            
        final_data.append({
            "Strategy": "Optuna (RF + SVM)",
            "F1-Score": best_val,
            "RunID": run_id
        })
        
        # B. Historial
        try:
            history = client.get_metric_history(run_id, key="best_so_far")
            for m in history:
                history_data.append({
                    "Iteration": m.step,
                    "F1-Score": m.value,
                    "Strategy": "Optuna (RF + SVM)",
                    "RunID": run_id
                })
        except:
            pass
            
    df_hist = pd.DataFrame(history_data)
    df_final = pd.DataFrame(final_data)
    
    print(f"   ✅ Baseline cargado: {len(df_final)} ejecuciones encontradas.")
    return df_hist, df_final


# --- EJECUCIÓN PRINCIPAL ---

# 1. Cargar datos
df_hist_hh, df_final_hh = get_hh_data_from_csv(PATH_CSV_HH, DATASET_FOCUS)
df_hist_base, df_final_base = get_baseline_data_from_mlflow(BASELINE_EXP_NAME, DATASET_FOCUS)

# 2. Unir datos
df_hist_total = pd.concat([df_hist_hh, df_hist_base], ignore_index=True)
df_final_total = pd.concat([df_final_hh, df_final_base], ignore_index=True)

if not df_hist_total.empty:
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid")

    # --- GRÁFICO 1: CONVERGENCIA ---
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_hist_total, 
        x="Iteration", 
        y="F1-Score", 
        hue="Strategy", 
        style="Strategy",
        markers=True, 
        dashes=False,
        linewidth=2.5,
        palette=["#e74c3c", "#3498db"] # Rojo (HH) vs Azul (Baseline)
    )
    plt.title(f"Convergence Comparison: {DATASET_FOCUS.upper()}", fontsize=16)
    plt.ylabel("Best F1-Score (Macro)", fontsize=14)
    plt.xlabel("Optimization Rounds", fontsize=14)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig("paper_convergence_final.png", dpi=300)
    print("\n✅ Generado: paper_convergence_final.png")

    # --- GRÁFICO 2: BOXPLOT ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df_final_total, 
        x="Strategy", 
        y="F1-Score", 
        hue="Strategy", # Arreglo para warning de seaborn
        palette=["#e74c3c", "#3498db"],
        width=0.5,
        showmeans=True,
        meanprops={"marker":"^", "markerfacecolor":"white", "markeredgecolor":"black"},
        legend=False # Ocultamos leyenda redundante
    )
    plt.title(f"Performance Stability: {DATASET_FOCUS.upper()}", fontsize=16)
    plt.ylabel("Final F1-Score", fontsize=14)
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig("paper_boxplot_final.png", dpi=300)
    print("✅ Generado: paper_boxplot_final.png")

else:
    print("\n❌ No se pudieron cargar datos conjuntos. Revisa rutas y nombres.")