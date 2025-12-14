import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURACIÓN ---
# Pon aquí los NOMBRES EXACTOS de tus experimentos en MLflow
# El primero suele ser "hyperheuristic_experiment" (donde está tu HH)
# El segundo el que creamos nuevo para el baseline
EXPERIMENT_NAMES = [
    "hyperheuristic_experiment", 
    "optuna_baseline_isolated_v2" # O el nombre que le hayas puesto al final
]

# Filtro opcional: Si usaste tags, ponlo aquí. Si no, déjalo en None.
# Ejemplo: "tags.version = 'paper_final_v1'"
FILTER_STRING = None 

# Métrica que queremos graficar
METRIC_KEY = "best_so_far" 

def get_experiment_ids(experiment_names):
    ids = []
    for name in experiment_names:
        exp = mlflow.get_experiment_by_name(name)
        if exp:
            ids.append(exp.experiment_id)
            print(f"✅ Encontrado experimento '{name}' (ID: {exp.experiment_id})")
        else:
            print(f"⚠️ No encontrado: '{name}'")
    return ids

def extract_full_history(experiment_ids):
    print("--> Extrayendo historial paso a paso (esto puede tardar unos segundos)...")
    client = MlflowClient()
    
    # 1. Obtener todas las corridas
    runs = mlflow.search_runs(experiment_ids=experiment_ids, filter_string=FILTER_STRING)
    
    history_data = []
    final_results = []
    
    for _, run in runs.iterrows():
        run_id = run.run_id
        
        # Intentamos obtener el nombre de la estrategia desde params o tags
        strategy = run.get('params.strategy', 'Unknown')
        dataset = run.get('params.dataset', 'Unknown')
        
        # Ignorar corridas fallidas
        if run.status != 'FINISHED':
            continue

        # A. Datos para Boxplot (Solo valor final)
        final_score = run.get(f'metrics.{METRIC_KEY}', 0)
        # Si tienes registrado qué modelo ganó
        winning_model = run.get('params.winning_model', 'N/A') 
        
        final_results.append({
            "Dataset": dataset,
            "Estrategia": strategy,
            "F1-Score": final_score,
            "Modelo Ganador": winning_model
        })
        
        # B. Datos para Curvas (Historial completo)
        # MLflow guarda el historial métrico, hay que pedirlo explícitamente
        metric_history = client.get_metric_history(run_id, key=METRIC_KEY)
        
        for m in metric_history:
            history_data.append({
                "Step": m.step,
                "Score": m.value,
                "Estrategia": strategy,
                "Dataset": dataset,
                "RunID": run_id
            })
            
    return pd.DataFrame(history_data), pd.DataFrame(final_results)

def plot_convergence(df_history):
    datasets = df_history["Dataset"].unique()
    
    for ds in datasets:
        subset = df_history[df_history["Dataset"] == ds]
        
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # Lineplot calcula automáticamente la media y el intervalo de confianza (la sombra)
        sns.lineplot(
            data=subset, 
            x="Step", 
            y="Score", 
            hue="Estrategia", 
            style="Estrategia",
            markers=True, 
            dashes=False,
            palette="deep", 
            linewidth=2.5
        )
        
        plt.title(f"Curva de Convergencia - Dataset: {ds.upper()}", fontsize=14)
        plt.ylabel("F1-Score (Best so far)", fontsize=12)
        plt.xlabel("Iteraciones (Presupuesto)", fontsize=12)
        plt.legend(title="Estrategia", loc="lower right")
        
        filename = f"plot_convergence_{ds}.png"
        plt.savefig(filename, dpi=300)
        print(f"📊 Gráfico guardado: {filename}")
        plt.close()

def plot_boxplot(df_final):
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    sns.boxplot(
        data=df_final, 
        x="Dataset", 
        y="F1-Score", 
        hue="Estrategia", 
        palette="Set2",
        showmeans=True
    )
    
    plt.title("Comparación de Rendimiento Final (Distribución por Semillas)", fontsize=14)
    plt.ylabel("F1-Score Macro", fontsize=12)
    plt.legend(loc="lower right")
    
    filename = "plot_boxplot_comparison.png"
    plt.savefig(filename, dpi=300)
    print(f"📊 Gráfico guardado: {filename}")
    plt.close()

def print_stats(df_final):
    print("\n--- RESUMEN ESTADÍSTICO (Para copiar a LaTeX) ---")
    summary = df_final.groupby(["Dataset", "Estrategia"])["F1-Score"].agg(["mean", "std", "max"])
    print(summary)
    
    # Análisis de Modelos Ganadores del Baseline (si existen)
    if "Modelo Ganador" in df_final.columns:
        baseline_data = df_final[df_final["Estrategia"].str.contains("Baseline", case=False)]
        if not baseline_data.empty:
            print("\n--- PREFERENCIAS DEL BASELINE (¿Qué modelo eligió Optuna?) ---")
            print(baseline_data.groupby("Dataset")["Modelo Ganador"].value_counts())

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    mlflow.set_tracking_uri("file:./mlruns") # Ajusta si es necesario
    
    ids = get_experiment_ids(EXPERIMENT_NAMES)
    
    if ids:
        df_hist, df_final = extract_full_history(ids)
        
        if not df_hist.empty:
            print(f"Datos extraídos: {len(df_hist)} puntos de historial.")
            
            # 1. Generar Curvas
            plot_convergence(df_hist)
            
            # 2. Generar Boxplots
            plot_boxplot(df_final)
            
            # 3. Imprimir Tablas
            print_stats(df_final)
        else:
            print("❌ No se encontraron datos. Revisa los nombres de los experimentos o los filtros.")
    else:
        print("❌ No se encontraron los IDs de los experimentos.")