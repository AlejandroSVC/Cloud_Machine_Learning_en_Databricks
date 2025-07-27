# Cloud - XGBoost escalable y distribuido en Databricks usando PySpark

![banner_databricks](docs/assets/images/banner_databricks.jpg)

## Descripción general

Flujo de trabajo integral de XGBoost en Databricks
Este tutorial le guía en el entrenamiento y la implementación de un clasificador binario de XGBoost con Databricks, aprovechando la computación distribuida y PySpark para un aprendizaje automático escalable, rentable y seguro. Incluye monitorización, seguimiento de la calidad del modelo y recomendaciones para diferentes tamaños de datos.

## Paso 1: Configuración del entorno

1.1. Configuración del clúster

• Datos pequeños (<1 GB): Un solo nodo, Standard_DS3_v2 (o similar), 1-2 trabajadores.

• Datos medianos (1-50 GB): 4-8 trabajadores, escalado automático habilitado, Standard_DS4_v2.

• Datos grandes (>50 GB): Más de 16 trabajadores, instancias optimizadas para computación (p. ej., Standard_F16s_v2), escalado automático, instancias puntuales para ahorrar costos.

• Seguridad: Habilite roles de IAM a nivel de clúster y configure listas de control de acceso. • Monitoreo: Habilite las métricas del clúster y la entrega de registros a un almacenamiento seguro.

Referencias:

• Dimensionamiento del clúster

https://docs.databricks.com/en/clusters/cluster-sizing.html

• Mejores prácticas de seguridad

https://docs.databricks.com/en/security/index.html

• Monitoreo de clústeres

https://docs.databricks.com/en/administration-guide/clusters/cluster-metrics.html

# XGBoost en Databricks

## Paso 2: Ingesta y Procesamiento de Datos

**2.1. Leer Parquet usando PySpark**

```
from pyspark.sql import SparkSession                       # Importar SparkSession para operaciones de Spark
```
Crear sesión de Spark (Databricks crea una automáticamente como 'spark')

Para pruebas locales: spark = SparkSession.builder.appName("XGBoost_Binary_Classification").getOrCreate()
```
parquet_path = "/mnt/data/your_dataset.parquet"            # Ruta al archivo Parquet
df = spark.read.parquet(parquet_path)                      # Leer archivo Parquet en un DataFrame de Spark
```
Seguridad: Restringir acceso a columnas sensibles si es necesario
```
df = df.drop("PII_column")                                 # Ejemplo: Eliminar columnas sensibles
print(f"Se cargaron {df.count()} filas y {len(df.columns)} columnas")  # Imprimir dimensiones del conjunto de datos
```
**2.2. Perfilado y Preprocesamiento de Datos**

```
from pyspark.sql.functions import col, isnan, when, count  # Importar funciones de SQL de Spark
```
Identificar valores faltantes
```
missing = df.select([                                      # Calcular valores faltantes por columna
    count(when(isnan(c) | col(c).isNull(), c)).alias(c) 
    for c in df.columns
])
missing.show()                                             # Mostrar conteo de valores faltantes
```
Eliminar filas con valores faltantes en el objetivo
```
target_col = "target"                                      # Definir nombre de la variable objetivo
df_clean = df.filter(col(target_col).isNotNull())          # Filtrar valores nulos en el objetivo
```
Selección de características: seleccionar solo columnas numéricas
```
numerical_cols = [  # Identificar características numéricas
    f.name for f in df.schema.fields 
    if str(f.dataType) in ["IntegerType", "DoubleType"] and f.name != target_col
]
```
Opcional: Normalizar características (para XGBoost, no es estrictamente necesario)
```
from pyspark.ml.feature import StandardScaler, VectorAssembler  # Importar herramientas de preprocesamiento

assembler = VectorAssembler(                               # Crear ensamblador de vectores de características
    inputCols=numerical_cols, 
    outputCol="features_vec"
)
df_vec = assembler.transform(df_clean)                     # Generar vectores de características

scaler = StandardScaler(                                   # Inicializar escalador
    inputCol="features_vec", 
    outputCol="features", 
    withMean=True, 
    withStd=True
)
scaler_model = scaler.fit(df_vec)                          # Ajustar escalador a los datos
df_final = scaler_model.transform(df_vec).select("features", target_col)  # Aplicar escalado
```
## Paso 3: Entrenamiento Distribuido de XGBoost

**3.1. Instalar XGBoost4J-Spark y MLflow**

•  Usar Databricks Runtime ML (preinstalado), o: 

```
bash
pip install xgboost mlflow
```
**3.2. Convertir Spark DataFrame a Pandas (si <10GB), o Usar Entrenamiento Distribuido**

```
import mlflow                                              # Importar MLflow para seguimiento de experimentos
import xgboost as xgb                                      # Importar XGBoost para entrenamiento
```
Paso condicional basado en el tamaño de los datos
```
data_size_gb = df_final.count() * len(numerical_cols) * 8 / (1024**3)  # Calcular tamaño de datos en GB

if data_size_gb < 10:                                      # Verificar si el conjunto de datos cabe en un solo nodo
    # Datos pequeños/medianos: recolectar en el driver para entrenamiento en un solo nodo
    pd_df = df_final.toPandas()                            # Convertir a DataFrame de Pandas
    X = np.vstack(pd_df["features"].values)                # Preparar matriz de características
    y = pd_df[target_col].values                           # Preparar vector objetivo
    dtrain = xgb.DMatrix(X, label=y)                       # Crear estructura de datos de XGBoost
else:
    # Datos grandes: usar entrenamiento distribuido
    from xgboost.spark import SparkXGBClassifier           # Importar XGBoost distribuido

    xgb_classifier = SparkXGBClassifier(                   # Configurar clasificador distribuido
        num_workers=8,                                     # Usar 8 trabajadores
        max_depth=6,                                       # Profundidad del árbol
        n_estimators=100,                                  # Número de árboles
        objective="binary:logistic",                       # Clasificación binaria
        eval_metric="logloss"                              # Métrica de evaluación
    )

    model = xgb_classifier.fit(df_final)                   # Entrenar modelo distribuido
```
Seguimiento del modelo (con MLflow)
```
mlflow.start_run(run_name="xgboost_binary_classification") # Iniciar ejecución de MLflow
mlflow.log_param("data_size_gb", data_size_gb)             # Registrar parámetro de tamaño de datos
mlflow.log_param("num_features", len(numerical_cols))      # Registrar conteo de características

if data_size_gb < 10:
    model = xgb.train(                                     # Entrenar modelo en un solo nodo
        {"objective": "binary:logistic", "max_depth": 6}, 
        dtrain, 
        num_boost_round=100
    )
    mlflow.xgboost.log_model(model, "model")               # Registrar modelo XGBoost
else:
    mlflow.spark.log_model(model, "model")                 # Registrar modelo Spark

mlflow.end_run()                                           # Finalizar ejecución de MLflow
```
## Paso 4: Evaluación y Monitoreo del Modelo
```
from sklearn.metrics import roc_auc_score, accuracy_score  # Importar métricas de evaluación

if data_size_gb < 10:
    y_pred = model.predict(dtrain)                         # Generar predicciones
    auc = roc_auc_score(y, (y_pred > 0.5).astype(int))     # Calcular AUC
    acc = accuracy_score(y, (y_pred > 0.5).astype(int))    # Calcular precisión
else:
    # Para SparkXGBClassifier
    predictions = model.transform(df_final)                # Generar predicciones distribuidas
    auc = predictions.selectExpr(                          # Calcular AUC usando correlación
        "float(target) as label", 
        "float(prediction) as prediction"
    ).stat.corr("label", "prediction")
    acc = predictions.filter(                              # Calcular precisión
        predictions.prediction == predictions[target_col]
    ).count() / predictions.count()

print(f"AUC: {auc:.4f}, Precisión: {acc:.4f}")             # Imprimir métricas de evaluación
```
Registrar métricas en MLflow
```
mlflow.log_metric("AUC", auc)                              # Registrar métrica AUC
mlflow.log_metric("Accuracy", acc)                         # Registrar métrica de precisión
```
## Paso 5: Despliegue del Modelo (Serving)
Registrar el modelo en MLflow
```
mlflow.register_model("runs:/<run_id>/model", "XGBoost_Binary_Classifier")  # Registrar el modelo en MLflow
```
