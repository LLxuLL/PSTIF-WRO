import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():

    parser = argparse.ArgumentParser(description='XGBoost Baseline for NYC Taxi')


    parser.add_argument('--dataset', type=str, default='nyc_taxi')
    parser.add_argument('--data_path', type=str, default='./data/raw')
    parser.add_argument('--sample_size', type=int, default=100000)
    parser.add_argument('--test_size', type=float, default=0.15)
    parser.add_argument('--val_size', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./outputs/xgboost_baseline')

    # XGBoost
    parser.add_argument('--max_depth', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--subsample', type=float, default=0.8)

    parser.add_argument('--device', type=str, default='cuda', help='[Ignored] PyTorch specific')
    parser.add_argument('--epochs', type=int, default=50, help='[Ignored] PyTorch specific')
    parser.add_argument('--batch_size', type=int, default=1024, help='[Ignored] PyTorch specific')
    parser.add_argument('--config', type=str, default=None, help='[Ignored] PyTorch specific')

    return parser.parse_args()


def load_nyc_taxi_data(data_path, sample_size=100000, seed=42):

    filepath = f"{data_path}/yellowtaxi_data.csv"

    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Creating synthetic data...")
        np.random.seed(seed)
        n_samples = sample_size

        start_date = pd.Timestamp('2016-01-01')
        end_date = pd.Timestamp('2016-06-30')

        random_times = start_date + pd.to_timedelta(
            np.random.randint(0, int((end_date - start_date).total_seconds()), n_samples),
            unit='s'
        )

        df = pd.DataFrame({
            'vendor_id': np.random.randint(1, 3, n_samples),
            'pickup_datetime': random_times,
            'passenger_count': np.random.randint(1, 7, n_samples),
            'pickup_longitude': np.random.uniform(-74.2, -73.7, n_samples),
            'pickup_latitude': np.random.uniform(40.6, 40.9, n_samples),
            'dropoff_longitude': np.random.uniform(-74.2, -73.7, n_samples),
            'dropoff_latitude': np.random.uniform(40.6, 40.9, n_samples),
            'trip_duration': np.random.randint(60, 3600, n_samples)
        })
    else:
        df = pd.read_csv(filepath)
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=seed)
            df = df.reset_index(drop=True)


    pickup_time = pd.to_datetime(df['pickup_datetime'])

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    distance = haversine(
        df['pickup_latitude'].values,
        df['pickup_longitude'].values,
        df['dropoff_latitude'].values,
        df['dropoff_longitude'].values
    )

    features = pd.DataFrame({
        'hour': pickup_time.dt.hour.values,
        'day_of_week': pickup_time.dt.dayofweek.values,
        'month': pickup_time.dt.month.values,
        'is_weekend': (pickup_time.dt.dayofweek >= 5).astype(int),
        'is_rush_hour': (
            ((pickup_time.dt.hour >= 7) & (pickup_time.dt.hour <= 9)) |
            ((pickup_time.dt.hour >= 17) & (pickup_time.dt.hour <= 19))
        ).astype(int),
        'pickup_latitude': df['pickup_latitude'].values,
        'pickup_longitude': df['pickup_longitude'].values,
        'dropoff_latitude': df['dropoff_latitude'].values,
        'dropoff_longitude': df['dropoff_longitude'].values,
        'passenger_count': df['passenger_count'].values,
        'distance': distance,
        'vendor_id': df['vendor_id'].values,
    })

    median_duration = df['trip_duration'].median()
    labels = (df['trip_duration'].values > median_duration).astype(int)

    print(f"Label distribution: {np.bincount(labels)} "
          f"({np.mean(labels)*100:.1f}% positive class)")

    return features.values, labels, features.columns.tolist()


def evaluate_model(model, X, y, dataset_name="Validation"):

    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    auc = roc_auc_score(y, y_pred_proba)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    print(f"\n{dataset_name} Metrics:")
    print(f"  AUC:        {auc:.4f}")
    print(f"  Accuracy:   {accuracy:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1-Score:   {f1:.4f}")

    return {
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }


def plot_confusion_matrix(y_true, y_pred, save_path):

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Short', 'Long'],
                yticklabels=['Short', 'Long'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('XGBoost Confusion Matrix')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_feature_importance(model, feature_names, save_path):

    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(importance)), importance[indices], color='skyblue', edgecolor='navy', alpha=0.7)
    plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.title('XGBoost Feature Importance', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)


    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance saved to {save_path}")


def main():

    args = parse_args()


    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("XGBoost Baseline for NYC Taxi Trip Duration Classification")
    print("=" * 60)
    print(f"Note: --device={args.device}, --epochs={args.epochs}, --batch_size={args.batch_size} "
          f"are ignored (XGBoost doesn't use them)")
    print(f"Random Seed: {args.seed}")
    print(f"Sample Size: {args.sample_size}")
    print(f"XGBoost Params: max_depth={args.max_depth}, lr={args.learning_rate}, n_estimators={args.n_estimators}")
    print("=" * 60)


    np.random.seed(args.seed)


    print("\n[1/4] Loading data...")
    X, y, feature_names = load_nyc_taxi_data(args.data_path, args.sample_size, args.seed)
    print(f"Data shape: {X.shape}, Features: {X.shape[1]}, Samples: {len(y)}")

    print("\n[2/4] Splitting data (70/15/15)...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=args.test_size + args.val_size, random_state=args.seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")


    print("\n[3/4] Training XGBoost...")
    print(f"Training on {len(X_train)} samples...")
    start_time = time.time()

    model = xgb.XGBClassifier(
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        subsample=args.subsample,
        colsample_bytree=0.1,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=args.seed,
        n_jobs=-1,
        early_stopping_rounds=20,
        verbosity=0
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    training_time = time.time() - start_time

    if hasattr(model, 'best_iteration'):
        print(f"Best iteration: {model.best_iteration}")
    print(f"Training completed in {training_time:.2f} seconds")

    print("\n[4/4] Evaluating...")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")

    results = {
        'model': 'XGBoost',
        'dataset': args.dataset,
        'sample_size': args.sample_size,
        'test_auc': float(test_metrics['auc']),
        'test_accuracy': float(test_metrics['accuracy']),
        'test_precision': float(test_metrics['precision']),
        'test_recall': float(test_metrics['recall']),
        'test_f1': float(test_metrics['f1']),
        'val_auc': float(val_metrics['auc']),
        'training_time_seconds': float(training_time),
        'hyperparameters': {
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'n_estimators': args.n_estimators,
            'subsample': args.subsample
        }
    }

    results_file = os.path.join(args.output_dir, 'xgboost_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    plot_confusion_matrix(y_test, test_metrics['predictions'],
                         os.path.join(args.output_dir, 'confusion_matrix.png'))
    plot_feature_importance(model, feature_names,
                           os.path.join(args.output_dir, 'feature_importance.png'))

    print("\n" + "=" * 60)
    print("COMPARISON WITH PSTIF-WRO")
    print("=" * 60)
    print(f"XGBoost Test AUC:      {test_metrics['auc']:.4f}")
    print(f"XGBoost Val AUC:       {val_metrics['auc']:.4f}")
    print(f"PSTIF-WRO Best Val AUC: 0.8832 (from your log)")
    print(f"Training Time:         {training_time:.1f}s (XGBoost) vs ~2400s (PSTIF-WRO)")

    gap = 0.9102 - test_metrics['auc']
    print(f"\nPerformance Gap:       {gap:+.4f} ({gap*100:+.2f}%)")

    if gap > 0.02:
        print("PSTIF-WRO significantly outperforms XGBoost (>2% improvement)")
    elif gap > 0.005:
        print("✅ PSTIF-WRO marginally better than XGBoost (0.5-2% improvement)")
    elif gap > -0.01:
        print("⚠️  Performance comparable (within 1%)")
    else:
        print("❌ XGBoost performs better (PSTIF-WRO needs improvement)")

    print("=" * 60)


if __name__ == '__main__':
    main()