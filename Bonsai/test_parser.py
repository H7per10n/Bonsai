import json
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
from parser import UniversalParser
import os


class ParserTester:
    """Comprehensive parser testing class with concise, structured output"""

    def __init__(self, tolerance=2e-5):
        self.tolerance = tolerance

    # ---------------------- DATA CREATION ----------------------
    def create_regression_data(self, n=2000, d=5):
        return make_regression(n_samples=n, n_features=d, noise=0.1, random_state=42)

    def create_binary_data(self, n=2000, d=5):
        return make_classification(n_samples=n, n_features=d, n_classes=2,
                                   n_informative=d, n_redundant=0,
                                   n_clusters_per_class=1, random_state=42)

    def create_multiclass_data(self, n=2000, d=5, k=3):
        return make_classification(n_samples=n, n_features=d, n_classes=k,
                                   n_informative=d, n_redundant=0,
                                   n_clusters_per_class=1, random_state=42)

    # ---------------------- HELPER OUTPUT ----------------------
    def print_header(self, title):
        print(f"\n{'='*60}\n{title}\n{'='*60}")

    def print_model_info(self, m):
        print(f"Model: {m.task_type.value} | Trees={m.num_trees}, "
              f"Features={m.num_features}, Classes={m.num_classes}, "
              f"Base={m.base_score:.4f}")

    def print_comparison(self, metric_name, orig, parsed, diff):
        print(f"{metric_name:<15} Orig={orig:.6f}  Parsed={parsed:.6f}  Δ={abs(orig - parsed):.6f}")
        print(f"Max Δ={np.max(diff):.6e}, Mean Δ={np.mean(diff):.6e}")

    def print_samples(self, pred1, pred2, diff, label="Sample"):
        print("\nExamples:")
        for i in range(min(5, len(pred1))):
            print(f"{label} {i:<2}: {pred1[i]:.6f} vs {pred2[i]:.6f}  (Δ={diff[i]:.2e})")

    # ---------------------- GENERIC TEST RUNNER ----------------------
    def run_test(self, title, train_fn, predict_fn, metric_fn, file):
        self.print_header(title)
        try:
            # Train + save
            model, orig_pred, y_test = train_fn()
            if isinstance(model, lgb.Booster):
                with open(file, "w") as f:
                    json.dump(model.dump_model(), f)
            else:  # XGB
                model.save_model(file)

            # Parse + predict
            parsed = UniversalParser.parse(file)
            parsed_pred = parsed.predict(predict_fn())

            # Metadata
            self.print_model_info(parsed)

            # Metric + diffs
            diff = np.abs(orig_pred - parsed_pred)
            score_orig, score_parsed = metric_fn(y_test, orig_pred, parsed_pred)
            self.print_comparison(metric_fn.__name__, score_orig, score_parsed, diff)

            # Show samples
            if orig_pred.ndim == 1:
                self.print_samples(orig_pred, parsed_pred, diff)
            else:
                print("\nExamples (first row):")
                print("Orig:   ", orig_pred[0])
                print("Parsed: ", parsed_pred[0])
                print("Δ:      ", diff[0])

            success = np.max(diff) < self.tolerance
            print(f"\nResult: {'✅ PASS' if success else '❌ FAIL'}")
            return success
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
        finally:
            if os.path.exists(file):
                os.remove(file)

    # ---------------------- WRAPPERS FOR EACH CASE ----------------------
    def test_xgboost_regression(self):
        X, y = self.create_regression_data()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

        def train():
            dtr = xgb.DMatrix(Xtr, label=ytr)
            model = xgb.train({'objective': 'reg:squarederror', 'max_depth': 3, 'eta': 0.1}, dtr, 10)
            return model, model.predict(xgb.DMatrix(Xte)), yte

        def predict(): return Xte
        def metric(y, p1, p2): return mean_squared_error(y, p1), mean_squared_error(y, p2)

        return self.run_test("XGBoost Regression", train, predict, metric, "xgb_reg.json")

    def test_xgboost_binary(self):
        X, y = self.create_binary_data()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

        def train():
            dtr = xgb.DMatrix(Xtr, label=ytr)
            model = xgb.train({'objective': 'binary:logistic', 'max_depth': 3, 'eta': 0.1}, dtr, 10)
            return model, model.predict(xgb.DMatrix(Xte)), yte

        def predict(): return Xte
        def metric(y, p1, p2): return roc_auc_score(y, p1), roc_auc_score(y, p2)

        return self.run_test("XGBoost Binary Classification", train, predict, metric, "xgb_bin.json")

    def test_xgboost_multiclass(self):
        X, y = self.create_multiclass_data()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
        K = len(np.unique(y))

        def train():
            dtr = xgb.DMatrix(Xtr, label=ytr)
            model = xgb.train({'objective': 'multi:softprob', 'num_class': K, 'max_depth': 3, 'eta': 0.1}, dtr, 10)
            return model, model.predict(xgb.DMatrix(Xte)), yte

        def predict(): return Xte
        def metric(y, p1, p2):
            return accuracy_score(y, np.argmax(p1, axis=1)), accuracy_score(y, np.argmax(p2, axis=1))

        return self.run_test("XGBoost Multiclass Classification", train, predict, metric, "xgb_multi.json")
    
    def test_lightgbm_regression(self):
        X, y = self.create_regression_data()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

        def train():
            dtr = lgb.Dataset(Xtr, label=ytr)
            model = lgb.train({'objective': 'regression', 'metric': 'mse', 'max_depth': 3, 'learning_rate': 0.1,"verbose": -1}, dtr, 10)
            return model, model.predict(Xte), yte

        def predict(): return Xte
        def metric(y, p1, p2): return mean_squared_error(y, p1), mean_squared_error(y, p2)

        return self.run_test("LightGBM Regression", train, predict, metric, "lgb_reg.json")

    def test_lightgbm_binary(self):
        X, y = self.create_binary_data()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

        def train():
            dtr = lgb.Dataset(Xtr, label=ytr)
            model = lgb.train({'objective': 'binary', 'metric': 'auc', 'max_depth': 3, 'learning_rate': 0.1,"verbose": -1}, dtr, 10)
            return model, model.predict(Xte), yte

        def predict(): return Xte
        def metric(y, p1, p2): return roc_auc_score(y, p1), roc_auc_score(y, p2)

        return self.run_test("LightGBM Binary Classification", train, predict, metric, "lgb_bin.json")

    def test_lightgbm_multiclass(self):
        X, y = self.create_multiclass_data()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
        K = len(np.unique(y))

        def train():
            dtr = lgb.Dataset(Xtr, label=ytr)
            model = lgb.train({'objective': 'multiclass', 'num_class': K, 'metric': 'multi_logloss',
                               'max_depth': 3, 'learning_rate': 0.1,"verbose": -1}, dtr, 10)
            return model, model.predict(Xte), yte

        def predict(): return Xte
        def metric(y, p1, p2):
            return accuracy_score(y, np.argmax(p1, axis=1)), accuracy_score(y, np.argmax(p2, axis=1))

        return self.run_test("LightGBM Multiclass Classification", train, predict, metric, "lgb_multi.json")

    # ---------------------- SUMMARY ----------------------
    def run_all_tests(self):
        tests = [
            self.test_xgboost_regression,
            self.test_xgboost_binary,
            self.test_xgboost_multiclass,
            self.test_lightgbm_regression,
            self.test_lightgbm_binary,
            self.test_lightgbm_multiclass,
        ]
        print("\nCOMPREHENSIVE PARSER TEST SUITE\n" + "="*60)
        results = {t.__name__: t() for t in tests}
        print("\n" + "="*60 + "\nFINAL SUMMARY\n" + "="*60)
        for name, ok in results.items():
            print(f"{name:<35} {'✅' if ok else '❌'}")
        return all(results.values())


if __name__ == "__main__":
    ParserTester().run_all_tests()
