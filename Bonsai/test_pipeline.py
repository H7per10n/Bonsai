import os
import json
import re
import numpy as np
from typing import List, Tuple
from sklearn.datasets import make_classification, make_regression
import xgboost as xgb
import lightgbm as lgb

from model_definitions import TreeData, UnifiedModel, TaskType
from parser import UniversalParser
from generator import MinimalEmbeddedTreeGenerator, EmbeddedConfig

def create_datasets() -> List[Tuple[np.ndarray, np.ndarray, str, TaskType]]:
    datasets = []
    print("Creating datasets...")
    # Binary classification
    X_bin, y_bin = make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=42
    )
    datasets.append((X_bin, y_bin, "binary", TaskType.BINARY_CLASSIFICATION))

    # Multiclass classification
    X_multi, y_multi = make_classification(
        n_samples=1000, n_features=15, n_classes=4, n_informative=10, random_state=42
    )
    datasets.append((X_multi, y_multi, "multiclass", TaskType.MULTICLASS_CLASSIFICATION))

    # Regression
    X_reg, y_reg = make_regression(
        n_samples=1000, n_features=8, noise=0.1, random_state=42
    )
    datasets.append((X_reg, y_reg, "regression", TaskType.REGRESSION))

    return datasets


def train_and_save_xgboost(X, y, task_type, output_path):
    params = {'max_depth': 20, 'eta': 0.1, 'seed': 42}
    if task_type == TaskType.REGRESSION:
        params['objective'] = 'reg:squarederror'
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        params['objective'] = 'binary:logistic'
    else:
        params['objective'] = 'multi:softprob'
        params['num_class'] = len(np.unique(y))

    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(params, dtrain, num_boost_round=5)
    model.save_model(output_path)
    return model


def train_and_save_lightgbm(X, y, task_type, output_path):
    params = {'max_depth': 20, 'num_leaves': 80, 'learning_rate': 0.1,
              'num_iterations': 5, 'seed': 42, 'verbose': -1}
    if task_type == TaskType.REGRESSION:
        params['objective'] = 'regression'
        params['metric'] = 'mse'
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        params['objective'] = 'binary'
        params['metric'] = 'auc'
    else:
        params['objective'] = 'multiclass'
        params['num_classes'] = len(np.unique(y))
        params['metric'] = 'multi_logloss'

    train_data = lgb.Dataset(X, label=y)
    model = lgb.train(params, train_data, num_boost_round=5)

    with open(output_path, 'w') as f:
        json.dump(model.dump_model(), f)
    return model


def extract_memory_info(code: str) -> Tuple[int, int, int]:
    """Extract basic memory info from generated code"""
    total_nodes_match = re.search(r"// Total nodes: (\d+)", code)
    node_size_match = re.search(r"// Node size: (\d+) bytes", code)
    total_memory_match = re.search(r"// Total memory: (\d+) bytes", code)
    
    total_nodes = int(total_nodes_match.group(1)) if total_nodes_match else 0
    node_size = int(node_size_match.group(1)) if node_size_match else 0
    total_memory = int(total_memory_match.group(1)) if total_memory_match else 0
    
    return total_nodes, node_size, total_memory


def write_master_makefile(gen_dir, model_folders):
    """Generate master Makefile for all models"""
    makefile_path = os.path.join(gen_dir, "Makefile")
    
    # Build targets and clean targets
    targets = []
    clean_targets = []
    run_targets = []
    
    for folder_name in model_folders:
        targets.append(f"{folder_name}/test_model")
        clean_targets.append(f"clean-{folder_name}")
        run_targets.append(f"run-{folder_name}")
    
    makefile_content = f"""CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99

# Build all models
all: {' '.join(targets)}

# Individual model targets
"""
    
    for folder_name in model_folders:
        model_type = folder_name.split('_')[0]  # Extract xgboost/lightgbm
        makefile_content += f"""{folder_name}/test_model: {folder_name}/main.c {folder_name}/{model_type}.h
\t$(CC) $(CFLAGS) -o {folder_name}/test_model {folder_name}/main.c -lm

"""
    
    # Clean targets
    makefile_content += f"""# Clean targets
clean: {' '.join(clean_targets)}

"""
    
    for folder_name in model_folders:
        makefile_content += f"""clean-{folder_name}:
\trm -f {folder_name}/test_model

"""
    
    # Run targets
    makefile_content += f"""# Run targets
run-all: {' '.join(run_targets)}

"""
    
    for folder_name in model_folders:
        makefile_content += f"""run-{folder_name}: {folder_name}/test_model
\t@echo "Running {folder_name}:"
\t@./{folder_name}/test_model
\t@echo

"""
    
    makefile_content += f""".PHONY: all clean {' '.join(clean_targets)} run-all {' '.join(run_targets)}
"""
    
    with open(makefile_path, "w", encoding='utf-8') as f:
        f.write(makefile_content)
    print(f"Created master Makefile in {gen_dir}")


def write_main_c(gen_folder, X_test, y_test, model_type, task_type, model):
    """Generate main.c file with batched input vectors and Python predictions for comparison in a loop"""
    main_c_path = os.path.join(gen_folder, "main.c")

    # Use first 5 samples for testing
    num_test = min(5, len(X_test))
    num_features = X_test.shape[1]
    feature_strs = []
    py_preds = []

    # Collect input features and Python predictions
    for i in range(num_test):
        feature_vector = X_test[i].flatten()
        feature_str = ", ".join([f"{x:.4f}f" for x in feature_vector])
        feature_strs.append(feature_str)

        input_data = X_test[i:i+1]
        if model_type == "xgboost":
            py_pred = model.predict(xgb.DMatrix(input_data))
        else:
            py_pred = model.predict(input_data)
        
        if task_type == TaskType.MULTICLASS_CLASSIFICATION:
            py_pred = py_pred[0]
            py_preds.append(py_pred)
        else:
            py_pred = py_pred[0]
            py_preds.append(py_pred)

    # Generate input vector set
    input_array = ",\n        ".join([f"{{{s}}}" for s in feature_strs])
    
    # Generate Python predictions array
    if task_type == TaskType.MULTICLASS_CLASSIFICATION:
        output_size = len(py_preds[0])
        py_pred_rows = [
            "{" + ", ".join([f"{x:.4f}f" for x in pred]) + "}"
            for pred in py_preds
        ]
        py_pred_array = ",\n        ".join(py_pred_rows)
        py_preds_decl = f"float python_preds[{num_test}][{output_size}]"
        pred_decl = f"float output[{output_size}];"
        pred_call = "predict(inputs[i], output);"
        print_c = f'printf("C prediction: "); for(int j=0; j<{output_size}; j++) printf("%.4f ", output[j]); printf("\\n");'
        print_py = f'printf("Python prediction: "); for(int j=0; j<{output_size}; j++) printf("%.4f ", python_preds[i][j]); printf("\\n");'
        compare = f"""
        int match = 1;
        for(int j=0; j<{output_size}; j++) {{
            if(fabs(output[j] - python_preds[i][j]) > 0.01f) {{
                match = 0;
                break;
            }}
        }}
        printf("Match: %s\\n", match ? "Yes" : "No");
"""
    else:
        output_size = 1
        py_pred_array = ", ".join([f"{pred:.4f}f" for pred in py_preds])
        py_preds_decl = f"float python_preds[{num_test}]"
        pred_decl = "float output;"
        pred_call = "output = predict(inputs[i]);"
        print_c = 'printf("C prediction: %.4f\\n", output);'
        print_py = 'printf("Python prediction: %.4f\\n", python_preds[i]);'
        compare = 'printf("Match: %s\\n", fabs(output - python_preds[i]) < 0.01f ? "Yes" : "No");'

    main_code = f"""#include <stdio.h>
#include <math.h>
#include "{model_type}.h"

int main() {{
    // Input vector set
    float inputs[{num_test}][{num_features}] = {{
        {input_array}
    }};

    // Python predictions
    {py_preds_decl} = {{
        {py_pred_array}
    }};

    // Test loop
    for(int i=0; i<{num_test}; i++) {{
        printf("Test sample %d:\\n", i+1);
        {pred_decl}
        {pred_call}
        {print_c}
        {print_py}
        {compare}
        printf("\\n");
    }}

    return 0;
}}
"""
    with open(main_c_path, "w") as f:
        f.write(main_code)
    print(f"Created main.c in {gen_folder} with {num_test} test samples in batched format")

def write_makefile(gen_folder, model_type):
    """Generate Makefile for easy compilation"""
    makefile_path = os.path.join(gen_folder, "Makefile")
    makefile_content = f"""CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
TARGET = test_model
SOURCES = main.c

all: $(TARGET)

$(TARGET): $(SOURCES) {model_type}.h
\t$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) -lm

clean:
\trm -f $(TARGET)

run: $(TARGET)
\t./$(TARGET)

.PHONY: all clean run
"""
    with open(makefile_path, "w") as f:
        f.write(makefile_content)
    print(f"Created Makefile in {gen_folder}")





import os
import logging
from generator import MinimalEmbeddedTreeGenerator, EmbeddedConfig, OptimizationMetrics
from model_definitions import UnifiedModel, TaskType

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def optimized_demo():
    """Demo the minimal memory embedded tree generator"""
    logger.debug("Starting optimized_demo")
    models_dir = "models"
    gen_dir = "gen_minimal"
    print
    
    try:
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(gen_dir, exist_ok=True)
        logger.debug(f"Created directories: {models_dir}, {gen_dir}")
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        return

    try:
        datasets = create_datasets()
        logger.debug(f"Retrieved {len(datasets)} datasets")
    except Exception as e:
        logger.error(f"Error in create_datasets: {e}")
        return
    
    if not datasets:
        logger.warning("No datasets returned from create_datasets")
        return

    generated_files = []
    model_folders = []

    print("MINIMAL MEMORY EMBEDDED TREE GENERATOR DEMO")
    print("=" * 50)
    print("Optimized for minimal memory using single node array, bitfields, smart types, unions, and deduplication")
    print()

    for X, y, name, task_type in datasets:
        logger.debug(f"Processing dataset: {name}, task_type: {task_type}")
        print(f"Processing {name} dataset...")

        model_configs = [
            ("xgboost", f"xgb_{name}.json", train_and_save_xgboost),
            ("lightgbm", f"lgb_{name}.json", train_and_save_lightgbm)
        ]

        for model_type, model_filename, train_fn in model_configs:
            logger.debug(f"Training {model_type} model for {name}")
            model_path = os.path.join(models_dir, model_filename)
            print(f"  Training {model_type} model...")
            
            try:
                # Train and save model
                model = train_fn(X, y, task_type, model_path)
                logger.debug(f"Trained {model_type} model, saved to {model_path}")
            except Exception as e:
                logger.error(f"Error training {model_type} model: {e}")
                continue

            try:
                # Parse model for code generation
                unified_model = UniversalParser.parse(model_path)
                logger.debug(f"Parsed model from {model_path}")
            except Exception as e:
                logger.error(f"Error parsing model {model_path}: {e}")
                continue

            # Create folder per model
            folder_name = f"{model_type}_{name}"
            gen_folder = os.path.join(gen_dir, folder_name)
            try:
                os.makedirs(gen_folder, exist_ok=True)
                logger.debug(f"Created folder: {gen_folder}")
            except Exception as e:
                logger.error(f"Error creating folder {gen_folder}: {e}")
                continue
            model_folders.append(folder_name)

            # Generate optimized C code
            config = EmbeddedConfig(threshold_precision=2, leaf_precision=2, pack_structs=True)
            lib_path = os.path.join(gen_folder, f"{model_type}.h")
            try:
                generator = MinimalEmbeddedTreeGenerator(unified_model, config)
                metrics = generator.analyze_and_optimize()
                code = generator.generate_code(lib_path)
                logger.debug(f"Generated C code at {lib_path}")
            except Exception as e:
                logger.error(f"Error generating C code for {lib_path}: {e}")
                continue

            try:
                # Write main.c with multiple test samples (first 5 from dataset)
                num_test = min(5, len(X))
                X_test = X[:num_test]
                y_test = y[:num_test]
                write_main_c(gen_folder, X_test, y_test, model_type, task_type, model)
                logger.debug(f"Wrote main.c in {gen_folder} with {num_test} test samples")
            except Exception as e:
                logger.error(f"Error writing main.c in {gen_folder}: {e}")
                continue

            try:
                # Write Makefile
                write_makefile(gen_folder, model_type)
                logger.debug(f"Wrote Makefile in {gen_folder}")
            except Exception as e:
                logger.error(f"Error writing Makefile in {gen_folder}: {e}")
                continue

            # Extract memory info directly from metrics
            total_nodes = sum(max(t.node_count, 1) for t in unified_model.trees)
            node_size = metrics.node_size_reduction[1]
            total_memory = metrics.optimized_memory.total_bytes

            generated_files.append((
                lib_path, 
                len(code), 
                total_nodes, 
                node_size, 
                total_memory,
                metrics.compression_ratio,
                metrics.threshold_deduplication,
                metrics.leaf_deduplication,
                folder_name
            ))

            print(f"    Generated {lib_path}")
            print(f"    Code size: {len(code)} characters")
            print(f"    Total nodes: {total_nodes}")
            print(f"    Node size: {node_size} bytes")
            print(f"    Total memory: {total_memory} bytes")
            print(f"    Compression ratio: {metrics.compression_ratio:.2f}x")
            print(f"    Threshold deduplication: {metrics.threshold_deduplication[0]} -> {metrics.threshold_deduplication[1]}")
            print(f"    Leaf deduplication: {metrics.leaf_deduplication[0]} -> {metrics.leaf_deduplication[1]}")
            print(f"    Memory breakdown:")
            print(f"      - Nodes: {metrics.optimized_memory.node_bytes} bytes")
            print(f"      - Thresholds: {metrics.optimized_memory.threshold_bytes} bytes")
            print(f"      - Leaves: {metrics.optimized_memory.leaf_bytes} bytes")
            print(f"      - Metadata: {metrics.optimized_memory.metadata_bytes} bytes")

    try:
        # Write master Makefile
        write_master_makefile(gen_dir, model_folders)
        logger.debug(f"Wrote master Makefile in {gen_dir}")
    except Exception as e:
        logger.error(f"Error writing master Makefile: {e}")
        return

    # Summary
    print()
    print("GENERATION SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Code Size':<10} {'Nodes':<8} {'Node Size':<10} {'Memory':<10} {'Compression':<12} {'Thr Dedup':<12} {'Leaf Dedup':<12} {'Bytes/Node':<10}")
    print("-" * 80)
    
    total_code_size = 0
    total_nodes_all = 0
    total_memory_all = 0
    
    for lib_path, code_size, total_nodes, node_size, total_memory, compression_ratio, thr_dedup, leaf_dedup, folder_name in generated_files:
        bytes_per_node = total_memory / total_nodes if total_nodes > 0 else 0
        compression_str = f"{compression_ratio:.2f}x" if compression_ratio > 1.0 else "N/A"
        thr_dedup_str = f"{thr_dedup[0]}->{thr_dedup[1]}" if thr_dedup[0] > 0 else "N/A"
        leaf_dedup_str = f"{leaf_dedup[0]}->{leaf_dedup[1]}" if leaf_dedup[0] > 0 else "N/A"
        
        print(f"{folder_name:<20} {code_size:<10} {total_nodes:<8} {node_size:<10} {total_memory:<10} {compression_str:<12} {thr_dedup_str:<12} {leaf_dedup_str:<12} {bytes_per_node:.1f}")
        
        total_code_size += code_size
        total_nodes_all += total_nodes
        total_memory_all += total_memory

    print("-" * 80)
    avg_bytes_per_node = total_memory_all / total_nodes_all if total_nodes_all > 0 else 0
    
    print(f"{'TOTALS':<20} {total_code_size:<10} {total_nodes_all:<8} {'-':<10} {total_memory_all:<10} {'-':<12} {'-':<12} {'-':<12} {avg_bytes_per_node:.1f}")

    print()
    print("COMPILATION INSTRUCTIONS:")
    print("=" * 50)
    print(f"cd {gen_dir}")
    print("make all              # Build all models")
    print("make run-all          # Run all models and compare predictions")
    print("make clean            # Clean all models")
    print()
    print("Individual models:")
    for folder_name in model_folders:
        print(f"make run-{folder_name}")
    
    print()
    print("GENERATOR CHARACTERISTICS:")
    print("- Single global node array with start indices")
    print("- Deduplicated thresholds and leaves in shared float arrays")
    print("- Separate precision controls for thresholds and leaves")
    print("- Minimal integer types for indices and features")
    print("- Bitfield for is_leaf flag")
    print("- Union for leaf/internal node data")
    print("- Packed structs for minimal memory")


if __name__ == "__main__":
    exit(optimized_demo())