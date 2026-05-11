# model_definitions.py
"""
Final model definitions with empirically verified tree traversal logic
Key fix: XGBoost uses < (not <=) for tree decision logic
"""
from typing import List
from enum import Enum
import array
import numpy as np

class TaskType(Enum):
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary"
    MULTICLASS_CLASSIFICATION = "multiclass"

class TreeData:
    """Tree data structure matching JSON format exactly"""
    
    def __init__(self, children_left, children_right, features, thresholds, values, node_count):
        self.children_left = array.array('i', children_left)
        self.children_right = array.array('i', children_right)
        self.features = array.array('i', features)
        self.thresholds = array.array('f', thresholds)
        self.values = array.array('f', values)
        self.node_count = node_count

class UnifiedModel:
    """Unified model with empirically verified prediction logic"""
    
    def __init__(self, trees: List[TreeData], num_features: int, task_type: TaskType, 
                 num_classes: int = 1, feature_names: List[str] = None, base_score: float = 0.0):
        self.trees = trees
        self.num_features = num_features
        self.task_type = task_type
        self.num_classes = num_classes
        self.feature_names = feature_names or [f"f{i}" for i in range(num_features)]
        self.base_score = base_score
    
    @property
    def num_trees(self) -> int:
        return len(self.trees)
    
    @property 
    def total_nodes(self) -> int:
        return sum(tree.node_count for tree in self.trees)
    
    def _traverse_tree(self, tree: TreeData, x: np.ndarray) -> float:
        """
        Tree traversal using EMPIRICALLY VERIFIED logic
        CRITICAL: XGBoost uses < (not <=) for comparison
        """
        node = 0
        
        while node >= 0 and node < tree.node_count:
            # Check if leaf
            if tree.children_left[node] == -1:
                return tree.values[node]
            
            # Internal node: use < comparison (CRITICAL FIX)
            feature_idx = tree.features[node]
            threshold = tree.thresholds[node]
            
            if x[feature_idx] < threshold:  # CRITICAL: Use < not <=
                node = tree.children_left[node]
            else:
                node = tree.children_right[node]
        
        return 0.0
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediction with base_score handling"""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n_samples = X.shape[0]
        
        if self.task_type == TaskType.REGRESSION:
            predictions = np.full(n_samples, self.base_score, dtype=np.float32)
            for tree in self.trees:
                for i in range(n_samples):
                    predictions[i] += self._traverse_tree(tree, X[i])
            return predictions
            
        elif self.task_type == TaskType.BINARY_CLASSIFICATION:
            # Start with base_score converted to logit
            if 0 < self.base_score < 1:
                base_logit = np.log(self.base_score / (1.0 - self.base_score))
            else:
                base_logit = 0.0

            logits = np.full(n_samples, base_logit, dtype=np.float32)
            for tree in self.trees:
                for i in range(n_samples):
                    logits[i] += self._traverse_tree(tree, X[i])
            return 1.0 / (1.0 + np.exp(-logits))
            
        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            predictions = np.zeros((n_samples, self.num_classes), dtype=np.float32)
            
            for tree_idx, tree in enumerate(self.trees):
                class_idx = tree_idx % self.num_classes
                for i in range(n_samples):
                    predictions[i, class_idx] += self._traverse_tree(tree, X[i])
            
            predictions += self.base_score
            exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
            return exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")