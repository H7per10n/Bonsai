# parser.py
"""
Parser built from empirical investigation of XGBoost and LightGBM JSON structures
"""
import json
from typing import Dict, Any, List, Tuple, Optional
from model_definitions import TreeData, UnifiedModel, TaskType

def detect_task_type(objective: str, num_class: Optional[int] = None) -> Tuple[TaskType, int]:
    """Detect task type from objective string"""
    obj_lower = objective.lower()
    
    if any(x in obj_lower for x in ['reg:', 'regression', 'rmse', 'mae', 'mse']):
        return TaskType.REGRESSION, 1
    elif any(x in obj_lower for x in ['binary', 'logloss', 'logistic']):
        return TaskType.BINARY_CLASSIFICATION, 1
    elif any(x in obj_lower for x in ['multi:', 'multiclass', 'softmax', 'softprob']):
        classes = num_class if num_class and num_class > 1 else 3
        return TaskType.MULTICLASS_CLASSIFICATION, classes
    else:
        return TaskType.REGRESSION, 1

class XGBoostParser:
    """XGBoost parser based on empirical JSON structure investigation"""
    
    @staticmethod
    def parse(data: Dict[str, Any]) -> UnifiedModel:
        """Parse XGBoost JSON based on actual structure"""
        
        # Navigate to learner section
        learner = data.get('learner', {})
        if not learner:
            raise ValueError("Invalid XGBoost JSON: missing 'learner'")
        
        # Extract learner parameters including base_score
        learner_params = learner.get('learner_model_param', {})
        base_score = float(learner_params.get('base_score', 0.0))
        
        # Extract objective information
        objective_info = learner.get('objective', {})
        if isinstance(objective_info, dict):
            objective = objective_info.get('name', 'reg:squarederror')
            num_class = objective_info.get('num_class')
        else:
            objective = str(objective_info) if objective_info else 'reg:squarederror'
            num_class = None
        
        # Check learner params for num_class
        if num_class is None and 'num_class' in learner_params:
            num_class = int(learner_params['num_class'])
        
        # Navigate to trees: learner -> gradient_booster -> model -> trees
        gradient_booster = learner.get('gradient_booster', {})
        model = gradient_booster.get('model', {})
        trees_json = model.get('trees', [])
        
        if not trees_json:
            raise ValueError("No trees found in XGBoost model")
        
        # Parse all trees
        trees = []
        max_feature_idx = -1
        
        for tree_json in trees_json:
            tree, tree_max_feature = XGBoostParser._parse_tree(tree_json)
            if tree is not None:
                trees.append(tree)
                max_feature_idx = max(max_feature_idx, tree_max_feature)
        
        if not trees:
            raise ValueError("No valid trees parsed from XGBoost model")
        
        # Determine model parameters
        num_features = max_feature_idx + 1 if max_feature_idx >= 0 else 1
        task_type, num_classes = detect_task_type(objective, num_class)
        
        return UnifiedModel(trees, num_features, task_type, num_classes, 
                          feature_names=None, base_score=base_score)
    
    @staticmethod
    def _parse_tree(tree_json: Dict[str, Any]) -> Tuple[Optional[TreeData], int]:
        """
        Parse single XGBoost tree based on empirical structure:
        - left_children, right_children: navigation arrays
        - split_indices: feature indices for internal nodes
        - split_conditions: thresholds for internal nodes  
        - base_weights: values for all nodes (used for leaves)
        """
        
        # Extract the core arrays from JSON
        left_children = tree_json.get('left_children', [])
        right_children = tree_json.get('right_children', [])
        split_indices = tree_json.get('split_indices', [])
        split_conditions = tree_json.get('split_conditions', [])
        base_weights = tree_json.get('base_weights', [])
        
        if not left_children or not base_weights:
            return None, -1
        
        node_count = len(left_children)
        max_feature_idx = -1
        
        # Build our unified arrays
        features = []
        thresholds = []
        values = []
        
        for i in range(node_count):
            # All nodes store their value from base_weights
            values.append(float(base_weights[i]) if i < len(base_weights) else 0.0)
            
            # Check if leaf: left_children[i] == -1
            if i < len(left_children) and left_children[i] == -1:
                # Leaf node: no feature/threshold needed
                features.append(-1)
                thresholds.append(0.0)
            else:
                # Internal node: extract split info
                feature_idx = int(split_indices[i]) if i < len(split_indices) else -1
                threshold = float(split_conditions[i]) if i < len(split_conditions) else 0.0
                
                features.append(feature_idx)
                thresholds.append(threshold)
                
                if feature_idx >= 0:
                    max_feature_idx = max(max_feature_idx, feature_idx)
        
        tree = TreeData(
            left_children,
            right_children,
            features,
            thresholds,
            values,
            node_count
        )
        
        return tree, max_feature_idx

class LightGBMParser:
    """LightGBM parser based on empirical JSON structure investigation"""
    
    @staticmethod  
    def parse(data: Dict[str, Any]) -> UnifiedModel:
        """Parse LightGBM JSON based on actual structure"""
        
        # Extract basic model info
        objective = data.get('objective', 'regression')
        feature_names = data.get('feature_names', [])
        num_class = data.get('num_class')
        
        # Extract trees from tree_info
        tree_info_list = data.get('tree_info', [])
        if not tree_info_list:
            raise ValueError("No tree_info found in LightGBM model")
        
        trees = []
        for tree_info in tree_info_list:
            tree_structure = tree_info.get('tree_structure')
            if tree_structure:
                tree = LightGBMParser._parse_tree(tree_structure)
                if tree is not None:
                    trees.append(tree)
        
        if not trees:
            raise ValueError("No valid trees found in LightGBM model")
        
        # Determine model parameters
        num_features = len(feature_names) if feature_names else 1
        task_type, num_classes = detect_task_type(objective, num_class)
        
        # LightGBM doesn't use base_score concept (or it's 0)
        return UnifiedModel(trees, num_features, task_type, num_classes, 
                          feature_names, base_score=0.0)
    
    @staticmethod
    def _parse_tree(tree_structure: Dict[str, Any]) -> Optional[TreeData]:
        """
        Parse LightGBM tree from recursive structure:
        - Flatten recursive {left_child, right_child, split_feature, threshold} 
        - Convert to same format as XGBoost for unified handling
        """
        
        # Flatten the recursive structure into a list of nodes
        nodes = []
        
        def flatten_node(node_data: Dict[str, Any]) -> int:
            """Recursively flatten tree structure"""
            current_idx = len(nodes)
            
            if 'leaf_value' in node_data:
                # Leaf node
                nodes.append({
                    'left_child': -1,
                    'right_child': -1,
                    'feature': -1,
                    'threshold': 0.0,
                    'value': float(node_data['leaf_value'])
                })
            else:
                # Internal node
                feature = int(node_data.get('split_feature', -1))
                threshold = float(node_data.get('threshold', 0.0))
                
                # Add placeholder node (children will be updated)
                nodes.append({
                    'left_child': -1,
                    'right_child': -1, 
                    'feature': feature,
                    'threshold': threshold,
                    'value': 0.0
                })
                
                # Process children recursively
                if 'left_child' in node_data:
                    left_idx = flatten_node(node_data['left_child'])
                    nodes[current_idx]['left_child'] = left_idx
                
                if 'right_child' in node_data:
                    right_idx = flatten_node(node_data['right_child'])
                    nodes[current_idx]['right_child'] = right_idx
            
            return current_idx
        
        # Start flattening from root
        flatten_node(tree_structure)
        
        if not nodes:
            return None
        
        # Convert to TreeData format (same as XGBoost)
        tree = TreeData(
            [node['left_child'] for node in nodes],
            [node['right_child'] for node in nodes],
            [node['feature'] for node in nodes],
            [node['threshold'] for node in nodes],
            [node['value'] for node in nodes],
            len(nodes)
        )
        
        return tree

class UniversalParser:
    """Auto-detecting parser for XGBoost and LightGBM"""
    
    @staticmethod
    def parse(file_path: str) -> UnifiedModel:
        """Parse model file with format auto-detection"""
        
        # Load JSON file
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to load JSON file '{file_path}': {e}")
        
        if not isinstance(data, dict):
            raise ValueError("JSON file must contain an object")
        
        # Auto-detect format based on structure
        if 'learner' in data:
            # XGBoost format has 'learner' section
            return XGBoostParser.parse(data)
        elif 'tree_info' in data and 'objective' in data:
            # LightGBM format has 'tree_info' and 'objective'
            return LightGBMParser.parse(data)
        else:
            # Provide helpful error message
            keys = list(data.keys())
            raise ValueError(
                f"Unrecognized model format. Found keys: {keys}. "
                f"Expected XGBoost (with 'learner') or LightGBM (with 'tree_info' and 'objective')"
            )