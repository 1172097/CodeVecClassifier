import ast
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Any, Tuple
import json
import random


@dataclass
class PathContext:
    """Represents a single path context (xs, p, xt)"""
    start_token: str  # xs: start terminal value
    path: str        # p: syntactic path
    end_token: str   # xt: end terminal value

@dataclass
class MethodFeatures:
    """Contains all path contexts for a method/function"""
    name: str
    path_contexts: List[PathContext] = field(default_factory=list)

class ASTNode:
    def __init__(self, ast_node: ast.AST, parent=None, child_id: int = 0):
        self.ast_node = ast_node
        self.parent = parent
        self.child_id = child_id
        self.type = type(ast_node).__name__
        # Store actual value for terminal nodes
        self.value = self._extract_value(ast_node)
    
    def _extract_value(self, node: ast.AST) -> str:
        """Extract actual values from terminal nodes"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Num):
            return str(node.n)
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Constant):
            return str(node.value)
        return ""

    def is_terminal(self) -> bool:
        """Check if node is a terminal node with a meaningful value"""
        return bool(self.value)

class PathContextExtractor:
    LPAREN = "("
    RPAREN = ")"
    UP_SYMBOL = "^"
    DOWN_SYMBOL = "_"
    
    # Configuration
    MAX_PATH_LENGTH = 8
    MAX_PATH_WIDTH = 2
    MAX_CONTEXTS = 200  # Maximum number of contexts to extract per method
    
    def __init__(self):
        self.ast_nodes: List[ASTNode] = []
    
    def extract_path_contexts(self, code: str) -> List[PathContext]:
        """Extract path contexts from code snippet"""
        try:
            tree = ast.parse(code)
            return self._generate_path_contexts(tree)
        except SyntaxError:
            # Try wrapping in a function if it's a code snippet
            wrapped_code = f"def wrapper():\n{code}"
            try:
                tree = ast.parse(wrapped_code)
                return self._generate_path_contexts(tree)
            except SyntaxError as e:
                raise ValueError(f"Could not parse code: {e}")
    
    def _generate_path_contexts(self, tree: ast.AST) -> List[PathContext]:
        # Convert AST to our custom nodes with parent links
        self.ast_nodes = []
        self._convert_ast_to_nodes(tree)
        
        # Get all terminal nodes
        terminals = [node for node in self.ast_nodes if node.is_terminal()]
        
        # Generate path contexts
        contexts = []
        for i, start_node in enumerate(terminals):
            for end_node in terminals[i + 1:]:
                path = self._generate_path(start_node, end_node)
                if path:  # Only add if valid path found
                    context = PathContext(
                        start_token=start_node.value,
                        path=path,
                        end_token=end_node.value
                    )
                    contexts.append(context)
                    
                    # Limit number of contexts
                    if len(contexts) >= self.MAX_CONTEXTS:
                        return contexts
        
        return contexts
    
    def _convert_ast_to_nodes(self, ast_node: ast.AST, parent: Optional[ASTNode] = None, child_id: int = 0) -> ASTNode:
        node = ASTNode(ast_node, parent, child_id)
        self.ast_nodes.append(node)
        
        for i, child in enumerate(ast.iter_child_nodes(ast_node)):
            self._convert_ast_to_nodes(child, node, i)
            
        return node
    
    def _get_tree_stack(self, node: ASTNode) -> List[ASTNode]:
        stack = []
        current = node
        while current is not None:
            stack.append(current)
            current = current.parent
        return stack
    
    def _generate_path(self, start_node: ASTNode, end_node: ASTNode) -> Optional[str]:
        start_stack = self._get_tree_stack(start_node)
        end_stack = self._get_tree_stack(end_node)
        
        # Find common ancestor
        common_prefix = 0
        while (common_prefix < len(start_stack) and 
               common_prefix < len(end_stack) and 
               start_stack[-1 - common_prefix] == end_stack[-1 - common_prefix]):
            common_prefix += 1
            
        path_length = len(start_stack) + len(end_stack) - 2 * common_prefix
        if path_length > self.MAX_PATH_LENGTH:
            return None
            
        # Check path width
        if len(start_stack) > common_prefix and len(end_stack) > common_prefix:
            path_width = abs(end_stack[-1 - common_prefix].child_id - 
                           start_stack[-1 - common_prefix].child_id)
            if path_width > self.MAX_PATH_WIDTH:
                return None
        
        # Build the path string
        path_parts = []
        
        # Upward path from start to common ancestor
        for node in start_stack[:-common_prefix]:
            path_parts.append(f"{self.LPAREN}{node.type}{self.RPAREN}")
            path_parts.append(self.UP_SYMBOL)
        
        # Common ancestor
        if common_prefix > 0:
            common_node = start_stack[-common_prefix]
            path_parts.append(f"{self.LPAREN}{common_node.type}{self.RPAREN}")
        
        # Downward path from common ancestor to end
        for node in reversed(end_stack[:-common_prefix]):
            path_parts.append(f"{self.DOWN_SYMBOL}{self.LPAREN}{node.type}{self.RPAREN}")
            
        return "".join(path_parts)
    

def load_data_and_extract_contexts(json_filepath: str) -> Dict[int, List[PathContext]]:
    """Load data from JSON file and extract path contexts for each code snippet"""
    with open(json_filepath, 'r') as file:
        data = json.load(file)
    
    extractor = PathContextExtractor()
    contexts_by_code_id = {}
    
    for item in data:
        code_id = item['code_id']
        code = item['code']
        label = item['label']
        contexts = extractor.extract_path_contexts(code)
        contexts_by_code_id[code_id] = (contexts, label)

    
    return contexts_by_code_id

def transform_to_training_data(json_filepath: str, output_filepath: str) -> None:
    """Transform data from JSON file into training data structure and write to output file"""
    contexts_by_code_id = load_data_and_extract_contexts(json_filepath)
    
    training_data = []
    for code_id, contexts in contexts_by_code_id.items():
        training_data.append({
            "code_id": code_id,
            "contexts": [{"start_token": ctx.start_token, "path": ctx.path, "end_token": ctx.end_token} for ctx in contexts[0]],
            "label": contexts[1]  # Placeholder label, update as needed
        })
    
    # Scramble the order of the elements
    random.shuffle(training_data)
    
    with open(output_filepath, 'w') as file:
        json.dump(training_data, file, indent=4)


def main():
    json_filepath = "./data.json"
    output_filepath = "./training_data.json"
    transform_to_training_data(json_filepath, output_filepath)

    print(f"Training data written to {output_filepath}")




# main()

if __name__ == "__main__":
    # Example usage
    sample_code = """
  
    def weird_binary_search(arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = random.randint(left, right)
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1


    """
    # x = 42
    # y = "hello"
    # z = [1, 2, 3]
    # a = {"key": "value"}
    # b = (4, 5, 6)
    # c = {7, 8, 9}
    # d = True
    # e = None



    extractor = PathContextExtractor()
    contexts = extractor.extract_path_contexts(sample_code)
    final = []
    for context in contexts:
        final.append({
            "start_token": context.start_token,
            "path": context.path,
            "end_token": context.end_token
        })
    print(final)

    