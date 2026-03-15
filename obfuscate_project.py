import ast
import os
import json
import shutil
import sys

SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "site-packages",
}


# ==============================
# PASS 1: 收集所有函数和类定义
# ==============================

class DefinitionCollector(ast.NodeVisitor):

    def __init__(self):
        self.mapping = {
            "functions": {},
            "classes": {},
            "methods": {},
        }
        self.func_counter = 1
        self.class_counter = 1
        self.class_stack = []
        self.function_depth = 0

    def add_function(self, name):
        if name.startswith("__"):
            return

        if name not in self.mapping["functions"]:
            self.mapping["functions"][name] = f"func_op_{self.func_counter}"
            self.func_counter += 1

    def add_class(self, name):

        if name not in self.mapping["classes"]:
            self.mapping["classes"][name] = f"ClassEntity_{self.class_counter}"
            self.class_counter += 1

    def add_method(self, class_path, name):
        if name.startswith("__"):
            return

        if class_path not in self.mapping["methods"]:
            self.mapping["methods"][class_path] = {}

        if name not in self.mapping["methods"][class_path]:
            self.mapping["methods"][class_path][name] = f"func_op_{self.func_counter}"
            self.func_counter += 1

    def visit_FunctionDef(self, node):
        if self.class_stack and self.function_depth == 0:
            class_path = ".".join(self.class_stack)
            self.add_method(class_path, node.name)
        elif not self.class_stack and self.function_depth == 0:
            self.add_function(node.name)

        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    def visit_AsyncFunctionDef(self, node):
        if self.class_stack and self.function_depth == 0:
            class_path = ".".join(self.class_stack)
            self.add_method(class_path, node.name)
        elif not self.class_stack and self.function_depth == 0:
            self.add_function(node.name)

        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    def visit_ClassDef(self, node):
        if self.function_depth == 0:
            self.add_class(node.name)

        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()


# ==============================
# PASS 2: 重写代码
# ==============================

class CodeObfuscator(ast.NodeTransformer):

    def __init__(self, mapping):
        self.mapping = mapping
        self.class_stack = []
        self.function_depth = 0

    def _current_class_path(self):
        if not self.class_stack:
            return None
        return ".".join(self.class_stack)

    def _rename_top_level_symbol(self, name):
        if name in self.mapping["functions"]:
            return self.mapping["functions"][name]
        if name in self.mapping["classes"]:
            return self.mapping["classes"][name]
        return name

    # 删除 docstring
    def remove_docstring(self, node):

        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body.pop(0)

    # -------- 函数 --------

    def visit_FunctionDef(self, node):

        self.remove_docstring(node)

        if self.class_stack and self.function_depth == 0:
            class_path = self._current_class_path()
            class_methods = self.mapping["methods"].get(class_path, {})
            if node.name in class_methods:
                node.name = class_methods[node.name]
        elif not self.class_stack and self.function_depth == 0:
            if node.name in self.mapping["functions"]:
                node.name = self.mapping["functions"][node.name]

        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1
        return node

    def visit_AsyncFunctionDef(self, node):

        self.remove_docstring(node)

        if self.class_stack and self.function_depth == 0:
            class_path = self._current_class_path()
            class_methods = self.mapping["methods"].get(class_path, {})
            if node.name in class_methods:
                node.name = class_methods[node.name]
        elif not self.class_stack and self.function_depth == 0:
            if node.name in self.mapping["functions"]:
                node.name = self.mapping["functions"][node.name]

        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1
        return node

    # -------- 类 --------

    def visit_ClassDef(self, node):

        self.remove_docstring(node)

        original_name = node.name
        original_path = ".".join(self.class_stack + [original_name])

        if self.function_depth == 0 and node.name in self.mapping["classes"]:
            node.name = self.mapping["classes"][node.name]

        # 继承类修复
        for base in node.bases:
            if isinstance(base, ast.Name):
                base.id = self._rename_top_level_symbol(base.id)

        self.class_stack.append(original_name)
        self.generic_visit(node)
        self.class_stack.pop()

        renamed_methods = self.mapping["methods"].pop(original_path, None)
        renamed_path = ".".join(self.class_stack + [node.name])
        if renamed_methods is not None:
            self.mapping["methods"][renamed_path] = renamed_methods

        return node

    # -------- 函数调用 --------

    def visit_Call(self, node):

        if isinstance(node.func, ast.Name):
            node.func.id = self._rename_top_level_symbol(node.func.id)

        elif isinstance(node.func, ast.Attribute):
            class_path = self._current_class_path()
            class_methods = self.mapping["methods"].get(class_path, {})
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id in {"self", "cls"}
                and node.func.attr in class_methods
            ):
                node.func.attr = class_methods[node.func.attr]

        self.generic_visit(node)
        return node

    # -------- 名称引用 --------

    def visit_Name(self, node):

        if isinstance(node.ctx, ast.Load):
            node.id = self._rename_top_level_symbol(node.id)

        return node

    # -------- import --------

    def visit_ImportFrom(self, node):

        # 仅处理相对导入，避免误改第三方或标准库导入
        if node.level <= 0:
            return node

        for alias in node.names:
            alias.name = self._rename_top_level_symbol(alias.name)

        return node

    def visit_Import(self, node):
        # import 的 alias.name 是模块路径，不能按函数/类映射替换
        return node

    # -------- __all__ --------

    def visit_Assign(self, node):

        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__all__"
        ):

            if isinstance(node.value, ast.List):

                for elt in node.value.elts:

                    if (
                        isinstance(elt, ast.Constant)
                        and isinstance(elt.value, str)
                    ):
                        elt.value = self._rename_top_level_symbol(elt.value)

            elif isinstance(node.value, ast.Tuple):

                for elt in node.value.elts:

                    if (
                        isinstance(elt, ast.Constant)
                        and isinstance(elt.value, str)
                    ):
                        elt.value = self._rename_top_level_symbol(elt.value)

        self.generic_visit(node)
        return node


# ==============================
# 工具函数
# ==============================

def find_python_files(root):

    files = []

    for dirpath, dirnames, filenames in os.walk(root):

        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for f in filenames:

            if f.endswith(".py"):
                files.append(os.path.join(dirpath, f))

    return files


def collect_definitions(py_files):

    collector = DefinitionCollector()

    for f in py_files:

        try:

            with open(f, "r", encoding="utf-8") as file:
                tree = ast.parse(file.read())

            collector.visit(tree)

        except Exception:
            print(f"skip parse error: {f}")

    return collector.mapping


def rewrite_file(src, dst, mapping):

    try:

        with open(src, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)

        transformer = CodeObfuscator(mapping)

        new_tree = transformer.visit(tree)

        ast.fix_missing_locations(new_tree)

        code = ast.unparse(new_tree)

        os.makedirs(os.path.dirname(dst), exist_ok=True)

        with open(dst, "w", encoding="utf-8") as f:
            f.write(code)

    except Exception as e:

        print("rewrite failed:", src, e)

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)


def copy_non_python(src_root, dst_root):

    for dirpath, dirnames, filenames in os.walk(src_root):

        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for f in filenames:

            if not f.endswith(".py"):

                src = os.path.join(dirpath, f)

                rel = os.path.relpath(src, src_root)

                dst = os.path.join(dst_root, rel)

                os.makedirs(os.path.dirname(dst), exist_ok=True)

                shutil.copy(src, dst)


# ==============================
# 主流程
# ==============================

def obfuscate_project(src_root, dst_root):

    print("Scanning project...")

    py_files = find_python_files(src_root)

    print("Python files:", len(py_files))

    print("Collecting definitions...")

    mapping = collect_definitions(py_files)

    print("Symbols collected:", len(mapping))

    print("Rewriting project...")

    for src in py_files:

        rel = os.path.relpath(src, src_root)

        dst = os.path.join(dst_root, rel)

        rewrite_file(src, dst, mapping)

    copy_non_python(src_root, dst_root)

    with open(os.path.join(dst_root, "mapping.json"), "w") as f:
        json.dump(mapping, f, indent=4, ensure_ascii=False)

    print("Done.")
    print("Output:", dst_root)


# ==============================
# CLI
# ==============================

if __name__ == "__main__":

    if len(sys.argv) != 3:

        print("Usage:")
        print("python obfuscate_project.py input_project output_project")
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2]

    obfuscate_project(src, dst)