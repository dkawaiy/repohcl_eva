import ast
import os
import json
import shutil
import sys
import re

try:
    from rope.base.project import Project
    from rope.refactor.rename import Rename
except ImportError:
    print("Error: 'rope' library is not installed. Please install it using 'pip install rope'.")
    sys.exit(1)

SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "site-packages",
    ".ropeproject"
}


# ==============================
# PASS 1: 扫描搜集器 (基于AST)
# ==============================
class Collector(ast.NodeVisitor):
    def __init__(self, rel_path):
        self.rel_path = rel_path
        self.current_path = []
        self.jobs = []
        
    def _check_and_add(self, name, kind):
        # 忽略 private 及 magic 方法
        if name.startswith("__"):
            return
        
        path_copy = self.current_path + [name]
        self.jobs.append({
            "filepath": self.rel_path,
            "ast_path": path_copy,
            "old_name": name,
            "kind": kind
        })

    def visit_FunctionDef(self, node):
        self._check_and_add(node.name, "function")
        self.current_path.append(node.name)
        self.generic_visit(node)
        self.current_path.pop()

    def visit_AsyncFunctionDef(self, node):
        self._check_and_add(node.name, "async_function")
        self.current_path.append(node.name)
        self.generic_visit(node)
        self.current_path.pop()

    def visit_ClassDef(self, node):
        self._check_and_add(node.name, "class")
        self.current_path.append(node.name)
        self.generic_visit(node)
        self.current_path.pop()


# ==============================
# PASS 2: 精确定位器 (基于AST定位字符串Offset)
# ==============================
class PathLocator(ast.NodeVisitor):
    def __init__(self, target_path, source):
        self.target_path = target_path
        self.current_path = []
        self.source = source
        self.found_offset = None

        self.line_offsets = [0]
        for line in source.splitlines(True):
            self.line_offsets.append(self.line_offsets[-1] + len(line))

    def _get_char_offset(self, lineno, col_offset):
        if lineno - 1 < len(self.line_offsets):
            return self.line_offsets[lineno - 1] + col_offset
        return len(self.source)

    def _check_and_traverse(self, name, node, kind):
        if self.found_offset is not None:
            return

        self.current_path.append(name)
        
        if self.current_path == self.target_path:
            start_ofs = self._get_char_offset(node.lineno, 0)
            end_lineno = getattr(node, 'end_lineno', None)
            end_col_offset = getattr(node, 'end_col_offset', 0)
            
            if end_lineno:
                end_ofs = self._get_char_offset(end_lineno, end_col_offset)
            else:
                end_ofs = len(self.source)
                
            snippet = self.source[start_ofs:end_ofs]
            
            # 使用正找精确查找 def/class/async def 后面的声明名称
            match = re.search(r'\b(?:def|class|async\s+def)\s+(' + re.escape(name) + r')\b', snippet)
            if match:
                self.found_offset = start_ofs + match.start(1)
            else:
                # 兜底查找
                match = re.search(r'\b' + re.escape(name) + r'\b', snippet)
                if match:
                    self.found_offset = start_ofs + match.start()
            
            self.current_path.pop()
            return

        self.generic_visit(node)
        self.current_path.pop()

    def visit_FunctionDef(self, node):
        self._check_and_traverse(node.name, node, "function")

    def visit_AsyncFunctionDef(self, node):
        self._check_and_traverse(node.name, node, "async_function")

    def visit_ClassDef(self, node):
        self._check_and_traverse(node.name, node, "class")


# ==============================
# 主流程
# ==============================
def obfuscate_project(src_root, dst_root):
    print("1. 准备工作环境 (Copying to destination)...")
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)

    def ignore_patterns(path, names):
        return [n for n in names if n in SKIP_DIRS]

    shutil.copytree(src_root, dst_root, ignore=ignore_patterns)

    print("2. 扫描 Python 文件 (Collecting AST Definitions)...")
    jobs = []
    py_files_count = 0
    for root_dir, dirs, files in os.walk(dst_root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if f.endswith(".py"):
                py_files_count += 1
                full_path = os.path.join(root_dir, f)
                rel_path = os.path.relpath(full_path, dst_root)
                rel_path_rope = rel_path.replace(os.path.sep, '/')
                
                try:
                    with open(full_path, "r", encoding="utf-8") as file_:
                        content = file_.read()
                    
                    tree = ast.parse(content)
                    collector = Collector(rel_path_rope)
                    collector.visit(tree)
                    jobs.extend(collector.jobs)
                except Exception as e:
                    print(f"[-] AST Parse Error in {rel_path}: {e}")

    print(f"找到 {py_files_count} 个 Python 文件，提取出 {len(jobs)} 个待混淆符号。")

    # 分配混淆名
    func_counter = 1
    class_counter = 1
    for job in jobs:
        if job["kind"] == "class":
            job["new_name"] = f"ClassEntity_{class_counter}"
            class_counter += 1
        else:
            job["new_name"] = f"func_op_{func_counter}"
            func_counter += 1

    # 重要：排序作业，按 AST 作用域深度降序。底层的先重命名，防止父级被重命名后导致其无法被找到。
    jobs.sort(key=lambda j: len(j["ast_path"]), reverse=True)

    print("3. 启动 Rope 分析引擎 (Initializing Refactoring Project)...")
    project = Project(dst_root)
    mapping_json = {}

    success_count = 0
    fail_count = 0

    print("开始重命名...")
    for idx, job in enumerate(jobs, 1):
        ast_path = job["ast_path"]
        old_name = job["old_name"]
        new_name = job["new_name"]
        rel_path = job["filepath"]

        try:
            resource = project.get_resource(rel_path)
            source = resource.read()
            
            # 使用针对当前最新代码文本的精确定位器
            locator = PathLocator(ast_path, source)
            tree = ast.parse(source)
            locator.visit(tree)
            
            offset = locator.found_offset
            if offset is not None:
                renamer = Rename(project, resource, offset)
                changes = renamer.get_changes(new_name)
                project.do(changes)
                
                path_str = ".".join(ast_path)
                mapping_json[f"{rel_path}::{path_str}"] = {
                    "old": old_name,
                    "new": new_name
                }
                success_count += 1
                if idx % 50 == 0:
                    print(f"  ... 已处理 {idx}/{len(jobs)} 项")
            else:
                fail_count += 1
                print(f"[-] 无法找到符号精确位点: {old_name} in {rel_path}")

        except Exception as e:
            fail_count += 1
            print(f"[-] Rope 操作失败: {ast_path} in {rel_path} -> {e}")

    project.close()
    
    print(f"重命名完成，成功: {success_count}，失败/跳过: {fail_count}。")
    print("4. 输出 Mapping 文件 (Saving mapping.json)...")
    map_path = os.path.join(dst_root, "mapping.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(mapping_json, f, indent=4, ensure_ascii=False)

    print(f"Done! 输出目录: {dst_root}")


# ==============================
# CLI
# ==============================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python obfuscate_rope.py <input_project> <output_project>")
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2]
    obfuscate_project(src, dst)
