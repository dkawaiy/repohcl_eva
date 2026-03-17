import os
import ast
import argparse
import json
import subprocess
from typing import List, Dict, Any, Optional
from prompts import SYSTEM_PROMPTS
import openai
from dotenv import load_dotenv

# ==============================
# 文档分块处理模块
# ==============================
class DocChunker:
    @staticmethod
    def split_markdown_chunks(md_text: str, file_path: str) -> list:
        """
        将 markdown 文本按 ### 标题分割为多个 chunk。
        - 每个 chunk 以 '### ' 开头，包含该标题及其下属内容，直到下一个 '### ' 或文档结尾。
        - 若文档没有任何 '###'，则整体作为一个 chunk。
        - 每个 chunk 记录 type、title、content、file_path、order 等元信息。
        """
        import re
        pattern = re.compile(r'(^### .*$)', re.MULTILINE)
        matches = list(pattern.finditer(md_text))
        chunks = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i+1].start() if i+1 < len(matches) else len(md_text)
            chunk_text = md_text[start:end].strip()
            title = match.group(1).strip().lstrip('#').strip()  # 提取标题文本
            chunks.append({
                "type": "markdown_chunk",  # 标记为 markdown_chunk 类型
                "title": title,             # 该 chunk 的标题（去除 #）
                "content": chunk_text,      # chunk 的全部内容
                "file_path": file_path,     # 来源文件路径
                "order": i                  # 在当前文件中的顺序编号
            })
        # 若文档没有任何 '###'，整体作为一个 chunk
        if not chunks and md_text.strip():
            chunks.append({
                "type": "markdown_chunk",
                "title": None,
                "content": md_text.strip(),
                "file_path": file_path,
                "order": 0
            })
        return chunks


# ==============================
# 代码和文档的分块与处理处理模块
# ==============================
class CodeChunker:
    @staticmethod
    def chunk_simple(code_text: str, chunk_size: int = 50) -> List[Dict[str, Any]]:
        """按固定行数进行简单拆分"""
        lines = code_text.splitlines()
        chunks = []
        for i in range(0, len(lines), chunk_size):
            chunk = "\n".join(lines[i:i + chunk_size])
            chunks.append({
                "type": "code_chunk_simple", 
                "content": chunk, 
                "start_line": i + 1
            })
        return chunks

    @staticmethod
    def chunk_complex(code_text: str, file_path: str = None) -> List[Dict[str, Any]]:
        """基于AST树进行高级语义拆分（函数、类粒度），并补全全局代码块，记录顺序"""
        chunks = []
        try:
            tree = ast.parse(code_text)
            lines = code_text.splitlines()
            covered = [False] * len(lines)
            order = 0
            node_order = []  # (order, start, end, chunk_dict)
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start_lineno = node.lineno - 1
                    end_lineno = getattr(node, 'end_lineno', node.lineno)
                    content = "\n".join(lines[start_lineno:end_lineno])
                    chunk = {
                        "type": "code_chunk_complex",
                        "name": node.name,
                        "content": content,
                        "start_line": start_lineno + 1,
                        "end_line": end_lineno,
                        "file_path": file_path,
                        "order": order
                    }
                    node_order.append((order, start_lineno, end_lineno, chunk))
                    order += 1
                    for i in range(start_lineno, end_lineno):
                        if 0 <= i < len(covered):
                            covered[i] = True
            # 补全未被覆盖的全局代码
            global_lines = [i for i, cov in enumerate(covered) if not cov]
            if global_lines:
                from itertools import groupby
                from operator import itemgetter
                for _, group in groupby(enumerate(global_lines), lambda x: x[0] - x[1]):
                    group_list = list(map(itemgetter(1), group))
                    start = group_list[0]
                    end = group_list[-1] + 1
                    content = "\n".join(lines[start:end])
                    if content.strip():
                        chunk = {
                            "type": "code_chunk_global",
                            "content": content,
                            "start_line": start + 1,
                            "end_line": end,
                            "file_path": file_path,
                            "order": order
                        }
                        node_order.append((order, start, end, chunk))
                        order += 1
            # 按order排序，保证顺序还原
            node_order.sort(key=lambda x: x[0])
            chunks = [item[3] for item in node_order]
        except SyntaxError:
            pass
        return chunks

# ==============================
# 检索模块 (Embeddings / VectorStore)
# ==============================
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

class Retriever:
    def __init__(self, persist_dir: str = ".chroma_store"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False
        ))
        self.collection = self.chroma_client.get_or_create_collection("eva_docs")
        self.doc_id_counter = 0

    def add_documents(self, docs: List[Dict[str, Any]]):
        texts = [doc.get("content", "") for doc in docs]
        embeddings = self.model.encode(texts, show_progress_bar=False).tolist()
        ids = [f"doc_{self.doc_id_counter + i}" for i in range(len(docs))]
        self.doc_id_counter += len(docs)
        # 存储原始文档内容和元数据
        metadatas = docs
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_emb = self.model.encode([query], show_progress_bar=False).tolist()[0]
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k
        )
        # 返回 metadatas（即原始 doc 信息）
        return results.get("metadatas", [[]])[0] if results.get("metadatas") else []

    # 根据文件路径和起止行号检索源码块
    def fetch_row_data(self, file_path: str, start_line: int, end_line: int) -> list:
        """
        只根据文件路径和起止行号检索源码块。
        :param file_path: 文件路径
        :param start_line: 起始行号（1-based）
        :param end_line: 结束行号（1-based）
        :return: 匹配的文档列表
        """
        all_docs = self.collection.get()["metadatas"] if self.collection.count() > 0 else []
        results = []
        for doc in all_docs:
            if doc.get("file_path") != file_path:
                continue
            chunk_start = doc.get("start_line", 0)
            chunk_end = chunk_start + doc.get("content", "").count("\n")
            # 判断是否有重叠
            if chunk_start <= end_line and chunk_end >= start_line:
                results.append(doc)
        return results

# ==============================
# 执行操作模块
# ==============================
class ExecutionOperator:
    def run_command(
        self,
        command: Any,
        cwd: Optional[str] = None,
        timeout: int = 120,
    ) -> Dict[str, Any]:
        """使用 subprocess 运行命令并返回统一结构结果。"""
        use_shell = isinstance(command, str)
        completed = subprocess.run(
            command,
            cwd=cwd,
            timeout=timeout,
            shell=use_shell,
            text=True,
            capture_output=True,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

    def write_code_and_run_command(
        self,
        target_path: str,
        code: str,
        command: Any,
        cwd: Optional[str] = None,
        timeout: int = 120,
    ) -> str:
        """将代码写入指定路径并执行命令，返回执行结果。"""
        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(code)

        exec_result = self.run_command(command=command, cwd=cwd, timeout=timeout)
        payload = {
            "saved_path": target_path,
            "command": command,
            "cwd": cwd,
            "returncode": exec_result.get("returncode", -1),
            "stdout": exec_result.get("stdout", ""),
            "stderr": exec_result.get("stderr", ""),
        }
        return json.dumps(payload, ensure_ascii=False)


# ==============================
# 代理核心：整合流工作流
# ==============================
class SoftwareAgent:
    def __init__(self, work_mode: str):
        self.work_mode = work_mode
        self.retriever = Retriever()
        self.execution_operator = ExecutionOperator()

    def prepare_code_context(self, code_path: str):
        """加载并处理代码库"""
        if not os.path.exists(code_path):
            print(f"[-] Code path not found: {code_path}")
            return
        
        if os.path.isfile(code_path):
            with open(code_path, "r", encoding="utf-8") as f:
                code_text = f.read()

            if self.work_mode == "code_chunk_simple":
                chunks = CodeChunker.chunk_simple(code_text)
            elif self.work_mode in ["code_chunk_complex", "repohcl_doc_augmentation"]:
                chunks = CodeChunker.chunk_complex(code_text)
            else:
                chunks = [{"type": "whole_code", "content": code_text}]
                
            self.retriever.add_documents(chunks)

    def prepare_doc_context(self, doc_path: str):
        """
        加载 markdown 文档，将每个 ### 单元分 chunk，支持目录递归。
        每个 chunk 记录标题、顺序、文件路径等信息，便于后续检索和结构化索引。
        """
        if not os.path.exists(doc_path):
            print(f"[-] Doc path not found: {doc_path}")
            return

        doc_chunks = []  # 存储所有分割后的文档块

        def process_file(file_path):
            """
            处理单个 markdown 文件，将其分割为 chunk 并加入 doc_chunks。
            只处理 .md 文件，其他类型跳过。
            """
            if not file_path.endswith('.md'):
                return
            with open(file_path, "r", encoding="utf-8") as f:
                doc_text = f.read()
            # 分割并收集 chunk
            doc_chunks.extend(DocChunker.split_markdown_chunks(doc_text, file_path))

        def walk_dir(base_path):
            """
            递归遍历目录，处理所有子目录和 .md 文件。
            """
            for entry in os.scandir(base_path):
                if entry.is_file():
                    process_file(entry.path)
                elif entry.is_dir():
                    walk_dir(entry.path)

        # 判断输入路径类型，分别处理单文件或目录
        if os.path.isfile(doc_path):
            process_file(doc_path)
        elif os.path.isdir(doc_path):
            walk_dir(doc_path)

        # 将所有分割后的 chunk 加入向量库/检索器
        self.retriever.add_documents(doc_chunks)

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """执行本地工具 (Tool Execution)"""
        if tool_name == "search_knowledge":
            query = arguments.get("query", "")
            print(f"      -> 正在检索关键词: '{query}'")
            results = self.retriever.search(query, top_k=3)
            # 格式化检索结果供模型阅读
            if not results:
                return "没有找到相关的代码或文档信息。"
            
            output = ""
            for idx, res in enumerate(results, 1):
                output += f"\n--- 结果 {idx} ({res.get('type')}) ---\n{res.get('content', '')[:300]}..."
            return output
        elif tool_name == "write_code_and_run_command":
            target_path = arguments.get("target_path")
            code = arguments.get("code", "")
            command = arguments.get("command")
            cwd = arguments.get("cwd")
            timeout = int(arguments.get("timeout", 120))

            if not target_path:
                return "Error: missing required argument 'target_path'."
            if command is None:
                return "Error: missing required argument 'command'."

            return self.execution_operator.write_code_and_run_command(
                target_path=target_path,
                code=code,
                command=command,
                cwd=cwd,
                timeout=timeout,
            )
        else:
            return f"Error: Tool {tool_name} not found."

    def invoke_llm(self, task_desc: str, initial_context: str = "") -> str:
        """
        多轮对话代理的核心循环 (Agent Loop with Tool Calling)
        通过 openai 协议调用本地/远程大模型，支持工具调用。
        配置项通过 .env 文件控制（如 OPENAI_API_BASE, OPENAI_API_KEY, OPENAI_MODEL 等）。
        """
        # 加载 .env 配置

        load_dotenv()
        openai.api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
        openai.api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        try:
            temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
        except Exception:
            temperature = 0.2

        print(f"\n[*] 初始化大模型多轮推理代理引擎 (OpenAI协议, model={model}, temperature={temperature})...")

        system_prompt = SYSTEM_PROMPTS.get(self.work_mode, "You are a helpful AI.")
        messages = [{"role": "system", "content": system_prompt}]

        user_content = f"任务目标:\n{task_desc}"
        if initial_context:
            user_content += f"\n\n已提供的上下文信息(Whole Doc):\n{initial_context[:2000]}..."  # 演示截断

        messages.append({"role": "user", "content": user_content})

        max_turns = 4
        for turn in range(1, max_turns + 1):
            print(f"\n  [Loop Turn {turn}] 等待大模型响应...")
            try:
                # 支持工具调用的 openai 协议（vllm/openai 兼容）
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    stream=False,
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "search_knowledge",
                                "description": "检索知识库中的相关代码或文档片段",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string", "description": "检索关键词"}
                                    },
                                    "required": ["query"]
                                }
                            }
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "write_code_and_run_command",
                                "description": "将代码写入指定路径，并在指定工作目录执行命令，返回stdout/stderr/returncode。",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "target_path": {
                                            "type": "string",
                                            "description": "代码写入的目标路径"
                                        },
                                        "code": {
                                            "type": "string",
                                            "description": "要写入文件的代码内容"
                                        },
                                        "command": {
                                            "description": "执行命令，可为字符串或字符串数组",
                                            "anyOf": [
                                                {"type": "string"},
                                                {
                                                    "type": "array",
                                                    "items": {"type": "string"}
                                                }
                                            ]
                                        },
                                        "cwd": {
                                            "type": "string",
                                            "description": "命令执行目录，可选"
                                        },
                                        "timeout": {
                                            "type": "integer",
                                            "description": "命令超时时间（秒）",
                                            "default": 120
                                        }
                                    },
                                    "required": ["target_path", "code", "command"]
                                }
                            }
                        }
                    ]
                )
                msg = response["choices"][0]["message"]
            except Exception as e:
                print(f"[!] LLM调用异常: {e}")
                return f"# Error: LLM调用异常: {e}"

            # 工具调用分支
            if "tool_calls" in msg and msg["tool_calls"]:
                messages.append({
                    "role": "assistant",
                    "tool_calls": msg["tool_calls"],
                    "content": None
                })
                for tool_call in msg["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"].get("arguments", {})
                    # openai协议下 arguments 可能是字符串需解析
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {}

                    print(f"    [LLM 动作]: 决定调用工具 -> {tool_name}({tool_args})")
                    tool_result_str = self.execute_tool(tool_name, tool_args)
                    print(f"    [Tool 结果]: (已成功获取)")
                    # 工具结果回传 LLM
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_name,
                        "content": tool_result_str
                    })
                continue  # 工具调用后继续下一轮

            # 终态答案分支
            if msg.get("content"):
                print(f"    [LLM 动作]: 已收集充分信息，输出终态答案。")
                messages.append({"role": "assistant", "content": msg["content"]})
                return msg["content"]

        print("[-] 达到最大Agent轮数限制。强制退出。")
        return "# Error: Exceeded max loop turns."

    def run(self, task_path: str, doc_path: str, code_path: str, target_path: str):
        print(f"==== Agent 启动 ====")
        print(f"Work Mode: {self.work_mode}")
        
        if os.path.exists(task_path):
            with open(task_path, "r", encoding="utf-8") as f:
                task_desc = f.read()
        else:
            task_desc = "Implement the module logic."

        self.prepare_code_context(code_path)
        self.prepare_doc_context(doc_path)
        
        # 对于 whole_doc，强制注入全量上下文给第一条用户 Prompt
        initial_context = ""
        if self.work_mode == "whole_doc":
            for doc in self.retriever.knowledge_base:
                initial_context += f"{doc.get('content', '')}\n\n"
            
        generated_code = self.invoke_llm(task_desc, initial_context=initial_context)
        
        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(generated_code)
        print(f"\n[*] 任务完成，目标代码已导出至: {target_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluator Agent Core")
    parser.add_argument("--task_path", required=True, help="Path to task description file")
    parser.add_argument("--doc_path", required=True, help="Path to API/Spec documents")
    parser.add_argument("--code_path", required=True, help="Path to base code repository")
    parser.add_argument(
        "--work_mode", 
        required=True, 
        choices=["code_chunk_simple", "code_chunk_complex", "whole_doc", "repohcl_doc_augmentation"],
        help="Strategy for RAG context extraction"
    )
    parser.add_argument("--target_path", required=True, help="Output path for the generated code")
    
    args = parser.parse_args()
    
    agent = SoftwareAgent(work_mode=args.work_mode)
    agent.run(
        task_path=args.task_path,
        doc_path=args.doc_path,
        code_path=args.code_path,
        target_path=args.target_path
    )