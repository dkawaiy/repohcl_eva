import os
import ast
import argparse
import json
import subprocess
from typing import List, Dict, Any, Optional
from prompts import SYSTEM_PROMPTS
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger

llm_logger = logger.bind(channel="llm")


def setup_logger(log_path: str = "logs/agent_for_eva.log") -> None:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    llm_log_path = os.path.join(os.path.dirname(log_path) or ".", "llm.log")
    logger.remove()
    logger.add(
        log_path,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        encoding="utf-8",
        filter=lambda record: record["extra"].get("channel") != "llm",
    )
    logger.add(
        llm_log_path,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        encoding="utf-8",
        filter=lambda record: record["extra"].get("channel") == "llm",
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        filter=lambda record: record["extra"].get("channel") != "llm",
    )


def _truncate_text(text: str, max_len: int = 4000) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "...<truncated>"


def _format_messages_for_log(messages: List[Dict[str, Any]], max_len: int = 4000) -> str:
    try:
        dumped = json.dumps(messages, ensure_ascii=False)
    except Exception:
        dumped = str(messages)
    return _truncate_text(dumped, max_len=max_len)

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
        # 扁平化元数据，保证只包含基本类型
        metadatas = []
        for doc in docs:
            meta = {}
            for k, v in doc.items():
                meta[k] = str(v) if v is not None else ""
            metadatas.append(meta)
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
        logger.info(
            "开始执行命令 | command={} | cwd={} | timeout={} | shell={}",
            command,
            cwd,
            timeout,
            use_shell,
        )
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                timeout=timeout,
                shell=use_shell,
                text=True,
                capture_output=True,
            )
            result = {
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        except Exception as e:
            logger.exception("命令执行异常")
            result = {
                "returncode": -1,
                "stdout": "",
                "stderr": f"Command execution error: {e}",
            }
        logger.info(
            "命令执行完成 | returncode={} | stdout_len={} | stderr_len={} | stdout={} | stderr={}",
            result["returncode"],
            len(result["stdout"] or ""),
            len(result["stderr"] or ""),
            result["stdout"],
            result["stderr"],
        )
        return result

    def write_code_and_run_command(
        self,
        target_path: str,
        code: str,
        command: Any,
        cwd: Optional[str] = None,
        timeout: int = 120,
    ) -> str:
        """将代码写入指定路径并执行命令，返回执行结果。"""
        logger.info("写入代码文件 | target_path={}", target_path)

        try:
            os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as e:
            logger.exception("写入代码文件失败")
            payload = {
                "saved_path": target_path,
                "command": command,
                "cwd": cwd,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Write file error: {e}",
            }
            logger.info("工具执行结果 | payload={}", payload)
            return json.dumps(payload, ensure_ascii=False)

        exec_result = self.run_command(command=command, cwd=cwd, timeout=timeout)
        payload = {
            "saved_path": target_path,
            "command": command,
            "cwd": cwd,
            "returncode": exec_result.get("returncode", -1),
            "stdout": exec_result.get("stdout", ""),
            "stderr": exec_result.get("stderr", ""),
        }
        logger.info("工具执行结果 | payload={}", payload)
        return json.dumps(payload, ensure_ascii=False)


# ==============================
# 代理核心：整合流工作流
# ==============================
class SoftwareAgent:
    def __init__(self, work_mode: str):
        self.work_mode = work_mode
        self.retriever = Retriever()
        self.execution_operator = ExecutionOperator()
        self.base_target_dir = "."

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
        logger.info("收到工具调用 | tool_name={} | arguments={}", tool_name, arguments)
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
            logger.info("检索工具返回 | result_count={}", len(results))
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

            # tool 中 target_path 只作为文件名，最终路径拼接到主流程 target_path 所在目录
            file_name = os.path.basename(str(target_path))
            final_target_path = os.path.join(self.base_target_dir, file_name)

            normalized_command: Any = command
            if isinstance(command, str):
                cmd_text = command.strip()
                if cmd_text.startswith("[") and cmd_text.endswith("]"):
                    try:
                        parsed = json.loads(cmd_text)
                        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                            normalized_command = parsed
                    except json.JSONDecodeError:
                        normalized_command = command

            # 约束命令必须执行刚写入的文件，避免模型传入 /tmp 等错误路径
            if isinstance(normalized_command, list) and normalized_command:
                if normalized_command[0].startswith("python"):
                    if len(normalized_command) == 1:
                        normalized_command.append(final_target_path)
                    else:
                        normalized_command[1] = final_target_path
            elif isinstance(normalized_command, str):
                cmd_text = normalized_command.strip()
                if cmd_text.startswith("python") or cmd_text.startswith("python3"):
                    normalized_command = f"python {final_target_path}"

            return self.execution_operator.write_code_and_run_command(
                target_path=final_target_path,
                code=code,
                command=normalized_command,
                cwd=cwd or self.base_target_dir,
                timeout=timeout,
            )
        else:
            return f"Error: Tool {tool_name} not found."

    def _build_codebase_overview(
        self,
        code_path: str,
        max_files: int = 60,
        max_chars_per_file: int = 2000,
        max_total_chars: int = 30000,
        include_extensions: Optional[List[str]] = None,
    ) -> str:
        """
        构建代码库概览文本，供 plan_and_execute 模式在第一阶段做整体分析。

        - 支持文件或目录输入。
        - 默认只抽取常见代码/配置文件，跳过缓存与依赖目录。
        - 自动进行单文件与总量截断，避免 prompt 过大。
        """
        if not code_path or not os.path.exists(code_path):
            logger.warning("构建代码库概览失败 | 路径不存在: {}", code_path)
            return ""

        exts = include_extensions or [
            ".py", ".md", ".txt", ".json", ".yaml", ".yml",
            ".toml", ".ini", ".cfg", ".sh",
        ]
        allowed_exts = {e.lower() for e in exts}
        skip_dirs = {
            ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
            ".venv", "venv", "env", "node_modules", "dist", "build", "logs",
        }

        file_list: List[str] = []
        if os.path.isfile(code_path):
            file_list = [os.path.abspath(code_path)]
            root_dir = os.path.dirname(os.path.abspath(code_path)) or "."
        else:
            root_dir = os.path.abspath(code_path)
            for cur_root, dirs, files in os.walk(root_dir):
                dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
                for name in files:
                    if name.startswith("."):
                        continue
                    full_path = os.path.join(cur_root, name)
                    if os.path.splitext(name)[1].lower() in allowed_exts:
                        file_list.append(full_path)

        file_list = sorted(file_list)
        if not file_list:
            logger.warning("构建代码库概览失败 | 未找到可读文件: {}", code_path)
            return ""

        selected_files = file_list[:max_files]
        overview_parts: List[str] = []
        total_chars = 0

        header = (
            f"Codebase Root: {os.path.abspath(code_path)}\n"
            f"Collected Files: {len(selected_files)}/{len(file_list)}\n"
            f"(max_files={max_files}, max_chars_per_file={max_chars_per_file}, max_total_chars={max_total_chars})\n"
        )
        overview_parts.append(header)
        total_chars += len(header)

        for idx, file_path in enumerate(selected_files, 1):
            rel_path = os.path.relpath(file_path, root_dir)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception as e:
                logger.warning("读取文件失败 | path={} | error={}", file_path, e)
                content = f"# [ReadError] {e}"

            line_count = content.count("\n") + (1 if content else 0)
            trimmed = content
            if len(trimmed) > max_chars_per_file:
                trimmed = trimmed[:max_chars_per_file] + "\n...<truncated per file>"

            block = (
                f"\n===== File {idx}: {rel_path} (lines={line_count}, chars={len(content)}) =====\n"
                f"{trimmed}\n"
            )

            if total_chars + len(block) > max_total_chars:
                remaining = max_total_chars - total_chars
                if remaining > 200:
                    overview_parts.append(block[:remaining] + "\n...<truncated total>\n")
                else:
                    overview_parts.append("\n...<truncated total>\n")
                break

            overview_parts.append(block)
            total_chars += len(block)

        overview = "".join(overview_parts).strip()
        logger.info(
            "代码库概览构建完成 | code_path={} | files={} | chars={}",
            code_path,
            len(selected_files),
            len(overview),
        )
        return overview

    def _get_plan_from_llm(self, task_desc: str, codebase_overview: str) -> str:
        """
        第一阶段：调用 LLM 以获取解决任务的执行计划。
        """
        print("\n[Phase 1] 开始代码库分析与执行计划生成...")
        load_dotenv()
        api_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
        api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
        model = os.getenv("MODEL", "gpt-3.5-turbo")
        client = OpenAI(api_key=api_key, base_url=api_url)

        system_prompt = SYSTEM_PROMPTS.get("planner", "You are a helpful AI.")
        user_content = (
            f"这是你需要分析的完整代码库：\n\n```python\n{codebase_overview}\n```\n\n"
            f"这是你需要完成的任务目标：\n{task_desc}"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,  # 低温以获取确定性计划
            )
            plan = response.choices[0].message.content
            print(f"    -> LLM 已生成执行计划:\n{plan}")
            logger.info("LLM生成执行计划 | plan={}", plan)
            return plan
        except Exception as e:
            print(f"[!] LLM在生成计划阶段异常: {e}")
            logger.exception("LLM在生成计划阶段异常")
            return "无法生成执行计划，将直接尝试执行任务。"

    def invoke_llm(self, task_desc: str, initial_context: str = "", code_path: str = "") -> str:
        """
        多轮对话代理的核心循环 (Agent Loop with Tool Calling)
        兼容 openai>=1.0.0 新接口。
        """
        load_dotenv()
        api_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
        api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
        model = os.getenv("MODEL", "gpt-3.5-turbo")
        try:
            temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
        except Exception:
            temperature = 0.2
        client = OpenAI(api_key=api_key, base_url=api_url)
        logger.info(
            "初始化LLM客户端 | base_url={} | model={} | temperature={}",
            api_url,
            model,
            temperature,
        )

        print(f"\n[*] 初始化大模型多轮推理代理引擎 (OpenAI协议, model={model}, temperature={temperature})...")

        system_prompt = SYSTEM_PROMPTS.get(self.work_mode, "You are a helpful AI.")
        if self.work_mode == "plan_and_execute":
            system_prompt = SYSTEM_PROMPTS.get("plan_and_execute", "You are a helpful AI.")
        else:
            system_prompt = SYSTEM_PROMPTS.get("default", "You are a helpful AI.") + SYSTEM_PROMPTS.get(self.work_mode, "You are a helpful AI.")
        messages = [{"role": "system", "content": system_prompt}]

        user_content = f"任务目标:\n{task_desc}"
        if code_path:
            user_content += (
                f"\n\n[环境与路径说明]\n"
                f"1. 你的基础代码目录(code_path)为: {code_path}\n"
                f"2. 生成的新代码文件会被存放到独立的输出目录，但**执行该代码时的工作目录(cwd)会被设为基础目录 {code_path}**。\n"
                f"3. 强烈建议：因此如果你要 import 基础目录里的模块，请在生成的代码首部显式添加 `import sys; sys.path.insert(0, '.')`，以避免 ModuleNotFoundError 找不到引用的问题。\n"
            )
        if initial_context:
            user_content += f"\n已提供的上下文信息(Whole Doc):\n{initial_context[:2000]}..."  # 演示截断

        user_content += (
            "\n\n执行约束:\n"
            "1. 你可以参考的内容只有任务文本和 search_knowledge 返回结果。\n"
            "2. 若需要落地代码，必须调用 write_code_and_run_command。\n"
            "3. write_code_and_run_command 的 target_path 只传文件名，例如 demo.py。\n"
            "4. command 必须执行该文件本身。\n"
            # --- 新增 ---
            "5. 当你的工具返回结果中 returncode 为 0，且跑通了要求的业务逻辑后，请直接以普通文本形式输出最终的代码文本，**不要再调用任何工具，以此结束对话**。\n"
            "6. 忽略非致命的系统 stderr 警告（例如 urllib3 的 SSL 警告），如果只是一些不影响业务最终输出的警告，不要因此反复去修复和尝试。"
        )

        messages.append({"role": "user", "content": user_content})

        max_turns = 50
        token_prompt_total = 0
        token_completion_total = 0
        token_total = 0
        for turn in range(1, max_turns + 1):
            print(f"\n  [Loop Turn {turn}] 等待大模型响应...")
            llm_logger.info(
                "对话上下文(发送前) | turn={} | messages={}",
                turn,
                _format_messages_for_log(messages),
            )
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
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
                                            "description": "代码文件名（仅文件名，不要绝对路径），例如 demo.py"
                                        },
                                        "code": {
                                            "type": "string",
                                            "description": "要写入文件的代码内容，只能包含可以直接运行的代码文本，不允许包含任何非代码的说明性文字。"
                                        },
                                        "command": {
                                            "description": "用于执行该代码文件的命令。必须运行刚写入的同一文件。",
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
                msg = response.choices[0].message
                usage = getattr(response, "usage", None)
                if usage is not None:
                    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
                    token_prompt_total += prompt_tokens
                    token_completion_total += completion_tokens
                    token_total += total_tokens
                    logger.info(
                        "LLM token使用 | turn={} | prompt={} | completion={} | total={}",
                        turn,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                    )
                llm_logger.info(
                    "模型响应(原始) | turn={} | content={} | tool_calls={}",
                    turn,
                    _truncate_text(str(getattr(msg, "content", ""))),
                    _truncate_text(str(getattr(msg, "tool_calls", None))),
                )
            except Exception as e:
                print(f"[!] LLM调用异常: {e}")
                logger.exception("LLM调用异常")
                return f"# Error: LLM调用异常: {e}"

            # 工具调用分支
            if getattr(msg, "tool_calls", None):
                assistant_tool_calls = []
                for tc in msg.tool_calls:
                    if hasattr(tc, "model_dump"):
                        assistant_tool_calls.append(tc.model_dump())
                    else:
                        assistant_tool_calls.append({
                            "id": getattr(tc, "id", ""),
                            "type": "function",
                            "function": {
                                "name": getattr(getattr(tc, "function", None), "name", ""),
                                "arguments": getattr(getattr(tc, "function", None), "arguments", "{}"),
                            },
                        })

                messages.append({
                    "role": "assistant",
                    "tool_calls": assistant_tool_calls,
                    "content": msg.content
                })

                for tool_call in msg.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {}

                    print(f"    [LLM 动作]: 决定调用工具 -> {tool_name}({tool_args})")
                    logger.info("模型决定调用工具 | tool_name={} | tool_args={}", tool_name, tool_args)
                    tool_result_str = self.execute_tool(tool_name, tool_args)
                    print(f"    [Tool 结果]: (已成功获取)")
                    logger.info("工具返回结果 | tool_name={} | result={}", tool_name, tool_result_str)
                    # 工具结果回传 LLM
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": tool_result_str
                    })
                continue  # 工具调用后继续下一轮

            # 终态答案分支
            if getattr(msg, "content", None):
                print(f"    [LLM 动作]: 已收集充分信息，输出终态答案。")
                logger.info(
                    "LLM最终输出 | token汇总 prompt={} completion={} total={}",
                    token_prompt_total,
                    token_completion_total,
                    token_total,
                )
                messages.append({"role": "assistant", "content": msg.content})
                return msg.content

        print("[-] 达到最大Agent轮数限制。强制退出。")
        logger.warning(
            "达到最大轮数退出 | token汇总 prompt={} completion={} total={}",
            token_prompt_total,
            token_completion_total,
            token_total,
        )
        return "# Error: Exceeded max loop turns."

    def run(self, task_path: str, doc_path: str, code_path: str, target_path: str) -> bool:
        print(f"==== Agent 启动 ====")
        print(f"Work Mode: {self.work_mode}")
        self.base_target_dir = os.path.abspath(os.path.dirname(target_path) or ".")
        logger.info(
            "Agent启动 | work_mode={} | task_path={} | doc_path={} | code_path={} | target_path={}",
            self.work_mode,
            task_path,
            doc_path,
            code_path,
            target_path,
        )
        
        if os.path.exists(task_path):
            with open(task_path, "r", encoding="utf-8") as f:
                task_desc = f.read()
        else:
            task_desc = "Implement the module logic."

        self.prepare_code_context(code_path)
        self.prepare_doc_context(doc_path)
        
        # 新增：plan_and_execute 模式
        if self.work_mode == "plan_and_execute":
            codebase_overview = self._build_codebase_overview(code_path)
            if codebase_overview:
                plan = self._get_plan_from_llm(task_desc, codebase_overview)
                task_desc = f"任务目标：\n{task_desc}\n\n我的执行计划：\n{plan}"
            else:
                print("[-] 未能构建代码库概览，将按常规模式执行。")

        # 对于 whole_doc，强制注入全量上下文给第一条用户 Prompt
        initial_context = ""
        if self.work_mode == "whole_doc":
            for doc in self.retriever.knowledge_base:
                initial_context += f"{doc.get('content', '')}\n\n"
            
        generated_code = self.invoke_llm(
            task_desc, 
            initial_context=initial_context,
            code_path=code_path
        )

        if not generated_code or generated_code.startswith("# Error:"):
            print("\n[!] 任务失败：LLM 未返回有效代码，已跳过目标文件写入。")
            print(f"[!] 详细信息: {generated_code}")
            logger.error("任务失败 | detail={}", generated_code)
            return False
        
        #os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        #with open(target_path, "w", encoding="utf-8") as f:
        #    f.write(generated_code)
        #print(f"\n[*] 任务完成，目标代码已导出至: {target_path}")
        logger.info("任务完成 | target_path={}", target_path)
        return True

if __name__ == "__main__":
    setup_logger()
    parser = argparse.ArgumentParser(description="Evaluator Agent Core")
    parser.add_argument("--task_path", required=True, help="Path to task description file")
    parser.add_argument("--doc_path", required=True, help="Path to API/Spec documents")
    parser.add_argument("--code_path", required=True, help="Path to base code repository")
    parser.add_argument(
        "--work_mode", 
        required=True, 
        choices=["code_chunk_simple", "code_chunk_complex", "whole_doc", "repohcl_doc_augmentation", "plan_and_execute"],
        help="Strategy for RAG context extraction"
    )
    parser.add_argument("--target_path", required=True, help="Output path for the generated code")
    
    args = parser.parse_args()
    
    agent = SoftwareAgent(work_mode=args.work_mode)
    ok = agent.run(
        task_path=args.task_path,
        doc_path=args.doc_path,
        code_path=args.code_path,
        target_path=args.target_path
    )
    if not ok:
        raise SystemExit(1)