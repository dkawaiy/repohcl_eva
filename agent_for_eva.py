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
    def chunk_simple(code_text: str, chunk_size: int = 50, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """按固定行数进行简单拆分"""
        lines = code_text.splitlines()
        chunks = []
        for i in range(0, len(lines), chunk_size):
            chunk = "\n".join(lines[i:i + chunk_size])
            chunks.append({
                "type": "code_chunk_simple", 
                "content": chunk, 
                "start_line": i + 1,
                "end_line": min(i + chunk_size, len(lines)),
                "file_path": file_path,
                "order": i // chunk_size,
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
        embedding_model_name = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
        self.model = SentenceTransformer(embedding_model_name)
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=persist_dir,
            anonymized_telemetry=False
        ))
        self.collections = {
            "chunk": self.chroma_client.get_or_create_collection("eva_docs_chunk"),
            "symbol": self.chroma_client.get_or_create_collection("eva_docs_symbol"),
            "module": self.chroma_client.get_or_create_collection("eva_docs_module"),
            "repo": self.chroma_client.get_or_create_collection("eva_docs_repo"),
        }
        self.doc_id_counters = {"chunk": 0, "symbol": 0, "module": 0, "repo": 0}
        logger.info("检索模型加载完成 | model={}", embedding_model_name)

    def _normalize_meta(self, doc: Dict[str, Any], level: str) -> Dict[str, str]:
        meta: Dict[str, str] = {"level": level}
        for k, v in doc.items():
            meta[k] = str(v) if v is not None else ""
        return meta

    def _add_to_collection(self, level: str, docs: List[Dict[str, Any]]):
        if not docs:
            return
        texts = [str(doc.get("content", "")) for doc in docs if str(doc.get("content", "")).strip()]
        if not texts:
            return

        filtered_docs = [doc for doc in docs if str(doc.get("content", "")).strip()]
        embeddings = self.model.encode(texts, show_progress_bar=False).tolist()
        start_id = self.doc_id_counters[level]
        ids = [f"{level}_{start_id + i}" for i in range(len(filtered_docs))]
        self.doc_id_counters[level] += len(filtered_docs)
        metadatas = [self._normalize_meta(doc, level=level) for doc in filtered_docs]
        self.collections[level].add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

    def _build_symbol_docs(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        symbol_docs: List[Dict[str, Any]] = []
        for doc in docs:
            symbol_name = str(doc.get("name") or doc.get("title") or "").strip()
            if not symbol_name:
                continue
            content = str(doc.get("content", ""))
            symbol_docs.append({
                "type": "symbol_summary",
                "symbol_name": symbol_name,
                "file_path": str(doc.get("file_path", "")),
                "content": f"{symbol_name}\n{content[:1600]}",
            })
        return symbol_docs

    def _build_module_docs(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for doc in docs:
            file_path = str(doc.get("file_path") or "").strip()
            if not file_path:
                continue
            grouped.setdefault(file_path, []).append(doc)

        module_docs: List[Dict[str, Any]] = []
        for file_path, items in grouped.items():
            symbol_names = []
            content_parts = []
            for it in items[:8]:
                name = str(it.get("name") or it.get("title") or "").strip()
                if name:
                    symbol_names.append(name)
                content_parts.append(str(it.get("content", ""))[:500])
            symbol_preview = ", ".join(symbol_names[:12])
            merged = "\n".join(content_parts)[:2400]
            module_docs.append({
                "type": "module_summary",
                "module_path": file_path,
                "symbol_preview": symbol_preview,
                "file_path": file_path,
                "content": f"module={file_path}\nsymbols={symbol_preview}\n{merged}",
            })
        return module_docs

    def _build_repo_doc(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        file_paths = sorted({str(doc.get("file_path") or "").strip() for doc in docs if str(doc.get("file_path") or "").strip()})
        symbols = sorted({str(doc.get("name") or doc.get("title") or "").strip() for doc in docs if str(doc.get("name") or doc.get("title") or "").strip()})
        if not file_paths and not symbols:
            return []
        repo_text = (
            "repo_overview\n"
            + "files:\n" + "\n".join(file_paths[:300])
            + "\n\nsymbols:\n" + "\n".join(symbols[:500])
        )[:5000]
        return [{
            "type": "repo_summary",
            "file_path": "<repo>",
            "content": repo_text,
            "file_count": len(file_paths),
            "symbol_count": len(symbols),
        }]

    def add_documents(self, docs: List[Dict[str, Any]]):
        if not docs:
            return
        # 全量内容默认进入 chunk 层，保证兜底可召回。
        self._add_to_collection("chunk", docs)

        repo_hint_docs: List[Dict[str, Any]] = []
        module_hint_docs: List[Dict[str, Any]] = []
        symbol_hint_docs: List[Dict[str, Any]] = []
        other_docs: List[Dict[str, Any]] = []

        for doc in docs:
            hinted_level = str(doc.get("doc_level") or "").strip().lower()
            if hinted_level == "repo":
                repo_hint_docs.append(doc)
            elif hinted_level == "module":
                module_hint_docs.append(doc)
            elif hinted_level == "symbol":
                symbol_hint_docs.append(doc)
            else:
                other_docs.append(doc)

        # 若文档显式给出了层级，优先直接入对应层，减少二次总结损耗。
        self._add_to_collection("repo", repo_hint_docs)
        self._add_to_collection("module", module_hint_docs)
        self._add_to_collection("symbol", symbol_hint_docs)

        # 其余内容再走自动聚合构建。
        self._add_to_collection("symbol", self._build_symbol_docs(other_docs))
        self._add_to_collection("module", self._build_module_docs(other_docs))
        self._add_to_collection("repo", self._build_repo_doc(other_docs))

    def _query_level(self, query_emb: List[float], level: str, top_k: int) -> List[Dict[str, Any]]:
        collection = self.collections.get(level)
        if collection is None or collection.count() <= 0:
            return []
        n_results = max(1, min(int(top_k), 50))
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=n_results,
        )
        return results.get("metadatas", [[]])[0] if results.get("metadatas") else []

    def search(self, query: str, top_k: int = 8, level: str = "auto") -> List[Dict[str, Any]]:
        requested_level = str(level or "auto").strip().lower()
        query_emb = self.model.encode([query], show_progress_bar=False).tolist()[0]
        if requested_level in {"repo", "module", "symbol", "chunk"}:
            return self._query_level(query_emb, requested_level, top_k)

        # auto: 自上而下分层检索，优先返回高层概览，再补充细粒度片段。
        merged: List[Dict[str, Any]] = []
        merged.extend(self._query_level(query_emb, "repo", min(1, top_k)))
        merged.extend(self._query_level(query_emb, "module", min(3, top_k)))
        merged.extend(self._query_level(query_emb, "symbol", min(4, top_k)))
        merged.extend(self._query_level(query_emb, "chunk", top_k))

        dedup: List[Dict[str, Any]] = []
        seen = set()
        for item in merged:
            key = (
                str(item.get("level", "")),
                str(item.get("file_path", "")),
                str(item.get("module_path", "")),
                str(item.get("symbol_name", "")),
                str(item.get("name", "")),
                str(item.get("title", "")),
                str(item.get("start_line", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            dedup.append(item)
        return dedup[: max(1, min(top_k * 2, 40))]

    # 根据文件路径和起止行号检索源码块
    def fetch_row_data(self, file_path: str, start_line: int, end_line: int) -> list:
        """
        只根据文件路径和起止行号检索源码块。
        :param file_path: 文件路径
        :param start_line: 起始行号（1-based）
        :param end_line: 结束行号（1-based）
        :return: 匹配的文档列表
        """
        chunk_collection = self.collections.get("chunk")
        all_docs = chunk_collection.get()["metadatas"] if chunk_collection and chunk_collection.count() > 0 else []
        results = []
        for doc in all_docs:
            if doc.get("file_path") != file_path:
                continue
            try:
                chunk_start = int(doc.get("start_line", 0) or 0)
            except Exception:
                chunk_start = 0
            chunk_end = chunk_start + str(doc.get("content", "")).count("\n")
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

    def write_code_file(self, target_path: str, code: str) -> Dict[str, Any]:
        """仅将代码写入指定路径，返回统一结构结果。"""
        logger.info("写入代码文件 | target_path={}", target_path)

        try:
            os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(code)
            payload = {
                "saved_path": target_path,
                "ok": True,
                "error": "",
            }
        except Exception as e:
            logger.exception("写入代码文件失败")
            payload = {
                "saved_path": target_path,
                "ok": False,
                "error": f"Write file error: {e}",
            }
        logger.info("工具执行结果 | payload={}", payload)
        return payload

    def delete_code_file(self, target_path: str) -> Dict[str, Any]:
        """删除指定文件（仅文件），返回统一结构结果。"""
        logger.info("删除代码文件 | target_path={}", target_path)
        try:
            if not os.path.exists(target_path):
                payload = {
                    "deleted_path": target_path,
                    "ok": True,
                    "error": "",
                    "existed": False,
                }
            elif os.path.isdir(target_path):
                payload = {
                    "deleted_path": target_path,
                    "ok": False,
                    "error": "Delete file error: target is a directory.",
                    "existed": True,
                }
            else:
                os.remove(target_path)
                payload = {
                    "deleted_path": target_path,
                    "ok": True,
                    "error": "",
                    "existed": True,
                }
        except Exception as e:
            logger.exception("删除代码文件失败")
            payload = {
                "deleted_path": target_path,
                "ok": False,
                "error": f"Delete file error: {e}",
                "existed": True,
            }
        logger.info("工具执行结果 | payload={}", payload)
        return payload


# ==============================
# 代理核心：整合流工作流
# ==============================
class SoftwareAgent:
    def __init__(self, work_mode: str, project_root: str):
        self.work_mode = work_mode
        self.project_root = os.path.abspath(project_root)
        self.retriever = Retriever(persist_dir=os.path.join(self.project_root, ".chroma_store"))
        self.execution_operator = ExecutionOperator()
        self.base_target_dir = "."
        self.base_code_dir = "."
        self.command_policy_path = os.path.join(self.project_root, "command_whitelist.json")
        self.command_policies = self._load_command_policies(self.command_policy_path)
        self.command_policy_projects = self._load_command_policy_projects(self.command_policy_path)
        self.active_command_policy: Dict[str, Any] = {}
        self.active_allowed_command_names: Optional[set] = None

        # 根据 work_mode 初始化 LLM 工具相关配置
        if self.work_mode in ["code_chunk_simple", "code_chunk_complex", "plan_and_execute"]:
            self.tools_config_path = os.path.join(self.project_root, "llm_tools_code.json")
            llm_tools_config = self._load_llm_tools_config(self.tools_config_path)
            self.llm_tools_enabled = llm_tools_config.get("enabled", True)
            self.llm_tools = llm_tools_config.get("tools", [])
        else:
            self.tools_config_path = os.path.join(self.project_root, "llm_tools_doc.json")
            llm_tools_config = self._load_llm_tools_config(self.tools_config_path)
            self.llm_tools_enabled = llm_tools_config.get("enabled", True)
            self.llm_tools = llm_tools_config.get("tools", [])
        
        self.file_edit_policy_path = os.path.join(self.project_root, "file_edit_whitelist.json")
        self.file_edit_policies = self._load_file_edit_policies(self.file_edit_policy_path)
        self.active_project_edit_policy: Dict[str, Any] = {}
        self.search_query_counts: Dict[str, int] = {}
        self.search_result_signature_cache: Dict[str, str] = {}
        self.successful_write_count: int = 0
        self.last_successful_write_path: Optional[str] = None
        self.last_executed_command_name: Optional[str] = None
        self.last_execute_write_count: int = 0

    def _is_path_within(self, base_dir: str, target_path: str) -> bool:
        base_abs = os.path.abspath(base_dir)
        target_abs = os.path.abspath(target_path)
        return target_abs == base_abs or target_abs.startswith(base_abs + os.sep)

    def _require_absolute_path(self, path_value: Any, field_name: str) -> str:
        text = str(path_value or "").strip()
        if not text:
            raise ValueError(f"{field_name} cannot be empty")
        if not os.path.isabs(text):
            raise ValueError(f"{field_name} must be an absolute path")
        return os.path.abspath(text)

    def _match_absolute_marker(self, marker: str, *paths: str) -> bool:
        """按绝对路径 marker 匹配多个路径。"""
        try:
            marker_abs = self._require_absolute_path(marker, "match_paths marker")
        except Exception:
            return False

        for p in paths:
            p_abs = os.path.abspath(str(p or ""))
            if p_abs == marker_abs:
                return True
            if self._is_path_within(marker_abs, p_abs):
                return True
            if self._is_path_within(p_abs, marker_abs):
                return True
        return False

    def _load_command_policies(self, path: str) -> Dict[str, Dict[str, Any]]:
        """加载命令白名单配置。"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            policies = raw.get("allowed_commands", {}) if isinstance(raw, dict) else {}
            if not isinstance(policies, dict):
                logger.warning("命令白名单格式错误，已忽略 | path={}", path)
                return {}
            logger.info("命令白名单加载完成 | path={} | count={}", path, len(policies))
            return policies
        except FileNotFoundError:
            logger.warning("命令白名单文件不存在 | path={}", path)
            return {}
        except Exception as e:
            logger.exception("加载命令白名单失败 | path={} | error={}", path, e)
            return {}

    def _load_file_edit_policies(self, path: str) -> Dict[str, Any]:
        """加载文件编辑白名单配置，支持顶层与 projects 两种格式。"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                logger.warning("文件编辑白名单格式错误，已忽略 | path={}", path)
                return {}

            projects = raw.get("projects", {})
            if projects and not isinstance(projects, dict):
                logger.warning("文件编辑白名单 projects 字段格式错误，已忽略 | path={}", path)
                return {}

            top_target = raw.get("editable_files_under_target", [])
            top_code = raw.get("editable_files_under_code", [])
            if top_target and not isinstance(top_target, list):
                logger.warning("文件编辑白名单 editable_files_under_target 格式错误，已忽略 | path={}", path)
                return {}
            if top_code and not isinstance(top_code, list):
                logger.warning("文件编辑白名单 editable_files_under_code 格式错误，已忽略 | path={}", path)
                return {}

            logger.info(
                "文件编辑白名单加载完成 | path={} | project_count={} | top_target_count={} | top_code_count={}",
                path,
                len(projects) if isinstance(projects, dict) else 0,
                len(top_target) if isinstance(top_target, list) else 0,
                len(top_code) if isinstance(top_code, list) else 0,
            )
            return raw
        except FileNotFoundError:
            logger.warning("文件编辑白名单文件不存在 | path={}", path)
            return {}
        except Exception as e:
            logger.exception("加载文件编辑白名单失败 | path={} | error={}", path, e)
            return {}

    def _load_llm_tools_config(self, path: str) -> Dict[str, Any]:
        """加载 LLM 工具定义配置。"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            if isinstance(raw, dict):
                enabled = bool(raw.get("enable_tools", True))
                tools = raw.get("tools", [])
                enabled_tool_names = raw.get("enabled_tool_names", [])
            elif isinstance(raw, list):
                enabled = True
                tools = raw
                enabled_tool_names = []
            else:
                logger.warning("LLM工具配置格式错误，已忽略 | path={}", path)
                return {"enabled": True, "tools": []}

            if not isinstance(tools, list):
                logger.warning("LLM工具配置 tools 字段格式错误，已忽略 | path={}", path)
                return {"enabled": enabled, "tools": []}

            if not isinstance(enabled_tool_names, list):
                enabled_tool_names = []

            valid_tools: List[Dict[str, Any]] = [t for t in tools if isinstance(t, dict)]
            if enabled_tool_names:
                enabled_name_set = {
                    str(x) for x in enabled_tool_names
                    if isinstance(x, str) and str(x).strip()
                }
                valid_tools = [
                    t for t in valid_tools
                    if str(t.get("function", {}).get("name", "")) in enabled_name_set
                ]
            logger.info(
                "LLM工具配置加载完成 | path={} | enabled={} | enabled_tool_names={} | count={}",
                path,
                enabled,
                enabled_tool_names,
                len(valid_tools),
            )
            return {"enabled": enabled, "tools": valid_tools}
        except FileNotFoundError:
            logger.warning("LLM工具配置文件不存在 | path={}", path)
            return {"enabled": True, "tools": []}
        except Exception as e:
            logger.exception("加载LLM工具配置失败 | path={} | error={}", path, e)
            return {"enabled": True, "tools": []}

    def _load_command_policy_projects(self, path: str) -> Dict[str, Any]:
        """加载项目级命令白名单配置。"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                logger.warning("命令白名单 projects 格式错误，已忽略 | path={}", path)
                return {}
            projects = raw.get("projects", {})
            if not isinstance(projects, dict):
                logger.warning("命令白名单 projects 字段格式错误，已忽略 | path={}", path)
                return {}
            logger.info("命令白名单项目配置加载完成 | path={} | project_count={}", path, len(projects))
            return projects
        except FileNotFoundError:
            logger.warning("命令白名单文件不存在(项目配置) | path={}", path)
            return {}
        except Exception as e:
            logger.exception("加载命令白名单项目配置失败 | path={} | error={}", path, e)
            return {}

    def _select_command_policy(self, task_path: str, code_path: str) -> Dict[str, Any]:
        """不再使用 match_paths，统一合并所有项目的 allowed_commands。"""
        projects = self.command_policy_projects if isinstance(self.command_policy_projects, dict) else {}
        if not projects:
            logger.info("未配置项目级命令限制，使用 allowed_commands 全量命令")
            return {}

        merged_allowed: set = set()
        for project_name, cfg in projects.items():
            if not isinstance(cfg, dict):
                continue
            items = cfg.get("allowed_commands", [])
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, str) and item in self.command_policies:
                    merged_allowed.add(item)

        if not merged_allowed:
            logger.info("项目级命令限制为空，使用 allowed_commands 全量命令")
            return {}

        selected = {
            "project_name": "ALL_PROJECTS",
            "allowed_commands": sorted(merged_allowed),
        }
        logger.info("应用合并后的项目级命令白名单 | allowed_commands={}", selected["allowed_commands"])
        return selected

    def _get_effective_command_policies(self) -> Dict[str, Dict[str, Any]]:
        """返回当前有效命令白名单（已应用项目级过滤）。"""
        if not isinstance(self.command_policies, dict):
            return {}
        if not self.active_allowed_command_names:
            return self.command_policies
        return {
            name: cfg
            for name, cfg in self.command_policies.items()
            if name in self.active_allowed_command_names
        }

    def _select_project_edit_policy(self, task_path: str, code_path: str, target_path: str) -> Dict[str, Any]:
        """不再使用 match_paths，统一合并顶层与所有 projects 的可编辑文件列表。"""
        if not isinstance(self.file_edit_policies, dict):
            return {}

        merged_target: List[str] = []
        merged_code: List[str] = []

        top_target = self.file_edit_policies.get("editable_files_under_target", [])
        top_code = self.file_edit_policies.get("editable_files_under_code", [])
        if isinstance(top_target, list):
            merged_target.extend([str(x) for x in top_target if str(x).strip()])
        if isinstance(top_code, list):
            merged_code.extend([str(x) for x in top_code if str(x).strip()])

        projects = self.file_edit_policies.get("projects", {})
        if isinstance(projects, dict):
            for _, cfg in projects.items():
                if not isinstance(cfg, dict):
                    continue
                p_target = cfg.get("editable_files_under_target", [])
                p_code = cfg.get("editable_files_under_code", [])
                if isinstance(p_target, list):
                    merged_target.extend([str(x) for x in p_target if str(x).strip()])
                if isinstance(p_code, list):
                    merged_code.extend([str(x) for x in p_code if str(x).strip()])

        if not merged_target and not merged_code:
            logger.info("未配置可编辑文件白名单")
            return {}

        selected = {
            "project_name": "ALL_PROJECTS",
            "editable_files_under_target": sorted(set(merged_target)),
            "editable_files_under_code": sorted(set(merged_code)),
        }
        logger.info(
            "应用合并后的文件编辑白名单 | target_count={} | code_count={}",
            len(selected["editable_files_under_target"]),
            len(selected["editable_files_under_code"]),
        )
        return selected

    def _resolve_allowed_path(self, base_dir: str, configured_path: str) -> str:
        """白名单路径统一要求为绝对路径。"""
        _ = os.path.abspath(base_dir)
        return self._require_absolute_path(configured_path, "whitelist path")

    def _validate_editable_target(self, abs_target_path: str) -> Optional[str]:
        """校验目标文件是否在当前项目可编辑白名单内。返回 None 表示允许。"""
        policy = self.active_project_edit_policy
        if not isinstance(policy, dict) or not policy:
            return None

        allow_under_target = policy.get("editable_files_under_target", [])
        allow_under_code = policy.get("editable_files_under_code", [])
        if not isinstance(allow_under_target, list):
            allow_under_target = []
        if not isinstance(allow_under_code, list):
            allow_under_code = []

        allowed_abs_paths = set()
        for rel in allow_under_target:
            try:
                allowed_abs_paths.add(self._resolve_allowed_path(self.base_target_dir, str(rel)))
            except Exception:
                continue
        for rel in allow_under_code:
            try:
                allowed_abs_paths.add(self._resolve_allowed_path(self.base_code_dir, str(rel)))
            except Exception:
                continue

        norm_target = os.path.abspath(abs_target_path)
        if norm_target in allowed_abs_paths:
            return None

        display_targets = sorted([os.path.relpath(p, self.base_code_dir) if p.startswith(os.path.abspath(self.base_code_dir)) else p for p in allowed_abs_paths])
        return (
            f"target file is not in editable whitelist for project "
            f"'{policy.get('project_name', 'unknown')}'. allowed={display_targets}"
        )

    def _resolve_under_base_dir(self, target_path: str) -> str:
        """要求绝对路径，并确保目标文件位于 target_base_dir 内。"""
        resolved = self._require_absolute_path(target_path, "target_path")
        base_dir = os.path.abspath(self.base_target_dir)
        if not self._is_path_within(base_dir, resolved):
            raise ValueError(f"Path escapes target dir: {target_path}")
        return resolved

    def _resolve_cwd(self, cwd: Optional[str]) -> str:
        """解析命令执行目录。

        支持：
        - None / "" / "$TARGET_BASE_DIR": 目标项目目录
        - "$CODE_PATH": 基础代码目录
        - 其他输入：必须是绝对路径
        """
        if not cwd:
            return os.path.abspath(self.base_target_dir)
        cwd_text = str(cwd).strip()
        if cwd_text == "$TARGET_BASE_DIR":
            return os.path.abspath(self.base_target_dir)
        if cwd_text == "$CODE_PATH":
            return os.path.abspath(self.base_code_dir)
        resolved = self._require_absolute_path(cwd_text, "cwd")
        allowed_bases = [
            os.path.abspath(self.project_root),
            os.path.abspath(self.base_target_dir),
            os.path.abspath(self.base_code_dir),
        ]
        if not any(self._is_path_within(base, resolved) for base in allowed_bases):
            raise ValueError(
                "cwd must be under project_root / target_base_dir / code_path"
            )
        return resolved

    def _resolve_under_scope_dir(self, absolute_path: str, scope: str = "target") -> str:
        """按作用域校验绝对路径，并确保不会逃逸到作用域目录外。"""
        if scope == "target":
            base_dir = os.path.abspath(self.base_target_dir)
        elif scope == "code":
            base_dir = os.path.abspath(self.base_code_dir)
        else:
            raise ValueError("scope must be 'target' or 'code'.")

        resolved = self._require_absolute_path(absolute_path, "path")
        if not self._is_path_within(base_dir, resolved):
            raise ValueError(f"Path escapes {scope} dir: {absolute_path}")
        return resolved

    def _get_project_structure(
        self,
        path: str = "",
        scope: str = "target",
        max_depth: int = 4,
        max_entries: int = 200,
        include_hidden: bool = False,
        include_files: bool = False,
    ) -> Dict[str, Any]:
        """获取目录结构，要求传入绝对路径。"""
        start_path = self._resolve_under_scope_dir(path, scope=scope)
        if not os.path.exists(start_path):
            raise FileNotFoundError(f"Path not found: {path}")

        payload: Dict[str, Any] = {
            "scope": scope,
            "input_path": path,
            "resolved_path": start_path,
            "include_files": include_files,
            "entries": [],
            "truncated": False,
        }

        if os.path.isfile(start_path):
            payload["entries"].append({
                "type": "file",
                "path": os.path.basename(start_path),
            })
            return payload

        entry_count = 0
        for cur_root, dirs, files in os.walk(start_path):
            rel_root = os.path.relpath(cur_root, start_path)
            depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1

            if depth > max_depth:
                dirs[:] = []
                continue

            if not include_hidden:
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                files = [f for f in files if not f.startswith(".")]

            dirs.sort()
            files.sort()

            for d in dirs:
                rel_path = os.path.relpath(os.path.join(cur_root, d), start_path)
                payload["entries"].append({"type": "dir", "path": rel_path})
                entry_count += 1
                if entry_count >= max_entries:
                    payload["truncated"] = True
                    return payload

            if include_files:
                for f in files:
                    rel_path = os.path.relpath(os.path.join(cur_root, f), start_path)
                    payload["entries"].append({"type": "file", "path": rel_path})
                    entry_count += 1
                    if entry_count >= max_entries:
                        payload["truncated"] = True
                        return payload

        return payload

    def _get_file_text(
        self,
        path: str,
        scope: str = "target",
        start_line: int = 1,
        end_line: Optional[int] = None,
        max_chars: int = 12000,
    ) -> Dict[str, Any]:
        """读取单个文件文本，要求传入绝对路径。"""
        resolved_path = self._resolve_under_scope_dir(path, scope=scope)
        if not os.path.isfile(resolved_path):
            raise FileNotFoundError(f"File not found: {path}")

        with open(resolved_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        total_lines = len(lines)
        safe_start = max(1, int(start_line or 1))
        safe_end = total_lines if end_line is None else max(safe_start, int(end_line))
        slice_lines = lines[safe_start - 1:safe_end]
        content = "".join(slice_lines)
        truncated = False
        if len(content) > max_chars:
            content = content[:max_chars] + "\n...<truncated>"
            truncated = True

        return {
            "scope": scope,
            "input_path": path,
            "resolved_path": resolved_path,
            "start_line": safe_start,
            "end_line": safe_end,
            "total_lines": total_lines,
            "truncated": truncated,
            "content": content,
        }

    def _build_command_from_policy(
        self,
        command_name: str,
        args: Optional[List[str]],
        target_path: Optional[str],
    ) -> List[str]:
        """根据白名单策略组装可执行命令。"""
        effective_policies = self._get_effective_command_policies()
        allowed_names = sorted(effective_policies.keys())
        if self.active_allowed_command_names and command_name not in self.active_allowed_command_names:
            raise ValueError(
                f"Command not allowed for current task: {command_name}. "
                f"Allowed command_name values: {allowed_names}"
            )

        policy = effective_policies.get(command_name)
        if not isinstance(policy, dict):
            raise ValueError(
                f"Command not allowed: {command_name}. "
                f"Allowed command_name values: {allowed_names}"
            )

        base_cmd = policy.get("command", [])
        if not isinstance(base_cmd, list) or not all(isinstance(x, str) for x in base_cmd):
            raise ValueError(f"Invalid policy command config: {command_name}")

        allow_extra_args = bool(policy.get("allow_extra_args", True))
        append_target_path = bool(policy.get("append_target_path", False))
        args_mode = str(policy.get("args_mode", "passthrough") or "passthrough")

        cmd = list(base_cmd)
        if append_target_path:
            if not target_path:
                raise ValueError(f"Command requires target_path: {command_name}")
            cmd.append(target_path)

        if args is None:
            args = []
        if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
            raise ValueError("args must be a list of strings")
        if args and not allow_extra_args:
            raise ValueError(f"Extra args not allowed for command: {command_name}")

        if args_mode == "shell_command":
            # 兼容两种输入：[-c, "cmd..."] 或 ["cmd..."]，最终都归一为单个 shell 命令字符串。
            normalized_args = list(args)
            if normalized_args and normalized_args[0] == "-c":
                normalized_args = normalized_args[1:]
            if not normalized_args:
                raise ValueError(f"Command requires shell command args: {command_name}")
            shell_cmd = normalized_args[0] if len(normalized_args) == 1 else " ".join(normalized_args)
            cmd.append(shell_cmd)
        else:
            cmd.extend(args)
        return cmd

    def _format_command_policy_for_prompt(self) -> str:
        """将白名单命令格式化为 Prompt 可读文本。"""
        effective_policies = self._get_effective_command_policies()
        if not effective_policies:
            return "(当前无可用白名单命令，请先配置 command_whitelist.json)"

        lines = []
        for name in sorted(effective_policies.keys()):
            item = effective_policies.get(name, {})
            base_cmd = item.get("command", [])
            append_target_path = bool(item.get("append_target_path", False))
            allow_extra_args = bool(item.get("allow_extra_args", True))
            args_mode = str(item.get("args_mode", "passthrough") or "passthrough")
            lines.append(
                f"- {name}: base={base_cmd}, append_target_path={append_target_path}, allow_extra_args={allow_extra_args}, args_mode={args_mode}"
            )
        if isinstance(self.active_command_policy, dict) and self.active_command_policy:
            lines.append(f"(active_command_policy={self.active_command_policy.get('project_name', 'unknown')})")
        return "\n".join(lines)

    def _format_file_edit_policy_for_prompt(self) -> str:
        """将文件编辑白名单格式化为 Prompt 可读文本。"""
        policy = self.active_project_edit_policy
        if not isinstance(policy, dict) or not policy:
            return "(当前未命中文件编辑白名单策略，写入将被拒绝)"

        allow_under_target = policy.get("editable_files_under_target", [])
        allow_under_code = policy.get("editable_files_under_code", [])
        if not isinstance(allow_under_target, list):
            allow_under_target = []
        if not isinstance(allow_under_code, list):
            allow_under_code = []

        lines = [f"active_edit_policy={policy.get('project_name', 'unknown')}"]
        lines.append(f"- editable_files_under_target: {allow_under_target}")
        lines.append(f"- editable_files_under_code: {allow_under_code}")
        return "\n".join(lines)

    def _resolve_target_base_dir(self, target_path: str) -> str:
        """将 target_path 解析为目标项目目录（非具体文件）。"""
        abs_target = os.path.abspath(target_path)
        if os.path.exists(abs_target) and os.path.isfile(abs_target):
            raise ValueError(
                f"target_path must be a project directory, got file path: {target_path}"
            )
        return abs_target

    def prepare_code_context(self, code_path: str):
        """加载并处理代码库"""
        if not os.path.exists(code_path):
            print(f"[-] Code path not found: {code_path}")
            return

        def _load_one_file(file_path: str) -> List[Dict[str, Any]]:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code_text = f.read()
            except Exception as e:
                logger.warning("读取代码文件失败，已跳过 | path={} | error={}", file_path, e)
                return []

            if self.work_mode in ["code_chunk_simple", "repohcl_doc_augmentation"]:
                return CodeChunker.chunk_simple(code_text, file_path=file_path)
            if self.work_mode in ["code_chunk_complex"]:
                return CodeChunker.chunk_complex(code_text, file_path=file_path)
            return [{"type": "whole_code", "content": code_text, "file_path": file_path, "order": 0}]

        chunk_count = 0
        file_count = 0
        allowed_exts = {
            ".py", ".java", ".kt", ".scala", ".js", ".ts", ".tsx", ".jsx",
            ".go", ".rs", ".cpp", ".cc", ".c", ".h", ".hpp", ".cs", ".php",
            ".rb", ".swift", ".m", ".mm", ".sql", ".xml", ".yaml", ".yml", ".json"
        }

        if os.path.isfile(code_path):
            chunks = _load_one_file(code_path)
            if chunks:
                self.retriever.add_documents(chunks)
                chunk_count += len(chunks)
                file_count += 1
        else:
            for cur_root, dirs, files in os.walk(code_path):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {"__pycache__", "node_modules", "build", "dist", ".git"}]
                for filename in files:
                    if filename.startswith("."):
                        continue
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in allowed_exts:
                        continue
                    file_path = os.path.join(cur_root, filename)
                    chunks = _load_one_file(file_path)
                    if not chunks:
                        continue
                    self.retriever.add_documents(chunks)
                    chunk_count += len(chunks)
                    file_count += 1

        logger.info("代码上下文加载完成 | code_path={} | file_count={} | chunk_count={}", code_path, file_count, chunk_count)

    def prepare_doc_context(self, doc_path: str) -> int:
        """
        加载 markdown 文档，将每个 ### 单元分 chunk，支持目录递归。
        每个 chunk 记录标题、顺序、文件路径等信息，便于后续检索和结构化索引。
        """
        if not os.path.exists(doc_path):
            print(f"[-] Doc path not found: {doc_path}")
            return 0

        doc_chunks = []  # 存储所有分割后的文档块

        def infer_doc_level(file_path: str) -> str:
            name = os.path.basename(file_path).lower()
            path_text = file_path.lower()
            if name.startswith("repo"):
                return "repo"
            if name.startswith("modules"):
                return "module"
            if name.endswith(".class.md") or name.endswith(".function.md"):
                return "symbol"
            if "/src/" in path_text:
                return "symbol"
            return "chunk"

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
            chunks = DocChunker.split_markdown_chunks(doc_text, file_path)
            level = infer_doc_level(file_path)
            for chunk in chunks:
                chunk["doc_level"] = level
            doc_chunks.extend(chunks)

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
        logger.info("文档分割完成 | doc_path={} | chunk_count={}", doc_path, len(doc_chunks))
        # 将所有分割后的 chunk 加入向量库/检索器
        self.retriever.add_documents(doc_chunks)
        return len(doc_chunks)

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """执行本地工具 (Tool Execution)"""
        logger.info("收到工具调用 | tool_name={} | arguments={}", tool_name, arguments)
        if tool_name == "search_knowledge":
            query = arguments.get("query", "")
            level = str(arguments.get("level", "auto") or "auto").strip().lower()
            if level not in {"auto", "repo", "module", "symbol", "chunk"}:
                level = "auto"
            raw_top_k = arguments.get("top_k", 8)
            try:
                top_k = int(raw_top_k)
            except Exception:
                top_k = 8
            top_k = max(1, min(top_k, 20))
            print(f"      -> 正在检索关键词: '{query}'")
            normalized_query = " ".join(str(query).lower().split())
            query_key = f"{normalized_query}||level={level}"
            repeat_count = self.search_query_counts.get(query_key, 0) + 1
            self.search_query_counts[query_key] = repeat_count

            results = self.retriever.search(query, top_k=top_k, level=level)
            result_signature = json.dumps(results, ensure_ascii=False, sort_keys=True)
            same_as_last = (self.search_result_signature_cache.get(query_key) == result_signature)
            self.search_result_signature_cache[query_key] = result_signature

            payload: Dict[str, Any] = {
                "query": query,
                "normalized_query": normalized_query,
                "level": level,
                "top_k": top_k,
                "repeat_count": repeat_count,
                "result_count": len(results),
                "same_as_last": same_as_last,
                "results": results,
            }

            if repeat_count >= 3 and same_as_last:
                payload["warning"] = (
                    "重复检索到相同结果。请停止重复 search_knowledge，改为基于当前结果直接写入代码或调用其他工具。"
                )

            logger.info(
                "检索工具返回 | query={} | level={} | top_k={} | repeat_count={} | result_count={} | same_as_last={}",
                normalized_query,
                level,
                top_k,
                repeat_count,
                len(results),
                same_as_last,
            )
            return json.dumps(payload, ensure_ascii=False)
        elif tool_name == "write_code_file":
            target_path = arguments.get("target_path")
            code = arguments.get("code", "")

            if not target_path:
                return "Error: missing required argument 'target_path'."

            try:
                final_target_path = self._resolve_under_base_dir(str(target_path))
            except Exception as e:
                return f"Error: invalid target_path: {e}"

            if not isinstance(self.active_project_edit_policy, dict) or not self.active_project_edit_policy:
                return (
                    "Error: write_code_file denied: no matched file edit whitelist policy (fail-closed). "
                    f"target_base_dir={os.path.abspath(self.base_target_dir)}"
                )

            edit_err = self._validate_editable_target(final_target_path)
            if edit_err:
                return (
                    f"Error: write_code_file denied: {edit_err}. "
                    f"target_base_dir={os.path.abspath(self.base_target_dir)}"
                )

            payload = self.execution_operator.write_code_file(
                target_path=final_target_path,
                code=code,
            )
            if bool(payload.get("ok", False)):
                # 记录写入版本号，供执行门禁判断。
                self.successful_write_count += 1
                self.last_successful_write_path = final_target_path
            payload["target_base_dir"] = os.path.abspath(self.base_target_dir)
            payload["input_target_path"] = target_path
            return json.dumps(payload, ensure_ascii=False)
        elif tool_name == "delete_code_file":
            target_path = arguments.get("target_path")
            if not target_path:
                return "Error: missing required argument 'target_path'."

            try:
                final_target_path = self._resolve_under_base_dir(str(target_path))
            except Exception as e:
                return f"Error: invalid target_path: {e}"

            edit_err = self._validate_editable_target(final_target_path)
            if edit_err:
                return f"Error: delete_code_file denied: {edit_err}"

            payload = self.execution_operator.delete_code_file(target_path=final_target_path)
            payload["target_base_dir"] = os.path.abspath(self.base_target_dir)
            payload["input_target_path"] = target_path
            return json.dumps(payload, ensure_ascii=False)
        elif tool_name == "execute_predefined_command":
            if self.successful_write_count <= 0:
                return (
                    "Error: execute_predefined_command denied: must call write_code_file successfully "
                    "before first command execution."
                )

            command_name = arguments.get("command_name")
            args = arguments.get("args", [])
            target_path = arguments.get("target_path")
            cwd = arguments.get("cwd")
            timeout = int(arguments.get("timeout", 120))

            cmd_name_text = str(command_name or "").strip()
            if not cmd_name_text:
                return "Error: missing required argument 'command_name'."

            # 禁止连续执行相同 command_name；若中间有成功写入则放行。
            if (
                self.last_executed_command_name == cmd_name_text
                and self.successful_write_count <= self.last_execute_write_count
            ):
                return (
                    "Error: execute_predefined_command denied: cannot execute the same command_name "
                    "consecutively without a successful write_code_file in between."
                )

            resolved_target_path = None
            if target_path:
                try:
                    resolved_target_path = self._resolve_under_base_dir(str(target_path))
                except Exception as e:
                    return f"Error: invalid target_path: {e}"

            try:
                resolved_cwd = self._resolve_cwd(cwd)
            except Exception as e:
                return f"Error: invalid cwd: {e}"

            try:
                command = self._build_command_from_policy(
                    command_name=str(command_name),
                    args=args,
                    target_path=resolved_target_path,
                )
            except Exception as e:
                return f"Error: command policy validation failed: {e}"

            exec_result = self.execution_operator.run_command(
                command=command,
                cwd=resolved_cwd,
                timeout=timeout,
            )
            payload = {
                "command_name": command_name,
                "command": command,
                "cwd": resolved_cwd,
                "target_path": resolved_target_path,
                "returncode": exec_result.get("returncode", -1),
                "stdout": exec_result.get("stdout", ""),
                "stderr": exec_result.get("stderr", ""),
                "last_successful_write_path": self.last_successful_write_path,
            }
            self.last_executed_command_name = cmd_name_text
            self.last_execute_write_count = self.successful_write_count
            logger.info("工具执行结果 | payload={}", payload)
            return json.dumps(payload, ensure_ascii=False)
        elif tool_name == "get_project_structure":
            try:
                scope_value = str(arguments.get("scope", "target"))
                if scope_value == "code":
                    default_path = os.path.abspath(self.base_code_dir)
                else:
                    default_path = os.path.abspath(self.base_target_dir)
                payload = self._get_project_structure(
                    path=str(arguments.get("path", default_path)),
                    scope=scope_value,
                    max_depth=int(arguments.get("max_depth", 4)),
                    max_entries=int(arguments.get("max_entries", 200)),
                    include_hidden=bool(arguments.get("include_hidden", False)),
                    include_files=bool(arguments.get("include_files", False)),
                )
                logger.info("项目结构工具执行结果 | entry_count={}", len(payload.get("entries", [])))
                return json.dumps(payload, ensure_ascii=False)
            except Exception as e:
                logger.exception("项目结构工具执行失败")
                return f"Error: get_project_structure failed: {e}"
        elif tool_name == "get_file_text":
            try:
                payload = self._get_file_text(
                    path=str(arguments.get("path", "")),
                    scope=str(arguments.get("scope", "target")),
                    start_line=int(arguments.get("start_line", 1)),
                    end_line=arguments.get("end_line"),
                    max_chars=int(arguments.get("max_chars", 12000)),
                )
                logger.info(
                    "单文件读取工具执行结果 | path={} | lines={}-{}",
                    payload.get("resolved_path", ""),
                    payload.get("start_line", 1),
                    payload.get("end_line", 1),
                )
                return json.dumps(payload, ensure_ascii=False)
            except Exception as e:
                logger.exception("单文件读取工具执行失败")
                return f"Error: get_file_text failed: {e}"
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

    def _get_plan_from_llm(self, client: OpenAI, model: str, task_desc: str, codebase_overview: str) -> Dict[str, Any]:
        """
        第一阶段：调用 LLM 以获取解决任务的执行计划，并返回 token 使用情况。
        """
        print("\n[Phase 1] 开始代码库分析与执行计划生成...")
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
            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
            total_tokens = int(getattr(usage, "total_tokens", 0) or 0) if usage is not None else 0
            print(f"    -> LLM 已生成执行计划:\n{plan}")
            logger.info(
                "LLM生成执行计划 | prompt={} | completion={} | total={} | plan={}",
                prompt_tokens,
                completion_tokens,
                total_tokens,
                plan,
            )
            return {
                "plan": plan,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        except Exception as e:
            print(f"[!] LLM在生成计划阶段异常: {e}")
            logger.exception("LLM在生成计划阶段异常")
            return {
                "plan": "无法生成执行计划，将直接尝试执行任务。",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

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

        max_turns = 50
        token_prompt_total = 0
        token_completion_total = 0
        token_total = 0
        # 将计划阶段内聚到 invoke_llm 中，统一计入 token 统计
        if self.work_mode == "plan_and_execute":
            codebase_overview = self._build_codebase_overview(code_path)
            if codebase_overview:
                plan_result = self._get_plan_from_llm(client, model, task_desc, codebase_overview)
                task_desc = f"任务目标：\n{task_desc}\n\n我的执行计划：\n{plan_result.get('plan', '')}"
                token_prompt_total += int(plan_result.get("prompt_tokens", 0) or 0)
                token_completion_total += int(plan_result.get("completion_tokens", 0) or 0)
                token_total += int(plan_result.get("total_tokens", 0) or 0)
                logger.info(
                    "计划阶段token累计 | prompt_total={} | completion_total={} | total={}",
                    token_prompt_total,
                    token_completion_total,
                    token_total,
                )
            else:
                print("[-] 未能构建代码库概览，将按常规模式执行。")

        system_prompt = SYSTEM_PROMPTS.get(self.work_mode, "You are a helpful AI.")
        if self.work_mode == "plan_and_execute":
            system_prompt = SYSTEM_PROMPTS.get("plan_and_execute", "You are a helpful AI.")
        else:
            system_prompt = (
                SYSTEM_PROMPTS.get("default", "You are a helpful AI.")
                + SYSTEM_PROMPTS.get(self.work_mode, "You are a helpful AI.")
                + SYSTEM_PROMPTS.get("error_repair", "")
            )
        messages = [{"role": "system", "content": system_prompt}]

        user_content = f"任务目标:\n{task_desc}"
        command_policy_desc = self._format_command_policy_for_prompt()
        file_edit_policy_desc = self._format_file_edit_policy_for_prompt()
        if code_path:
            user_content += (
                f"\n\n[环境与路径说明]\n"
                f"1. 你的基础代码目录(code_path)为: {code_path}\n"
                f"2. 你的目标项目目录(target_base_dir)为: {self.base_target_dir}\n"
                f"3. write_code_file/delete_code_file 的 target_path 必须传绝对路径，且必须位于 target_base_dir 内。\n"
                f"4. get_project_structure/get_file_text 的 path 也必须是绝对路径，并与 scope 保持一致。\n"
                f"5. execute_predefined_command 的 cwd 规则：\n"
                f"   - 不传或传 $TARGET_BASE_DIR：在目标项目目录执行\n"
                f"   - 传 $CODE_PATH：在基础代码目录执行（适合依赖项目本体时）\n"
                f"   - 其他情况必须传绝对路径，并且位于 project_root/target_base_dir/code_path 之内\n"
                f"6. 当代码依赖项目模块时，优先在 $CODE_PATH 执行，或在代码首部添加 import sys; sys.path.insert(0, '{code_path}')。\n"
                f"\n[命令白名单]\n{command_policy_desc}\n"
                f"\n[文件编辑白名单]\n{file_edit_policy_desc}\n"
            )
        if initial_context:
            user_content += f"\n已提供的上下文信息(Whole Doc):\n{initial_context[:2000]}..."  # 演示截断

        user_content += (
            "\n\n执行约束:\n"
            "1. 你可以参考任务文本、search_knowledge 返回结果，以及 get_project_structure/get_file_text 返回的上下文。\n"
            "2. 若需要落地代码，必须先调用 write_code_file。\n"
            "3. 若需先删后写，可调用 delete_code_file 删除目标文件后再 write_code_file。\n"
            "4. 代码写入后，如需执行，必须调用 execute_predefined_command。\n"
            "5. execute_predefined_command 只能使用白名单中的 command_name，不能直接传可执行命令文本。\n"
            "6. command_name 只允许你补充参数 args；命令主体由系统白名单决定。\n"
            "6.1 文件写入必须严格遵守文件编辑白名单：只能覆写白名单中的目标文件，禁止新建/修改任何其他文件。\n"
            "7. 所有路径参数必须使用绝对路径；禁止相对路径与 ..。\n"
            "8. 如需了解项目结构，可调用 get_project_structure（支持 target/code 两种作用域）。\n"
            "9. 如需查看单文件内容，可调用 get_file_text（支持按行读取，避免一次性拉全量）。\n"
            "10. Java 任务优先采用“写入 -> 执行”最短路径：先 write_code_file，再 execute_predefined_command，并优先在已配置好的项目目录执行。\n"
            "11. 当工具返回 returncode 为 0，且业务逻辑已跑通，请直接输出最终代码文本，不要再调用工具。\n"
            "12. 若 execute_predefined_command 返回失败，必须先基于报错做一次 search_knowledge(level=chunk) 精准检索，再改代码重试。\n"
            "13. 忽略非致命 stderr 警告（例如 SSL warning），不要为不影响最终结果的警告反复重试。"
        )

        messages.append({"role": "user", "content": user_content})

        messages_0 = messages
        for turn in range(1, max_turns + 1):
            print(f"\n  [Loop Turn {turn}] 等待大模型响应...")
            llm_logger.info(
                "对话上下文(发送前) | turn={} | messages={}",
                turn,
                _format_messages_for_log(messages),
            )
            try:
                request_kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if self.llm_tools_enabled and self.llm_tools:
                    request_kwargs["tools"] = self.llm_tools

                response = client.chat.completions.create(
                    **request_kwargs,
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

                    if tool_name == 'write_code_file':
                        messages = messages_0 + messages[2:]  # 重置对话上下文到初始状态，保留系统提示和用户任务描述
                        print("    [Context Reset]: write_code_file 调用后已重置对话上下文，保留初始用户输入，清除之前的工具调用历史。")
                        logger.info("write_code_file调用触发上下文重置 | turn={}", turn)
                    
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
        self.search_query_counts.clear()
        self.search_result_signature_cache.clear()
        self.successful_write_count = 0
        self.last_successful_write_path = None
        self.last_executed_command_name = None
        self.last_execute_write_count = 0
        try:
            self.base_target_dir = self._resolve_target_base_dir(target_path)
        except Exception as e:
            print(f"[!] target_path 参数错误: {e}")
            logger.error("target_path 参数错误 | target_path={} | error={}", target_path, e)
            return False
        self.base_code_dir = os.path.abspath(code_path) if code_path else os.path.abspath(self.project_root)
        self.active_command_policy = self._select_command_policy(task_path, self.base_code_dir)
        allowed_names = self.active_command_policy.get("allowed_commands", []) if isinstance(self.active_command_policy, dict) else []
        if isinstance(allowed_names, list) and allowed_names:
            self.active_allowed_command_names = {
                str(x) for x in allowed_names
                if isinstance(x, str) and x in self.command_policies
            }
            logger.info(
                "应用项目级命令白名单 | project={} | allowed_commands={}",
                self.active_command_policy.get("project_name", "unknown"),
                sorted(self.active_allowed_command_names),
            )
        else:
            self.active_allowed_command_names = None
        self.active_project_edit_policy = self._select_project_edit_policy(
            task_path=task_path,
            code_path=self.base_code_dir,
            target_path=self.base_target_dir,
        )
        os.makedirs(self.base_target_dir, exist_ok=True)
        logger.info(
            "Agent启动 | work_mode={} | task_path={} | doc_path={} | code_path={} | target_path={} | base_target_dir={} | base_code_dir={}",
            self.work_mode,
            task_path,
            doc_path,
            code_path,
            target_path,
            self.base_target_dir,
            self.base_code_dir,
        )
        
        if not os.path.exists(task_path):
            print(f"[!] task_path not found: {task_path}")
            logger.error("task_path not found | task_path={}", task_path)
            return False
        with open(task_path, "r", encoding="utf-8") as f:
            task_desc = f.read()

        if not os.path.exists(doc_path):
            print(f"[!] doc_path not found: {doc_path}")
            logger.error("doc_path not found | doc_path={}", doc_path)
            return False

        self.prepare_code_context(code_path)
        doc_chunk_count = self.prepare_doc_context(doc_path)
        if doc_chunk_count <= 0:
            print(f"[!] 文档上下文为空，无法进行文档检索: {doc_path}")
            logger.error("文档上下文为空 | doc_path={}", doc_path)
            return False

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
        
        logger.info("任务完成 | target_base_dir={}", self.base_target_dir)
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
    parser.add_argument(
        "--target_path",
        required=True,
        help="Target project directory used as base path for write/delete operations",
    )
    parser.add_argument(
        "--project_root",
        default=".",
        help="The absolute path to the root of the agent project.",
    )

    args = parser.parse_args()

    # Resolve all paths relative to the project root to ensure consistency
    project_root_abs = os.path.abspath(args.project_root)
    task_path_abs = os.path.abspath(os.path.join(project_root_abs, args.task_path))
    doc_path_abs = os.path.abspath(os.path.join(project_root_abs, args.doc_path))
    code_path_abs = os.path.abspath(os.path.join(project_root_abs, args.code_path))
    target_path_abs = os.path.abspath(os.path.join(project_root_abs, args.target_path))

    agent = SoftwareAgent(work_mode=args.work_mode, project_root=project_root_abs)
    ok = agent.run(
        task_path=task_path_abs,
        doc_path=doc_path_abs,
        code_path=code_path_abs,
        target_path=target_path_abs
    )
    if not ok:
        exit(1)