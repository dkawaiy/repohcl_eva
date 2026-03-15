import os
import ast
import argparse
import json
from typing import List, Dict, Any
from prompts import SYSTEM_PROMPTS

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
    def chunk_complex(code_text: str) -> List[Dict[str, Any]]:
        """基于AST树进行高级语义拆分（函数、类粒度）"""
        chunks = []
        try:
            tree = ast.parse(code_text)
            lines = code_text.splitlines()
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start_lineno = node.lineno - 1
                    end_lineno = getattr(node, 'end_lineno', node.lineno)
                    content = "\n".join(lines[start_lineno:end_lineno])
                    chunks.append({
                        "type": "code_chunk_complex",
                        "name": node.name,
                        "content": content
                    })
        except SyntaxError:
            pass
        return chunks

# ==============================
# 模拟检索模块 (Embeddings / VectorStore)
# ==============================
class Retriever:
    def __init__(self):
        self.knowledge_base = []

    def add_documents(self, docs: List[Dict[str, Any]]):
        self.knowledge_base.extend(docs)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """模拟向量检索(后期可替换为 Chroma/FAISS 等)"""
        query_words = query.lower().split()
        scored_docs = []
        for doc in self.knowledge_base:
            content = doc.get("content", "").lower()
            score = sum(1 for w in query_words if w in content)
            if score > 0:
                scored_docs.append((score, doc))
        
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]

# ==============================
# 代理核心：整合流工作流
# ==============================
class SoftwareAgent:
    def __init__(self, work_mode: str):
        self.work_mode = work_mode
        self.retriever = Retriever()

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
        """加载参考文档"""
        if not os.path.exists(doc_path):
            print(f"[-] Doc path not found: {doc_path}")
            return
        
        if os.path.isfile(doc_path):
            with open(doc_path, "r", encoding="utf-8") as f:
                doc_text = f.read()

            if self.work_mode == "repohcl_doc_augmentation":
                pass # 执行文档跟代码结构增强逻辑

            self.retriever.add_documents([{"type": "doc", "content": doc_text}])

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
        else:
            return f"Error: Tool {tool_name} not found."

    def invoke_llm(self, task_desc: str, initial_context: str = "") -> str:
        """多轮对话代理的核心循环 (Agent Loop with Tool Calling)"""
        print(f"\n[*] 初始化大模型多轮推理代理引擎...")
        
        system_prompt = SYSTEM_PROMPTS.get(self.work_mode, "You are a helpful AI.")
        messages = [{"role": "system", "content": system_prompt}]
        
        user_content = f"任务目标:\n{task_desc}"
        if initial_context:
            user_content += f"\n\n已提供的上下文信息(Whole Doc):\n{initial_context[:2000]}..." # 演示截断
            
        messages.append({"role": "user", "content": user_content})
        
        max_turns = 4
        
        for turn in range(1, max_turns + 1):
            print(f"\n  [Loop Turn {turn}] 正在等待大模型响应...")
            
            # --- 以下为模拟实际的模型调用 (例如 openai.chat.completions.create) ---
            # 伪造逻辑:
            # 如果不是 whole_doc，并且是第一/二轮，就假装模型想去搜一搜
            if self.work_mode != "whole_doc" and turn < 3:
                simulated_query = " ".join(task_desc.split()[:2]) or "query"
                tool_call = {
                    "name": "search_knowledge",
                    "arguments": {"query": simulated_query}
                }
                print(f"    [LLM 动作]: 决定调用工具 -> {tool_call['name']}({tool_call['arguments']})")
                
                # 压入记录
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": f"call_{turn}", "function": tool_call}]
                })
                
                # 本地执行工具 (Tool Exec)
                tool_result_str = self.execute_tool(tool_call["name"], tool_call["arguments"])
                print(f"    [Tool 结果]: (已成功获取)")
                
                # 结果回传大模型
                messages.append({
                    "role": "tool",
                    "tool_call_id": f"call_{turn}",
                    "name": tool_call["name"],
                    "content": tool_result_str
                })
                # 重新跳回循环头，让 LLM 再次决策
                continue

            else:
                # 模拟模型输出最终答案阶段
                print(f"    [LLM 动作]: 已收集充分信息，开始生成终态代码...")
                simulated_code = f"# ==== Target Output generated by Agent ====\n"
                simulated_code += f"# Task: {task_desc.strip()[:30]}...\n\n"
                simulated_code += f"def generated_solution_{self.work_mode}():\n    pass\n"
                messages.append({"role": "assistant", "content": simulated_code})
                return simulated_code

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