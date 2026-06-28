import asyncio
import os
import shutil
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from history_event_handler import handle_detected_question
from promt_automation import generate_dynamic_prompt
from langchain_core.tools import tool

load_dotenv()

memory = MemorySaver()

BASE_PROJECT_PATH = Path("base_project")
RUNS_DIR = Path("runs")

APP_ID = "id1512793879"


class AuditRecorder:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.audit_log_path = run_dir / "audit.jsonl"
        self.events: List[Dict[str, Any]] = []

    def write(self, event_type: str, payload: Dict[str, Any]):
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "payload": payload,
        }

        self.events.append(event)

        with self.audit_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def agent_decisions(self) -> List[Dict[str, Any]]:
        return [
            e["payload"]
            for e in self.events
            if e["event_type"] == "AGENT_DECISION"
        ]

    def mcp_tool_results(self) -> List[Dict[str, Any]]:
        return [
            e["payload"]
            for e in self.events
            if e["event_type"] == "MCP_TOOL_RESULT"
        ]


def run_command(command: List[str], cwd: Path) -> Dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=120,
        )

        return {
            "command": command,
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }

    except Exception as e:
        return {
            "command": command,
            "exit_code": -1,
            "error": str(e),
        }


def prepare_run_dir() -> Dict[str, Path]:
    if not BASE_PROJECT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {BASE_PROJECT_PATH}. Create a base_project folder first."
        )

    RUNS_DIR.mkdir(exist_ok=True)

    existing_runs = [
        d.name
        for d in RUNS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]

    indices = []

    for run_name in existing_runs:
        parts = run_name.split("_")

        if len(parts) == 2 and parts[1].isdigit():
            indices.append(int(parts[1]))

    next_index = max(indices) + 1 if indices else 1

    run_id = f"run_{next_index}"
    run_dir = RUNS_DIR / run_id
    sandbox_dir = run_dir / "sandbox_env"

    run_dir.mkdir(parents=True)
    shutil.copytree(BASE_PROJECT_PATH, sandbox_dir)

    return {
        "run_dir": run_dir,
        "sandbox_dir": sandbox_dir,
    }


def find_ios_project_root(sandbox_dir: Path) -> Path:
    candidates: List[Path] = []

    for path in sandbox_dir.rglob("*"):
        if not path.is_dir():
            continue

        has_xcodeproj = any(path.glob("*.xcodeproj"))
        has_xcworkspace = any(path.glob("*.xcworkspace"))
        has_podfile = (path / "Podfile").exists()
        has_package = (path / "Package.swift").exists()

        if has_xcodeproj or has_xcworkspace or has_podfile or has_package:
            candidates.append(path)

    if candidates:
        candidates.sort(key=lambda p: len(str(p)))
        return candidates[0].resolve()

    fallback = sandbox_dir / "ncp10" / "swift" / "basic_app"

    if fallback.exists():
        return fallback.resolve()

    return sandbox_dir.resolve()


def extract_tool_name_from_tool_message(msg: Any) -> Optional[str]:
    return getattr(msg, "name", None)


def normalize_tool_content(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def detect_clarification_question(content: str) -> bool:
    if not content:
        return False

    lowered = content.lower()

    question_markers = [
        "?",
        "please provide",
        "can you provide",
        "which option",
        "do you want",
        "should i",
        "scene delegate",
        "att",
        "customeruserid",
        "response listener",
        "please provide your answers",
    ]

    return any(marker in lowered for marker in question_markers)


def get_chat_history_safely(agent: Any, config: Dict[str, Any]) -> List[Any]:
    try:
        current_state = agent.get_state(config)
        return current_state.values.get("messages", [])
    except Exception:
        return []


def serialize_chat_history(chat_history: List[Any]) -> List[Dict[str, str]]:
    serialized = []

    for msg in chat_history:
        serialized.append({
            "type": getattr(msg, "type", "unknown"),
            "content": str(getattr(msg, "content", "")),
        })

    return serialized


def build_report(
    run_dir: Path,
    recorder: AuditRecorder,
    scenario_name: str,
    prompt: str,
    project_root: Path,
    discovered_tools: List[str],
    git_status: Dict[str, Any],
) -> Dict[str, Any]:
    decisions = recorder.agent_decisions()
    results = recorder.mcp_tool_results()

    decided_tools = [
        d.get("tool_name")
        for d in decisions
        if d.get("tool_name")
    ]

    completed_tools = [
        r.get("tool_name")
        for r in results
        if r.get("tool_name")
    ]

    report = {
        "scenario": scenario_name,
        "status": "COMPLETED",
        "prompt": prompt,
        "project_root": str(project_root),
        "discovered_mcp_tools": discovered_tools,
        "llm_selected_tools": decided_tools,
        "completed_tools": completed_tools,
        "git_status": git_status,
        "artifacts": {
            "audit_log": str(run_dir / "audit.jsonl"),
            "report": str(run_dir / "report.json"),
        },
    }

    with (run_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report

def save_mcp_output_file(
    run_dir: Path,
    tool_name: Optional[str],
    content: str,
) -> str:
    safe_tool_name = tool_name or "unknown_tool"

    output_dir = run_dir / "mcp_outputs"
    output_dir.mkdir(exist_ok=True)

    existing_files = list(output_dir.glob(f"{safe_tool_name}_*.txt"))
    index = len(existing_files) + 1

    output_path = output_dir / f"{safe_tool_name}_{index}.txt"

    output_path.write_text(content, encoding="utf-8")

    return str(output_path)



def safe_project_path(project_root: Path, requested_path: str) -> Path:
    requested = Path(requested_path)

    if requested.is_absolute():
        resolved = requested.resolve()
    else:
        resolved = (project_root / requested).resolve()

    project_root_resolved = project_root.resolve()

    if not str(resolved).startswith(str(project_root_resolved)):
        raise ValueError(
            f"Blocked unsafe file path outside project root: {requested_path}"
        )

    return resolved



async def run_agent():
    paths = prepare_run_dir()

    run_dir = paths["run_dir"]
    sandbox_dir = paths["sandbox_dir"]
    ios_project_root = find_ios_project_root(sandbox_dir)

    scenario_name = "ios-sdk-integration"

    user_goal = "Install AppsFlyer's SDK in my app using their MCP"

    print("⏳ Generating dynamic prompt...")
    dynamic_user_prompt = generate_dynamic_prompt(user_goal)

    final_execution_prompt = (
        f"{dynamic_user_prompt}\n\n"
        f"Project path:\n"
        f"{ios_project_root}\n\n"
        f"You are connected directly to the AppsFlyer MCP tools, "
        f"and you also have generic file tools similar to an IDE.\n\n"
        f"Important rules:\n"
        f"1. Use AppsFlyer MCP tools for AppsFlyer-specific guidance.\n"
        f"2. AppsFlyer MCP integrateSdk may return instructions and code, but it does not mean the project files were edited.\n"
        f"3. After integrateSdk returns instructions, you must inspect the project files with list_project_files and read_project_file.\n"
        f"4. You must apply the needed changes yourself using write_to_project_file.\n"
        f"5. Do not say the SDK is integrated until you have actually written the updated project files.\n"
        f"6. Do not run verifyIosSdk before writing the relevant project file changes.\n"
        f"7. The file writing tool is generic. It does not know AppsFlyer logic. You must prepare the full updated file content yourself.\n"
        f"8. After writing the files, use verifyIosSdk when appropriate.\n"
        f"9. If you need more information, you may ask a question."
    )


    recorder = AuditRecorder(run_dir)

    recorder.write("RUN_STARTED", {
        "run_dir": str(run_dir),
        "sandbox_dir": str(sandbox_dir),
        "scenario": scenario_name,
        "ios_project_root": str(ios_project_root),
        "raw_user_goal": user_goal,
        "dynamic_user_prompt": dynamic_user_prompt,
        "final_execution_prompt": final_execution_prompt,
    })

    print(f"\n🚀 Run dir: {run_dir}")
    print(f"📦 Sandbox: {sandbox_dir}")
    print(f"📍 iOS project root: {ios_project_root}")
    print(f"🧪 Scenario: {scenario_name}")
    print(f"💬 Prompt:\n{final_execution_prompt}")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    dev_key = os.getenv("APPSFLYER_DEV_KEY")
    if not dev_key:
        raise RuntimeError("Missing APPSFLYER_DEV_KEY in .env")

    model = ChatOpenAI(
        model="gpt-4o",
        api_key=openai_api_key,
        temperature=0,
    )

    mcp_client = MultiServerMCPClient({
        "appsflyer-sdk-mcp": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@appsflyer/sdk-mcp-server"],
            "env": {
                "APP_ID": APP_ID,
                "DEV_KEY": dev_key,
            },
        }
    })

    mcp_tools = await mcp_client.get_tools()

    discovered_tools = [
        getattr(t, "name", str(t))
        for t in mcp_tools
    ]

    recorder.write("TOOLS_DISCOVERED", {
        "tools": discovered_tools,
    })

    for mcp_tool in mcp_tools:
        recorder.write("TOOL_METADATA", {
            "name": getattr(mcp_tool, "name", ""),
            "description": getattr(mcp_tool, "description", ""),
        })

    print(f"🧰 AppsFlyer MCP tools exposed directly to agent: {discovered_tools}")

    @tool
    def list_project_files() -> str:
        """
        List relevant editable files in the iOS project.
        Use this before deciding which file to read or edit.
        """

        allowed_suffixes = {
            ".swift",
            ".plist",
            ".podspec",
            ".pbxproj",
            ".xcodeproj",
            ".xcworkspace",
        }

        allowed_names = {
            "Podfile",
            "Package.swift",
        }

        files = []

        for path in ios_project_root.rglob("*"):
            if path.is_file():
                if path.name in allowed_names or path.suffix in allowed_suffixes:
                    files.append(str(path.relative_to(ios_project_root)))

        return json.dumps({
            "project_root": str(ios_project_root),
            "files": files,
        }, ensure_ascii=False, indent=2)

    @tool
    def read_project_file(file_path: str) -> str:
        """
        Read a project file.
        file_path must be relative to the project root, for example:
        basic_app/AppDelegate.swift
        Podfile
        basic_app/Info.plist
        """

        try:
            path = safe_project_path(ios_project_root, file_path)

            if not path.exists():
                return json.dumps({
                    "status": "FAILED",
                    "reason": "File does not exist",
                    "file_path": file_path,
                    "resolved_path": str(path),
                }, ensure_ascii=False, indent=2)

            if not path.is_file():
                return json.dumps({
                    "status": "FAILED",
                    "reason": "Path is not a file",
                    "file_path": file_path,
                    "resolved_path": str(path),
                }, ensure_ascii=False, indent=2)

            content = path.read_text(encoding="utf-8")

            return json.dumps({
                "status": "OK",
                "file_path": file_path,
                "resolved_path": str(path),
                "content": content,
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "FAILED",
                "error_type": type(e).__name__,
                "error": str(e),
                "file_path": file_path,
            }, ensure_ascii=False, indent=2)

    @tool
    def write_to_project_file(file_path: str, content: str) -> str:
        """
        Write exact content to a project file.

        This tool does not understand AppsFlyer, Swift, Podfile, or iOS.
        It only writes the exact content provided by the LLM.
        The LLM is responsible for reading the MCP output, choosing the file,
        and preparing the correct updated file content.
        """

        try:
            path = safe_project_path(ios_project_root, file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            old_content = ""
            existed_before = path.exists()

            if existed_before and path.is_file():
                old_content = path.read_text(encoding="utf-8")

            path.write_text(content, encoding="utf-8")

            changed = old_content != content

            recorder.write("PROJECT_FILE_WRITTEN", {
                "file_path": file_path,
                "resolved_path": str(path),
                "existed_before": existed_before,
                "changed": changed,
                "old_size": len(old_content),
                "new_size": len(content),
            })

            return json.dumps({
                "status": "WRITTEN" if changed else "NO_CHANGES",
                "file_path": file_path,
                "resolved_path": str(path),
                "existed_before": existed_before,
                "changed": changed,
                "old_size": len(old_content),
                "new_size": len(content),
            }, ensure_ascii=False, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "FAILED",
                "error_type": type(e).__name__,
                "error": str(e),
                "file_path": file_path,
            }, ensure_ascii=False, indent=2)

    file_tools = [
        list_project_files,
        read_project_file,
        write_to_project_file,
    ]
    agent_tools = [
        *mcp_tools,
        *file_tools,
    ]
    recorder.write("AGENT_TOOLS_EXPOSED", {
        "tools": [
            getattr(t, "name", str(t))
            for t in agent_tools
        ]
    })

    agent = create_agent(
        model=model,
        tools=agent_tools,
        checkpointer=memory,
    )
    config = {
        "configurable": {
            "thread_id": f"appsflyer_mcp_cursor_like_{run_dir.name}"
        }
    }

    next_user_message = final_execution_prompt
    max_turns =8
    turn_index = 0

    try:
        while next_user_message and turn_index < max_turns:
            turn_index += 1

            recorder.write("USER_MESSAGE_SENT_TO_AGENT", {
                "turn_index": turn_index,
                "content": next_user_message,
            })

            current_user_message = next_user_message
            next_user_message = None

            async for chunk in agent.astream(
                    {"messages": [("user", current_user_message)]},
                    config=config,
                    stream_mode="updates",
            ):
                recorder.write("RAW_CHUNK", {
                    "turn_index": turn_index,
                    "chunk": str(chunk),
                })

                agent_chunk = chunk.get("agent") or chunk.get("model")

                if agent_chunk:
                    for msg in agent_chunk.get("messages", []):
                        tool_calls = getattr(msg, "tool_calls", None)

                        if tool_calls:
                            for call in tool_calls:
                                payload = {
                                    "tool_name": call.get("name"),
                                    "args": call.get("args", {}),
                                    "id": call.get("id"),
                                }

                                print(
                                    f"🔥 [AGENT_DECISION] "
                                    f"{payload['tool_name']} args={payload['args']}"
                                )

                                recorder.write("AGENT_DECISION", payload)

                        content = getattr(msg, "content", None)

                        if content:
                            recorder.write("AGENT_MESSAGE", {
                                "turn_index": turn_index,
                                "content": content,
                            })

                            if detect_clarification_question(content):
                                recorder.write("QUESTION_DETECTED", {
                                    "turn_index": turn_index,
                                    "content": content,
                                })

                                chat_history = get_chat_history_safely(agent, config)

                                auto_user_reply = handle_detected_question(chat_history)

                                recorder.write("QUESTION_HISTORY_SENT_TO_HANDLER", {
                                    "turn_index": turn_index,
                                    "message_count": len(chat_history),
                                    "messages": serialize_chat_history(chat_history),
                                    "auto_user_reply": auto_user_reply,
                                })

                                if auto_user_reply:
                                    print(f"👤 [AUTO_USER_REPLY] {auto_user_reply}")
                                    next_user_message = auto_user_reply

                if "tools" in chunk:
                    for msg in chunk["tools"].get("messages", []):
                        tool_name = extract_tool_name_from_tool_message(msg)

                        content = normalize_tool_content(
                            getattr(msg, "content", "")
                        )

                        output_file = save_mcp_output_file(
                            run_dir=run_dir,
                            tool_name=tool_name,
                            content=content,
                        )

                        payload = {
                            "tool_name": tool_name,
                            "tool_call_id": getattr(msg, "tool_call_id", None),
                            "content": content,
                            "saved_output_file": output_file,
                        }

                        print(
                            f"🟢 [MCP_TOOL_RESULT] "
                            f"{tool_name or 'unknown_tool'} → "
                            f"{content[:200]}..."
                        )

                        print(f"💾 [MCP_OUTPUT_SAVED] {output_file}")

                        recorder.write("MCP_TOOL_RESULT", payload)

    except Exception as e:
        error_info = {
            "error_type": type(e).__name__,
            "error": str(e),
        }

        recorder.write("RUN_ERROR", error_info)

        chat_history = get_chat_history_safely(agent, config)

        recorder.write("ERROR_HISTORY_SENT_TO_HANDLER", {
            "error": error_info,
            "message_count": len(chat_history),
            "messages": serialize_chat_history(chat_history),
        })



        print(f"\n❌ Run failed: {type(e).__name__}: {e}")
    chat_history = get_chat_history_safely(agent, config)

    recorder.write("FINAL_CHAT_HISTORY", {
        "messages": serialize_chat_history(chat_history),
    })

    git_status = run_command(
        ["git", "status", "--porcelain"],
        cwd=sandbox_dir,
    )

    recorder.write("GIT_STATUS", git_status)

    report = build_report(
        run_dir=run_dir,
        recorder=recorder,
        scenario_name=scenario_name,
        prompt=final_execution_prompt,
        project_root=ios_project_root,
        discovered_tools=discovered_tools,
        git_status=git_status,
    )

    print("\n==============================")
    print("📊 FINAL REPORT")
    print("==============================")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n==============================")
    print("📜 CHAT HISTORY CAPTURED")
    print("==============================")

    for msg in chat_history:
        print(
            f"[{getattr(msg, 'type', 'UNKNOWN').upper()}]: "
            f"{getattr(msg, 'content', '')}"
        )


if __name__ == "__main__":
    asyncio.run(run_agent())