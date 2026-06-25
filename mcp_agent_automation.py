import asyncio
import os
import shutil
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

load_dotenv()

BASE_PROJECT_PATH = Path("base_project")
RUNS_DIR = Path("runs")

APP_ID = "id1512793879"
APPLE_APP_ID_FOR_SWIFT = "1512793879"


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

    def tool_decisions(self) -> List[Dict[str, Any]]:
        return [
            e["payload"]
            for e in self.events
            if e["event_type"] == "AGENT_DECISION"
        ]

    def tool_results(self) -> List[Dict[str, Any]]:
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
        d.name for d in RUNS_DIR.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]

    indices = []
    for run_name in existing_runs:
        try:
            parts = run_name.split("_")
            if len(parts) == 2 and parts[1].isdigit():
                indices.append(int(parts[1]))
        except ValueError:
            continue

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


def extract_tool_name_from_tool_message(msg: Any) -> Optional[str]:
    return getattr(msg, "name", None)


def message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or item))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content) if content is not None else ""


def build_report(
        run_dir: Path,
        recorder: AuditRecorder,
        scenario: Dict[str, Any],
        git_status: Dict[str, Any],
) -> Dict[str, Any]:
    decisions = recorder.tool_decisions()
    results = recorder.tool_results()

    decided_tools = [d.get("tool_name") for d in decisions if d.get("tool_name")]
    result_tools = [r.get("tool_name") for r in results if r.get("tool_name")]

    actually_completed_tools = result_tools or decided_tools

    failures: List[str] = []

    for tool in scenario["required_tools"]:
        if tool not in actually_completed_tools and tool not in decided_tools:
            failures.append(f"Missing required tool call: {tool}")

    for tool in scenario.get("forbidden_tools", []):
        if tool in actually_completed_tools or tool in decided_tools:
            failures.append(f"Forbidden tool was called: {tool}")

    verify_tool = scenario["verify_tool"]
    verify_results = [
        r for r in results
        if r.get("tool_name") == verify_tool or verify_tool in json.dumps(r, ensure_ascii=False)
    ]

    if not verify_results:
        failures.append(f"Verify tool did not run: {verify_tool}")
        verify_text = ""
    else:
        verify_text = json.dumps(verify_results[-1], ensure_ascii=False)

    for field in scenario.get("expected_verify_fields", []):
        if field not in verify_text:
            failures.append(f"Verify result missing expected field: {field}")

    expected_app_id = scenario.get("expected_app_id")
    if expected_app_id and expected_app_id not in verify_text:
        failures.append(f"Verify result missing expected App ID: {expected_app_id}")

    report = {
        "scenario": scenario["name"],
        "status": "PASS" if not failures else "FAIL",
        "prompt": scenario["prompt"],
        "required_tools": scenario["required_tools"],
        "forbidden_tools": scenario.get("forbidden_tools", []),
        "decided_tools": decided_tools,
        "completed_tools": result_tools,
        "verify_tool": verify_tool,
        "failures": failures,
        "git_status": git_status,
        "artifacts": {
            "audit_log": str(run_dir / "audit.jsonl"),
            "report": str(run_dir / "report.json"),
        },
    }

    with (run_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def detect_clarification_question(content: str) -> bool:
    if not content:
        return False

    keywords = [
        "scene delegate",
        "att",
        "customeruserid",
        "response listener",
        "please provide your answers",
        "would you like",
        "do you want",
        "should i",
        "can you confirm",
        "which",
    ]

    content = content.lower()

    return "?" in content or any(keyword in content for keyword in keywords)


def build_tool_aligned_customer_policy(context: Dict[str, Any]) -> str:
    tool_metadata = context.get("mcp_tool_metadata", [])
    tool_names = [tool.get("name", "") for tool in tool_metadata]
    searchable_tools = " ".join(tool_names).lower()

    supported_workflows = []
    if "integratesdk" in searchable_tools or "integrate_sdk" in searchable_tools:
        supported_workflows.append("SDK installation and configuration")
    if "verify" in searchable_tools:
        supported_workflows.append("verification/status checks")
    if "deeplink" in searchable_tools or "deep_link" in searchable_tools:
        supported_workflows.append("deep linking and deferred deep linking")
    if "inappevent" in searchable_tools or "in_app_event" in searchable_tools:
        supported_workflows.append("in-app/app event tracking")
    if "cuid" in searchable_tools or "customer" in searchable_tools:
        supported_workflows.append("Customer User ID / CUID")

    if not supported_workflows:
        supported_workflows.append("AppsFlyer workflows exposed by the MCP tool catalog")

    tool_catalog = json.dumps(tool_metadata, ensure_ascii=False, indent=2)

    return f"""
AppsFlyer MCP tool catalog available to the coding agent:
{tool_catalog}

Customer behavior must be aligned to these MCP capabilities:
- Agree to AppsFlyer tasks only when they match the discovered MCP tools or their descriptions.
- Supported workflow categories detected: {", ".join(supported_workflows)}.
- If asked about deep linking and deep link tools exist, say yes and ask for a safe test configuration.
- If asked about in-app/app events and event tools exist, say yes and use safe test events.
- If asked about SDK install, verify, CUID, ATT, response listener, delegate callbacks, app IDs, dev key,
  platform, or project path, answer from the known app context and defaults.
- If asked for something not represented by AppsFlyer MCP tools, do not invent behavior. Say to keep the
  task focused on the AppsFlyer MCP-supported workflow and use the closest safe default.
""".strip()


def build_required_question_answer(content: str, context: Dict[str, Any]) -> Optional[str]:
    normalized = content.lower()
    answers = [
        (
            ["platform", "ios", "android", "swift"],
            "- Platform: iOS Swift",
        ),
        (
            ["project path", "projectpath", "path", "directory", "folder"],
            f"- Project path: {context['ios_project_root']}",
        ),
        (
            ["app id", "appid", "apps flyer app id", "appsflyer app id"],
            f"- AppsFlyer App ID: {context['app_id']}",
        ),
        (
            ["apple app id", "itunes", "app store id"],
            f"- Apple App ID: {context['apple_app_id']}",
        ),
        (
            ["dev key", "devkey", "developer key"],
            "- AppsFlyer dev key: use the DEV_KEY value already provided to the MCP environment",
        ),
        (
            ["cuid", "customer user id", "customeruserid"],
            "- Customer User ID / CUID: yes, use test_customer_user_id",
        ),
        (
            ["response listener", "isresponselistener"],
            "- Response listener: yes",
        ),
        (
            ["delegate", "isdelegate"],
            "- Delegate callbacks: yes",
        ),
        (
            ["att", "app tracking transparency", "tracking transparency", "isatt"],
            "- ATT: no unless this exact MCP workflow requires ATT and the app already has the required usage text",
        ),
    ]

    matched_answers = [
        answer
        for keywords, answer in answers
        if any(keyword in normalized for keyword in keywords)
    ]

    if not matched_answers:
        return None

    return "\n".join([
        "Here are the required AppsFlyer setup answers:",
        *matched_answers,
        "Continue with the relevant AppsFlyer MCP tool flow.",
    ])


def build_fallback_simulated_user_answer(content: str, context: Dict[str, Any]) -> str:
    normalized = content.lower()
    answer_lines = [
        "Use these answers and continue without waiting for more input:",
        "- Platform: iOS Swift",
        f"- Project path: {context['ios_project_root']}",
        f"- AppsFlyer App ID: {context['app_id']}",
        f"- Apple App ID: {context['apple_app_id']}",
        "- Enable response listener: yes",
        "- Enable delegate callbacks: yes",
        "- Enable ATT prompt: no, unless the app already has the required tracking usage text and the install task explicitly asks for ATT",
        "- Set Customer User ID / CUID: yes, use test_customer_user_id for this automation run",
        "- Use the AppsFlyer dev key from the MCP environment",
        "- For AppsFlyer deep linking: yes, use safe test link values if the MCP tool asks for them",
        "- For AppsFlyer in-app/app events: yes, use safe test events if the MCP tool asks for them",
        "Continue with the AppsFlyer MCP-supported tool flow.",
    ]

    if "customer" in normalized or "cuid" in normalized:
        answer_lines.append("Use test_customer_user_id as the Customer User ID.")

    if "deep link" in normalized or "deeplink" in normalized:
        answer_lines.append("Yes, add AppsFlyer deep linking with safe test values.")

    if "event" in normalized:
        answer_lines.append("Yes, add AppsFlyer app event tracking with a safe test event.")

    if "att" in normalized or "tracking" in normalized:
        answer_lines.append("Only add App Tracking Transparency if the project already has the proper usage description and this task explicitly asks for ATT.")

    if "response listener" in normalized:
        answer_lines.append("Enable the response listener.")

    return "\n".join(answer_lines)


async def build_simulated_user_answer(
        content: str,
        context: Dict[str, Any],
        responder_model: Any,
) -> str:
    required_answer = build_required_question_answer(content, context)
    if required_answer:
        return required_answer

    tool_policy = build_tool_aligned_customer_policy(context)
    system_prompt = f"""
You are simulating a normal AppsFlyer customer using the AppsFlyer MCP through a coding agent.

Answer the coding agent's clarification question naturally and briefly.
Stay strictly inside AppsFlyer MCP-supported workflows. Do not request unrelated features,
do not discuss general app product behavior, and do not invent requirements outside the MCP tool catalog.

Known app context:
- Platform: iOS Swift
- Project path: {context['ios_project_root']}
- AppsFlyer App ID: {context['app_id']}
- Apple App ID: {context['apple_app_id']}
- AppsFlyer dev key is already available in the MCP environment.

Default choices for AppsFlyer SDK setup:
- Response listener: yes
- Delegate callbacks: yes
- Customer User ID / CUID: yes, use test_customer_user_id for this automation run
- ATT: no unless the agent says it is required for this exact SDK install and the app already has the usage text
- Deep linking: yes if the discovered AppsFlyer MCP tools support it; use safe test values
- In-app/app events: yes if the discovered AppsFlyer MCP tools support it; use safe test events

{tool_policy}

If the question is not about an AppsFlyer MCP-supported workflow, answer:
"Let's keep this focused on the AppsFlyer MCP-supported workflow. Use the closest safe default and continue."

End every answer by telling the agent to continue with the relevant AppsFlyer MCP tool flow.
""".strip()

    try:
        response = await responder_model.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"The coding agent asked:\n{content}\n\nReply as the simulated customer."),
        ])
        answer = message_content_to_text(getattr(response, "content", None)).strip()
        if answer:
            return answer
    except Exception:
        pass

    return build_fallback_simulated_user_answer(content, context)


async def stream_agent_turn(
        agent: Any,
        messages: List[Any],
        recorder: AuditRecorder,
) -> Optional[str]:
    latest_assistant_content: Optional[str] = None

    async for chunk in agent.astream(
            {"messages": messages},
            stream_mode="updates",
    ):
        recorder.write("RAW_CHUNK", {"chunk": str(chunk)})

        if "agent" in chunk:
            for msg in chunk["agent"].get("messages", []):
                tool_calls = getattr(msg, "tool_calls", None)

                if tool_calls:
                    for call in tool_calls:
                        payload = {
                            "tool_name": call.get("name"),
                            "args": call.get("args", {}),
                            "id": call.get("id"),
                        }
                        print(f"🔥 [AGENT_DECISION] {payload['tool_name']} args={payload['args']}")
                        recorder.write("AGENT_DECISION", payload)

                content = message_content_to_text(getattr(msg, "content", None))
                if content:
                    latest_assistant_content = content
                    recorder.write("AGENT_MESSAGE", {"content": content})

        if "tools" in chunk:
            for msg in chunk["tools"].get("messages", []):
                tool_name = extract_tool_name_from_tool_message(msg)

                payload = {
                    "tool_name": tool_name,
                    "tool_call_id": getattr(msg, "tool_call_id", None),
                    "content": getattr(msg, "content", ""),
                }

                print(f"🟢 [MCP_TOOL_RESULT] {tool_name or 'unknown_tool'} → {str(payload['content'])[:200]}...")
                recorder.write("MCP_TOOL_RESULT", payload)

    return latest_assistant_content


async def run_agent_with_simulated_user(
        agent: Any,
        initial_prompt: str,
        recorder: AuditRecorder,
        simulation_context: Dict[str, Any],
        responder_model: Any,
        max_turns: int = 4,
) -> None:
    messages: List[Any] = [HumanMessage(content=initial_prompt)]
    answered_questions = set()

    for turn_index in range(1, max_turns + 1):
        recorder.write("AGENT_TURN_STARTED", {
            "turn": turn_index,
            "message_count": len(messages),
        })

        latest_content = await stream_agent_turn(agent, messages, recorder)

        if not latest_content or not detect_clarification_question(latest_content):
            return

        question_key = latest_content.strip()
        if question_key in answered_questions:
            recorder.write("CLARIFICATION_LOOP_STOPPED", {
                "turn": turn_index,
                "content": latest_content,
            })
            return

        answered_questions.add(question_key)

        simulated_answer = await build_simulated_user_answer(
            latest_content,
            simulation_context,
            responder_model,
        )
        recorder.write("CLARIFICATION_REQUESTED", {"content": latest_content})
        recorder.write("SIMULATED_USER_REPLY", {"content": simulated_answer})
        print(f"🙋 [SIMULATED_USER_REPLY]\n{simulated_answer}\n")

        messages.append(AIMessage(content=latest_content))
        messages.append(HumanMessage(content=simulated_answer))

    recorder.write("SIMULATED_USER_MAX_TURNS_REACHED", {
        "max_turns": max_turns,
    })


async def run_agent():
    paths = prepare_run_dir()
    run_dir = paths["run_dir"]
    sandbox_dir = paths["sandbox_dir"]

    ios_project_root = (sandbox_dir / "ncp10" / "swift" / "basic_app").resolve()
    ios_source_dir = ios_project_root / "basic_app"
    app_delegate_path = ios_source_dir / "AppDelegate.swift"

    scenario_config = {
        "name": "ios-sdk-integration",
        "prompt": (
            "You are an expert automation engineer. Execute exactly these steps:\n"
            "1. Call integrate_sdk_fixed.\n"
            "2. Call patch_app_delegate_fixed.\n"
            "3. Call verify_sdk_fixed.\n"
            "Do not ask questions. Do not skip steps."
        ),
        "required_tools": [
            "integrate_sdk_fixed",
            "patch_app_delegate_fixed",
            "verify_sdk_fixed",
        ],
        "verify_tool": "verify_sdk_fixed",
        "forbidden_tools": [
            "verifySdk",
            "createDeepLink",
            "createIosDeepLink",
            "createInAppEvent",
            "createIosInAppEvent",
            "verifyDeepLink",
            "verifyIosDeepLink",
            "verifyInAppEvent",
            "verifyIosInAppEvent",
            "integrateSdk",
            "verifyIosSdk",
        ],
        "expected_verify_fields": ["App ID", "UID", "Status", "Install time"],
        "expected_app_id": APP_ID,
    }

    recorder = AuditRecorder(run_dir)

    recorder.write("RUN_STARTED", {
        "run_dir": str(run_dir),
        "sandbox_dir": str(sandbox_dir),
        "scenario": scenario_config["name"],
        "ios_project_root": str(ios_project_root),
        "ios_source_dir": str(ios_source_dir),
        "app_delegate_path": str(app_delegate_path),
    })

    print(f"\n🚀 Run dir: {run_dir}")
    print(f"📦 Sandbox: {sandbox_dir}")
    print(f"📍 iOS project root: {ios_project_root}")
    print(f"📄 AppDelegate path: {app_delegate_path}")
    print(f"🧪 Scenario: {scenario_config['name']}")
    print(f"💬 Prompt: {scenario_config['prompt']}")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    dev_key = os.getenv("APPSFLYER_DEV_KEY")
    if not dev_key:
        raise RuntimeError("Missing APPSFLYER_DEV_KEY in .env")

    model = ChatOpenAI(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
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

    tools = await mcp_client.get_tools()

    mcp_tool_metadata = []
    for mcp_tool in tools:
        metadata = {
            "name": getattr(mcp_tool, "name", ""),
            "description": getattr(mcp_tool, "description", ""),
        }
        mcp_tool_metadata.append(metadata)
        recorder.write("TOOL_METADATA", metadata)

    discovered_tools = [getattr(t, "name", str(t)) for t in tools]

    recorder.write("TOOLS_DISCOVERED", {
        "tools": discovered_tools
    })

    print(f"🧰 MCP tools discovered: {discovered_tools}")

    from langchain_core.tools import tool

    @tool
    async def integrate_sdk_fixed(
            isResponseListener: bool = False,
            isDelegate: bool = False,
            isATT: bool = False,
            isCUID: bool = False,
    ) -> str:
        """Call this tool first to get or run the AppsFlyer SDK integration through the MCP server."""
        orig_tool = next(t for t in tools if t.name == "integrateSdk")

        result = await orig_tool.ainvoke({
            "projectPath": str(ios_project_root),
            "platform": "ios",
            "isResponseListener": isResponseListener,
            "isDelegate": isDelegate,
            "isATT": isATT,
            "isCUID": isCUID,
        })

        return str(result)

    @tool
    def patch_app_delegate_fixed() -> str:
        """Patch the real AppDelegate.swift file with AppsFlyer SDK integration code."""
        if not app_delegate_path.exists():
            return f"FAILED: AppDelegate.swift not found at {app_delegate_path}"

        content = app_delegate_path.read_text(encoding="utf-8")
        original_content = content

        if "import AppsFlyerLib" not in content:
            if "import AppTrackingTransparency" in content:
                content = content.replace(
                    "import AppTrackingTransparency",
                    "import AppTrackingTransparency\nimport AppsFlyerLib"
                )
            elif "import UIKit" in content:
                content = content.replace(
                    "import UIKit",
                    "import UIKit\nimport AppsFlyerLib"
                )
            else:
                content = "import AppsFlyerLib\n" + content

        if "AppsFlyerLib.shared().appsFlyerDevKey" not in content:
            old_signature = (
                "func application(_ application: UIApplication, "
                "didFinishLaunchingWithOptions launchOptions: "
                "[UIApplication.LaunchOptionsKey: Any]?) -> Bool {"
            )

            new_signature = (
                "func application(_ application: UIApplication, "
                "didFinishLaunchingWithOptions launchOptions: "
                "[UIApplication.LaunchOptionsKey: Any]?) -> Bool {\n"
                "        \n"
                f"        AppsFlyerLib.shared().appsFlyerDevKey = \"{dev_key}\"\n"
                f"        AppsFlyerLib.shared().appleAppID = \"{APPLE_APP_ID_FOR_SWIFT}\"\n"
                "        AppsFlyerLib.shared().delegate = self"
            )

            if old_signature in content:
                content = content.replace(old_signature, new_signature)
            else:
                return "FAILED: didFinishLaunchingWithOptions function signature was not found."

        if "AppsFlyerLib.shared().start()" not in content:
            old_empty_method = """@objc func didBecomeActiveNotification() {
    }"""

            new_method = """@objc func didBecomeActiveNotification() {
        AppsFlyerLib.shared().start()
    }"""

            if old_empty_method in content:
                content = content.replace(old_empty_method, new_method)
            else:
                content = content.replace(
                    "func application(_ application: UIApplication, continue userActivity: NSUserActivity, restorationHandler: @escaping ([UIUserActivityRestoring]?) -> Void) -> Bool {",
                    """@objc func didBecomeActiveNotification() {
        AppsFlyerLib.shared().start()
    }

    func application(_ application: UIApplication, continue userActivity: NSUserActivity, restorationHandler: @escaping ([UIUserActivityRestoring]?) -> Void) -> Bool {"""
                )

        if "AppsFlyerLib.shared().continue(userActivity, restorationHandler: nil)" not in content:
            content = content.replace(
                """func application(_ application: UIApplication, continue userActivity: NSUserActivity, restorationHandler: @escaping ([UIUserActivityRestoring]?) -> Void) -> Bool {
        return true
    }""",
                """func application(_ application: UIApplication, continue userActivity: NSUserActivity, restorationHandler: @escaping ([UIUserActivityRestoring]?) -> Void) -> Bool {
        AppsFlyerLib.shared().continue(userActivity, restorationHandler: nil)
        return true
    }"""
            )

        if "AppsFlyerLib.shared().handleOpen(url, options: options)" not in content:
            content = content.replace(
                """func application(_ app: UIApplication, open url: URL, options: [UIApplication.OpenURLOptionsKey : Any] = [:]) -> Bool {
        return true
    }""",
                """func application(_ app: UIApplication, open url: URL, options: [UIApplication.OpenURLOptionsKey : Any] = [:]) -> Bool {
        AppsFlyerLib.shared().handleOpen(url, options: options)
        return true
    }"""
            )

        if "AppsFlyerLib.shared().handlePushNotification(userInfo)" not in content:
            content = content.replace(
                """func application(_ application: UIApplication, didReceiveRemoteNotification userInfo: [AnyHashable : Any], fetchCompletionHandler completionHandler: @escaping (UIBackgroundFetchResult) -> Void) {
    }""",
                """func application(_ application: UIApplication, didReceiveRemoteNotification userInfo: [AnyHashable : Any], fetchCompletionHandler completionHandler: @escaping (UIBackgroundFetchResult) -> Void) {
        AppsFlyerLib.shared().handlePushNotification(userInfo)
    }"""
            )

        if "AppsFlyerLibDelegate" not in content:
            content = content.replace(
                "class AppDelegate: UIResponder, UIApplicationDelegate {",
                "class AppDelegate: UIResponder, UIApplicationDelegate, AppsFlyerLibDelegate {"
            )

        if "func onConversionDataSuccess" not in content:
            extension_code = """

extension AppDelegate {
    func onConversionDataSuccess(_ conversionInfo: [AnyHashable : Any]) {
        ConversionData = conversionInfo
        print("AppsFlyer conversion data success: \\(conversionInfo)")
    }

    func onConversionDataFail(_ error: Error) {
        print("AppsFlyer conversion data failed: \\(error.localizedDescription)")
    }

    func onAppOpenAttribution(_ attributionData: [AnyHashable : Any]) {
        print("AppsFlyer app open attribution: \\(attributionData)")
    }

    func onAppOpenAttributionFailure(_ error: Error) {
        print("AppsFlyer app open attribution failed: \\(error.localizedDescription)")
    }
}
"""
            content = content + extension_code

        app_delegate_path.write_text(content, encoding="utf-8")

        changed = content != original_content

        return json.dumps({
            "status": "PATCHED" if changed else "NO_CHANGES_NEEDED",
            "file": str(app_delegate_path),
            "contains_import": "import AppsFlyerLib" in content,
            "contains_dev_key": "AppsFlyerLib.shared().appsFlyerDevKey" in content,
            "contains_app_id": APPLE_APP_ID_FOR_SWIFT in content,
            "contains_start": "AppsFlyerLib.shared().start()" in content,
            "contains_delegate": "AppsFlyerLibDelegate" in content,
        }, ensure_ascii=False)

    @tool
    def verify_source_code_fixed() -> str:
        """Verify directly that AppDelegate.swift contains the expected AppsFlyer code."""
        if not app_delegate_path.exists():
            return f"FAILED: AppDelegate.swift not found at {app_delegate_path}"

        content = app_delegate_path.read_text(encoding="utf-8")

        checks = {
            "import AppsFlyerLib": "import AppsFlyerLib" in content,
            "appsFlyerDevKey": "AppsFlyerLib.shared().appsFlyerDevKey" in content,
            "appleAppID": "AppsFlyerLib.shared().appleAppID" in content,
            "start": "AppsFlyerLib.shared().start()" in content,
            "AppsFlyerLibDelegate": "AppsFlyerLibDelegate" in content,
        }

        status = "PASS" if all(checks.values()) else "FAIL"

        return json.dumps({
            "status": status,
            "file": str(app_delegate_path),
            "checks": checks,
        }, ensure_ascii=False, indent=2)

    @tool
    async def verify_sdk_fixed(action: str = "prepare") -> str:
        """Call this tool last to verify the SDK integration status through the MCP server."""
        orig_tool = next(t for t in tools if t.name == "verifyIosSdk")

        result = await orig_tool.ainvoke({
            "projectPath": str(ios_project_root),
            "action": action,
        })

        return str(result)

    custom_agent_tools = [
        integrate_sdk_fixed,
        patch_app_delegate_fixed,
        verify_source_code_fixed,
        verify_sdk_fixed,
    ]

    agent = create_react_agent(model, custom_agent_tools)

    try:
        await run_agent_with_simulated_user(
            agent=agent,
            initial_prompt=scenario_config["prompt"],
            recorder=recorder,
            simulation_context={
                "ios_project_root": ios_project_root,
                "app_id": APP_ID,
                "apple_app_id": APPLE_APP_ID_FOR_SWIFT,
                "mcp_tool_metadata": mcp_tool_metadata,
            },
            responder_model=model,
        )

    except Exception as e:
        recorder.write("RUN_ERROR", {
            "error_type": type(e).__name__,
            "error": str(e),
        })
        print(f"\n❌ Run failed: {type(e).__name__}: {e}")

    source_verify = run_command(
        [
            "grep",
            "-n",
            "AppsFlyerLib\\|appsFlyerDevKey\\|appleAppID\\|start()\\|AppsFlyerLibDelegate",
            str(app_delegate_path),
        ],
        cwd=sandbox_dir,
    )
    recorder.write("SOURCE_VERIFY_GREP", source_verify)

    git_status = run_command(["git", "status", "--porcelain"], cwd=sandbox_dir)
    recorder.write("GIT_STATUS", git_status)

    report = build_report(
        run_dir=run_dir,
        recorder=recorder,
        scenario=scenario_config,
        git_status=git_status,
    )

    report["source_verify"] = source_verify
    report["app_delegate_path"] = str(app_delegate_path)

    with (run_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n==============================")
    print("📊 FINAL REPORT")
    print("==============================")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n==============================")
    print("📄 CHECK THIS FILE")
    print("==============================")
    print(app_delegate_path)


if __name__ == "__main__":
    asyncio.run(run_agent())