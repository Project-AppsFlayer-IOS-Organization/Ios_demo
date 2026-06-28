from typing import Any, Dict, List, Optional


def _last_ai_message(chat_history: List[Any]) -> str:
    for msg in reversed(chat_history):
        msg_type = getattr(msg, "type", "")
        content = str(getattr(msg, "content", ""))

        if msg_type == "ai" and content.strip():
            return content

    return ""


def handle_detected_question(chat_history: List[Any]) -> Optional[str]:
    last_question = _last_ai_message(chat_history).lower()

    if "android" in last_question or "ios" in last_question:
        return (
            "Use iOS only. "
            "The provided project path is an iOS project."
        )

    if "scene delegate" in last_question:
        return (
            "No, do not support Scene Delegate. "
            "Use the basic iOS AppDelegate integration only."
        )

    if "response listener" in last_question:
        return (
            "No response listener. "
            "Use the basic iOS SDK integration."
        )

    if "att" in last_question or "tracking transparency" in last_question:
        return (
            "Do not include ATT setup. "
            "Use the basic iOS SDK integration."
        )

    if "customeruserid" in last_question or "customer user id" in last_question:
        return (
            "Do not include customerUserID setup. "
            "Use the basic iOS SDK integration."
        )

    if "verify" in last_question or "verification" in last_question:
        return (
            "Before verification, make sure you actually edited the project files "
            "using list_project_files, read_project_file, and write_to_project_file. "
            "Then verify the iOS SDK integration using verifyIosSdk."
        )

    return (
        "Continue with iOS only. "
        "Use AppsFlyer MCP output and edit the project files using the generic file tools."
    )