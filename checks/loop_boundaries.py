"""稳定复现关键边界，不调用模型；在代码根目录用 python3 -m checks.loop_boundaries 运行。"""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

from packages.errors import MiniPiError
from packages.pi_agent import AgentLoopConfig, run_agent
from packages.pi_agent.tools import AgentTool
from packages.pi_coding_agent.tools import ProjectFiles, create_project_tools


def text(value):
    return {"output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": value}]}]}


def call(identifier, name="echo", arguments=None):
    return {"type": "function_call", "call_id": identifier, "name": name, "arguments": json.dumps(arguments or {"path": "x"})}


class InspectingModel:
    def __init__(self, responses, check=None):
        self.responses = iter(responses)
        self.requests = []
        self.check = check

    def create(self, payload):
        self.requests.append(copy.deepcopy(payload))
        if self.check:
            self.check(payload, len(self.requests))
        return next(self.responses)


def config(turns=4):
    return AgentLoopConfig(model="boundary-only", instructions="", max_turns=turns)


def must_fail(fn):
    try:
        fn()
    except MiniPiError:
        return
    raise AssertionError("Expected an explicit failure")


passed = []
executed = []
tool = AgentTool("echo", "验证用工具", {}, lambda arguments: executed.append(arguments["path"]) or arguments["path"])
first = text("这只是中间文本")
first["output"] += [call("id-a", arguments={"path": "a"}), call("id-b", arguments={"path": "b"})]


def check_pairing(payload, turn):
    if turn == 2:
        rows = payload["input"]
        assert rows[1:4] == first["output"]
        assert [(item["call_id"], item["output"]) for item in rows[4:]] == [("id-a", "a"), ("id-b", "b")]


model = InspectingModel([first, text("最终回答")], check_pairing)
assert run_agent(model, [tool], config(), "问题") == "最终回答"
assert executed == ["a", "b"]
passed.append("文本与调用共存不提前返回；同名调用按ID配对，模型原始输出先于结果")


def check_unknown(payload, turn):
    if turn == 2:
        last = payload["input"][-1]
        assert last["call_id"] == "unknown-id" and "未知工具" in last["output"]


assert run_agent(InspectingModel([{"output": [call("unknown-id", "missing")]}, text("已纠正")], check_unknown), [tool], config(), "问题") == "已纠正"
passed.append("未知工具以关联结果反馈给下一轮")

must_fail(lambda: run_agent(InspectingModel([{"output": []}]), [], config(), "问题"))
passed.append("空响应明确失败")
looping = InspectingModel([{"output": [call("one")]}, {"output": [call("two")]}])
must_fail(lambda: run_agent(looping, [tool], config(2), "问题"))
assert len(looping.requests) == 2
passed.append("轮数耗尽停止模型请求")
must_fail(lambda: run_agent(InspectingModel([]), [tool, tool], config(), "问题"))
passed.append("重复工具名在模型调用前被拒绝")
bad_call = call("bad-json")
bad_call["arguments"] = "not JSON"
must_fail(lambda: run_agent(InspectingModel([{"output": [bad_call]}]), [tool], config(), "问题"))
passed.append("非法JSON调用参数明确协议失败")

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    (root / "project").mkdir()
    (root / "outside.txt").write_text("outside-secret", encoding="utf-8")
    (root / "project" / "inside.txt").write_text("inside", encoding="utf-8")
    (root / "project" / "escape.txt").symlink_to(root / "outside.txt")
    import os
    os.mkfifo(root / "project" / "pipe")
    with ProjectFiles(root / "project", 128) as files:
        registered = {t.name: t for t in create_project_tools(files)}
        read_tool = registered["read_file"]
        assert read_tool.execute({"path": "inside.txt"}).content == "inside"
        for arguments in [{"path": "../outside.txt"}, {"path": "escape.txt"}, {"path": 42}, {"path": "inside.txt", "extra": True}]:
            result = read_tool.execute(arguments)
            assert result.is_error and "outside-secret" not in result.content
        failed = read_tool.execute({"path": "absent.txt"})
        assert failed.is_error
        assert read_tool.execute({"path": "pipe"}).is_error
        assert read_tool.execute({"path": "."}).is_error
passed.append("只读工具可读取正常文件，拒绝越界、符号链接、错误参数、目录和FIFO")


class DirectoryThenReadModel:
    """验证用模型：下一次请求必须能看到上次真实工具的结果。"""

    def __init__(self):
        self.turn = 0

    def create(self, payload):
        self.turn += 1
        if self.turn == 1:
            return {"output": [call("list-id", "list_files", {"path": "reports"})]}
        result = payload["input"][-1]
        assert result["type"] == "function_call_output"
        if self.turn == 2:
            assert result["call_id"] == "list-id"
            # 不预先填写文件名；读取参数直接来自目录工具的输出。
            discovered_path = result["output"]
            assert discovered_path.startswith("reports/")
            return {"output": [call("read-id", "read_file", {"path": discovered_path})]}
        assert self.turn == 3 and result["call_id"] == "read-id"
        results = [row for row in payload["input"] if row.get("type") == "function_call_output"]
        assert [row["call_id"] for row in results] == ["list-id", "read-id"]
        data = json.loads(result["output"])
        return text(f"{data['orders']} 笔，{data['revenue_yuan']} 元")


with tempfile.TemporaryDirectory() as dependent_temporary:
    project = Path(dependent_temporary)
    (project / "reports").mkdir()
    (project / "reports" / "only-after-list-17.json").write_text(
        json.dumps({"orders": 73, "revenue_yuan": 5840}), encoding="utf-8"
    )
    with ProjectFiles(project, 128) as files:
        tools = create_project_tools(files)
        dependent_model = DirectoryThenReadModel()
        assert run_agent(dependent_model, tools, config(3), "找到并读取日报") == "73 笔，5840 元"
        assert dependent_model.turn == 3
        must_fail(lambda: run_agent(DirectoryThenReadModel(), tools, config(2), "找到并读取日报"))
passed.append("先列目录再按发现的路径读取：两种工具分属两轮，第三次请求才能回答")
print(json.dumps({"kind": "deterministic-boundary-smoke", "passed": passed}, ensure_ascii=False, indent=2))
