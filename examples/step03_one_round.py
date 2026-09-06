"""第三步：执行一批工具，回填结果，再请求一次，不自动循环。"""
import json
from packages.errors import MiniPiError
from packages.pi_agent import ToolResult
from packages.pi_ai.responses import output_text, parse_function_calls, response_output
from packages.pi_coding_agent.tools import ProjectFiles, create_project_tools
from .common import cli, connect, initial_history, make_payload, options


def main() -> None:
    args = options()
    client, config = connect(args)
    history = initial_history(args.question)
    with ProjectFiles(args.project, 32 * 1024) as files:
        tools = create_project_tools(files)
        tools_by_name = {tool.name: tool for tool in tools}
        response = client.create(make_payload(config, history, tools))
        output = response_output(response)
        calls = parse_function_calls(output)
        history.extend(output)
        if not calls:
            text = output_text(output, response)
            if not text:
                raise MiniPiError("模型既没有文本，也没有工具调用")
            print("模型直接回答，本次未经过工具执行：")
            print(text)
            return
        for call in calls:
            print(f"执行 {call.name}，call_id={call.call_id}")
            tool = tools_by_name.get(call.name)
            result = tool.execute(call.arguments) if tool else ToolResult(f"未知工具：{call.name}", is_error=True)
            print(result.content)
            history.append({"type": "function_call_output", "call_id": call.call_id, "output": result.content})
        next_response = client.create(make_payload(config, history, tools))
        next_output = response_output(next_response)
        print("回填之后，模型的下一次输出：")
        print(json.dumps(next_output, ensure_ascii=False, indent=2))
        if parse_function_calls(next_output):
            print("模型还需要工具。本阶段到此结束，任务尚未完成；下一步交给循环继续。")
        else:
            text = output_text(next_output, next_response)
            if not text:
                raise MiniPiError("模型既没有文本，也没有工具调用")
            print("最终回答：", text)


if __name__ == "__main__":
    cli(main)
