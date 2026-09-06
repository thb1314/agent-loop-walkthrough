"""第二步：把工具声明交给模型，只观察调用，不执行。"""
import json
from packages.pi_ai.responses import output_text, parse_function_calls, response_output
from packages.pi_coding_agent.tools import ProjectFiles, create_project_tools
from .common import cli, connect, initial_history, make_payload, options


def main() -> None:
    args = options()
    client, config = connect(args)
    with ProjectFiles(args.project, 32 * 1024) as files:
        tools = create_project_tools(files)
        response = client.create(make_payload(config, initial_history(args.question), tools))
    output = response_output(response)
    print("模型输出（本阶段尚未执行工具）：")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if not parse_function_calls(output):
        print("本次模型没有请求工具；不能据此声称已读取项目。")
        print(output_text(output, response))


if __name__ == "__main__":
    cli(main)
