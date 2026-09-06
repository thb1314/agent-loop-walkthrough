"""第五步：要求先列目录的对照；与主线的自动工具选择区分。"""
from dataclasses import replace
from packages.pi_agent import run_agent
from packages.pi_coding_agent.tools import ProjectFiles, create_project_tools
from .common import cli, connect, options, report


def main() -> None:
    args = options()
    client, config = connect(args)
    config = replace(config, required_tool_names=("list_files",))
    print("模式：要求 list_files 成功一次，然后自动选择")
    print("问题：", args.question)
    with ProjectFiles(args.project, 32 * 1024) as files:
        answer = run_agent(client, create_project_tools(files), config, args.question, report)
    print("\n[最终回答]\n" + answer)


if __name__ == "__main__":
    cli(main)
