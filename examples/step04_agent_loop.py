"""第四步：自动选择工具的完整 Agent Loop。"""
from packages.pi_agent import run_agent
from packages.pi_coding_agent.tools import ProjectFiles, create_project_tools
from .common import cli, connect, options, report


def main() -> None:
    args = options()
    client, config = connect(args)
    print("模式：自动选择工具")
    print("问题：", args.question)
    with ProjectFiles(args.project, 32 * 1024) as files:
        answer = run_agent(client, create_project_tools(files), config, args.question, report)
    print("\n[最终回答]\n" + answer)


if __name__ == "__main__":
    cli(main)
