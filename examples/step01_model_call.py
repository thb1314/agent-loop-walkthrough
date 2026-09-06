"""第一步：只发一次请求，不注册文件工具。"""
from dataclasses import replace
from packages.pi_ai.responses import output_text, response_output
from .common import cli, connect, initial_history, make_payload, options


def main() -> None:
    args = options()
    client, config = connect(args)
    config = replace(config, instructions="你还没有获得本地日报的内容。请说明回答问题还需要什么信息，不要猜测日报数据。")
    response = client.create(make_payload(config, initial_history(args.question), []))
    print(output_text(response_output(response), response))


if __name__ == "__main__":
    cli(main)
