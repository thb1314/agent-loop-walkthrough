# 从0到1设计一个简易版AgentLoop：配套代码

这是一套与文章同步阅读的 Python 示例。每一步做的事情不同，不是给同一个最终程序换五个入口。

## 1. 环境

- Python 3.11 或以上；本次实际验证环境记录在上一级 `evidence/`。
- Linux / 支持 `openat`、`O_NOFOLLOW` 的 POSIX 系统。文件工具没有未经验证的 Windows 降级实现。
- 一个本地 Qwen 模型接口，支持本文使用的 **Responses API** 与工具调用。仅有 Chat Completions 端点不能直接替代。
- 示例本身只使用 Python 标准库。在本目录运行命令，无需先安装本包。

## 2. 连接已经启动的模型

```bash
export QWEN_BASE_URL=http://127.0.0.1:8021
export QWEN_MODEL=qwen3-0.6b
export QWEN_REASONING=none
export QWEN_MAX_OUTPUT_TOKENS=2048
```

可选配置：`QWEN_TIMEOUT`（默认120秒）、`QWEN_MAX_TURNS`（默认8轮）、`QWEN_API_KEY`（仅显式使用，不搜索个人凭证目录）。教学入口只允许连接本机回环地址，不向远端发送密钥。演示默认 temperature=0，但仍不承诺不同运行环境下的调用轨迹完全一致。

## 3. 按步骤运行

所有命令均在当前 `code/` 目录执行。五个阶段都围绕同一句提问：“帮我看看 reports 目录里的日报，告诉我这份日报的日期、订单数和营收。”用户只描述想查询的内容，不指定工具或操作顺序。

```bash
python3 -m examples.step01_model_call
python3 -m examples.step02_tool_request
python3 -m examples.step03_one_round
python3 -m examples.step04_agent_loop
python3 -m examples.step05_required_tool
```

| 入口 | 做什么 | 不做什么 |
| --- | --- | --- |
| step01 | 把日报查询需求直接发给模型，观察缺少数据时的回复 | 不提供文件工具 |
| step02 | 发送两个工具声明，观察 `list_files` 请求 | 不执行工具 |
| step03 | 执行 `list_files` 并回填，看到后续 `read_file` 请求 | 不执行第二个请求，明确提示任务未完成 |
| step04 | 第1轮列目录，第2轮读日报，第3轮模型回答 | 不硬编码调用顺序与文件名，不设必选工具 |
| step05 | 要求 `list_files` 先成功一次，再切回自动 | 这是程序策略；不把两种工具都设为强制调用 |

指定项目和运行上限：

```bash
python3 -m examples.step04_agent_loop --project ./sample_project --max-turns 8
```

默认项目目录由代码位置定位，不依赖作者的个人路径。`sample_project/reports/` 中有一份教学日报，日期为 2026-09-06、订单数为 128、营收为 9600 元。目录工具帮助程序找到文件，读取工具取得回答需要的数据。

`list_files` 返回相对项目根目录的路径。例如列出 `reports`，返回 `reports/daily-2026-09-06.json`，可以直接作为 `read_file` 的 `path`。

本例需要两轮工具调用，但共需三次模型请求。运行 `python3 -m examples.step04_agent_loop --max-turns 2` 会在两个工具执行后明确报错，因为没有剩余轮次生成最终回答。

## 4. 尚未启动模型时

以下是与原 mini-pi-python 一致的运行配套，文章不讲其内部实现。配套适配器源码保留在 `packages/qianchat/`，这样它与原有导入路径一致，不必为移动目录改写协议代码。

先取得并构建 llama.cpp（来源：<https://github.com/ggml-org/llama.cpp>），准备具有相应使用授权的 Qwen3 GGUF 模型文件。本文实际使用 `Qwen3-0.6B-Q8_0.gguf`，模型校验值和运行器版本见上一级 `evidence/runtime.json`。可从 Qwen 官方模型组织 <https://huggingface.co/Qwen> 核对模型来源，并选择可信的 GGUF 分发；不同转换版本不能仅凭文件名视为完全相同。

在第一个终端，将两个路径变量设置为你机器上的实际路径，然后启动模型：

```bash
export LLAMA_SERVER=/你安装的位置/llama-server
export QWEN_GGUF=/你存放模型的位置/Qwen3-0.6B-Q8_0.gguf
"$LLAMA_SERVER" -m "$QWEN_GGUF" -c 8192 -ngl all -fa on -np 1 --host 127.0.0.1 --port 8012 --no-ui
```

这里的两个中文路径是需替换的配置项，不是可以原样复制的真实路径。GPU 参数必须适合你的构建和硬件；不要把作者机器上的配置当作所有平台的保证。

在第二个终端进入当前代码目录：

```bash
python3 -m packages.qianchat --backend-url http://127.0.0.1:8012 --host 127.0.0.1 --port 8021 --model qwen3-0.6b
```

确认 `http://127.0.0.1:8021/health` 返回 `{"status":"ok"}` 后，在第三个终端运行前述示例。接口根地址可通过 `QWEN_BASE_URL` 修改；客户端会补上 `/responses`。

## 5. 代码阅读顺序

1. `examples/step01_model_call.py`：真实模型请求。
2. `packages/pi_agent/tools.py`：工具声明和执行结果。
3. `examples/step03_one_round.py`：没有循环时的一次执行与反馈。
4. `packages/pi_agent/loop.py`：完整 `run_agent()`。
5. `examples/step04_agent_loop.py`：把配置、模型、工具组合起来。
6. `packages/pi_coding_agent/tools/project.py` 与 `filesystem.py`：实际参数校验与只读访问。

`examples/common.py` 负责配置和打印，不隐藏循环逻辑。`packages/pi_coding_agent/cli.py` 是组合式命令行入口，默认带必选读取策略；本文默认入口始终是 `examples.step04_agent_loop`，不要混用两者。

## 6. 安全与许可证

文件工具只读取指定根目录内的 UTF-8 普通文件，拒绝绝对路径、`..`、符号链接、非普通文件及超出上限的读取；本实现增加严格 `path` 字段检查，并用非阻塞打开避免在拒绝 FIFO 前挂起。这些限制不是一个完整的多租户沙箱，也不阻止模型看见你主动授权目录内的敏感文件，因此只在不含凭证的演示目录运行。

本仓库不打包模型权重、个人认证文件或第三方文章 PDF。除另有注明的第三方内容外，本仓库内容以 [Apache License 2.0](LICENSE) 发布。

## 7. 不依赖模型的边界检查

在本目录运行：

```bash
python3 -m checks.loop_boundaries
```

检查覆盖历史配对、文本与调用共存、未知工具、非法 JSON、空响应、轮数上限，以及文件访问边界；还会用另一份文件名与数据，验证先列目录、再使用发现的路径读取、第三次请求才能回答。它用于稳定验证程序行为，不是用脚本输出冒充真实模型。

真实 Qwen 日志位于上一级 `evidence/`：`step04_agent_loop-daily.json` 记录主线中两种工具分属前后两轮；`step03_one_round-daily.json` 记录手动一轮为什么还不能完成；`max-turns-2.json` 记录轮数不足时的明确失败。
