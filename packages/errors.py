"""mini-pi 各层共享的用户可见错误。"""


class MiniPiError(RuntimeError):
    """可以安全展示给命令行用户的运行错误。"""
