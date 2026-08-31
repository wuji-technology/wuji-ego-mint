"""视频过滤 op 注册表

每个 op 文件需导出（鸭子接口，无强制基类）:
    NAME: str
        子命令名（如 "hand_filter"）
    DESCRIPTION: str
        简介，用于 --help / --list
    add_arguments(parser) -> None
        在主解析器上注册 op 独有参数（公共参数已由 main 注入）
    default_output_suffix(args) -> str
        --output 未指定时，输出目录推断后缀（如 "_hand_filter"）
    process_video(video, output_dir, args) -> ProcessResult
        处理单个视频（同步、单线程）
        — 或 —
    process_videos(items, args) -> list[ProcessResult]
        op 自带批处理调度（如 hand_filter 的多进程流水线）。main.py 优先
        调用本函数，存在时跳过 ThreadPoolExecutor。items 元素为
        (video_path, per_video_output_dir)，per_video_output_dir 已经
        镜像源目录结构。

可选:
    describe_config(args) -> str       一行打印 op 当前配置
    plan(video, args) -> str           dry-run 时输出的计划字符串
    output_label(args) -> str          --output 指定时，在其下创建的子目录后缀标签
    IS_TERMINAL: bool                  默认 False。True 表示该 op 控制输出阶段
                                       （hand_filter 自己写多段 mp4 + .done 标记，
                                       属于 terminal）

新增 op 只需在本文件里 import + 注册即可，无需改 main.py。当出现第二个 op
且需要组合（pipeline / 多阶段）时，参考 reprocessed/ 的 PipelineOp / split_stages
范式回填。
"""
# 注意: action_align.py 不是 main.py 的 op(无 NAME/process_video), 它是带自己 main()
# 的独立脚本(filtered_v2 → task_split), 直接 `python filtered/ops/action_align.py` 调用,
# 不在此注册, 否则 main.py 加载 OPS 时会 AttributeError。
from . import hand_filter

OPS = {
    hand_filter.NAME: hand_filter,
}
