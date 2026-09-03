"""视频处理 op 注册表

每个 op 文件需导出（鸭子接口，无强制基类）:
    NAME: str
        子命令名（如 "segment"）
    DESCRIPTION: str
        简介，用于 --help / --list
    add_arguments(parser) -> None
        在自己的 subparser 上注册 op 独有参数（公共参数已由 main 注入）
    default_output_suffix(args) -> str
        --output 未指定时，输出目录推断后缀（如 "_Tseg60"）
    process_video(video: Path, output_dir: Path, args) -> ProcessResult
        处理单个视频；负责在 output_dir 下写输出文件

可选:
    describe_config(args) -> str       一行打印 op 当前配置
    plan(video: Path, args) -> str     dry-run 时输出的计划字符串
    output_label(args) -> str          --output 指定时，在其下创建的子目录后缀标签
                                       （形如 "Tseg60"，最终目录为
                                       "<input_dir_name>_<label>"）。未定义则
                                       退回 default_output_suffix() 去掉前导 "_"。

组合 op (pipeline)：
    在命令行用 `--ops a+b+c`，由 ops/pipeline.py 动态构造，把多个 op 的 vf 片段拼
    成一次 ffmpeg 调用（1 decode + 1 encode）。要参与组合的 op 需要额外提供：

    vf_chain(meta: dict, args) -> str | None
        返回本 op 在 -vf 链中贡献的过滤器片段（如 "fps=30" / "scale=1280:720:..."）。
        无变化（已满足）返回 None。
    IS_TERMINAL: bool                    默认 False。True 表示该 op 控制输出阶段
                                         （如 segment 用 -f segment muxer）。

新增 op 只需在本文件里 import + 注册即可，无需改 main.py。
"""
from . import fps, resolution, time_segment

OPS = {
    fps.NAME: fps,
    resolution.NAME: resolution,
    time_segment.NAME: time_segment,
}

# 说明：上游还有 action_segment（按动作切段，依赖 VLM 标注）与 fisheye_undistort
# （鱼眼去畸变）两个 op，未随本仓库发布 —— 前者属于打标流程，后者在本仓库已由
# ego_pipeline/backends/undistort_gpu.py 提供 GPU 实现。
