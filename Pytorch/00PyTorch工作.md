### 1. PyTorch: 深入框架的 C++ 内核

对于底层开发者，你关注的早已不是 `nn.Linear` 或 `model.train()`。你关注的是这些 Python API 背后的 C++ 世界。

*   **你的核心身份：** 框架核心开发者 (Framework Core Developer)
*   **你的工作内容：**
    1.  **编写自定义算子 (Custom Operators)：** 当 PyTorch 自带的算子无法满足性能需求或功能缺失时（例如，为一种新的注意力变种或量化算法编写 Kernel），你需要使用 C++ 和 CUDA/ROCm 编写高性能算子，并通过 `torch.utils.cpp_extension` 将其无缝集成到 PyTorch 中。
    2.  **优化调度器与执行引擎：** 深入 PyTorch 的核心调度器 `Aten/C10`，优化其分发逻辑，减少 `Kernel` 启动开销，或者改进其内存管理机制。
    3.  **开发 `torch.compile` 后端 (Inductor)：** 这是目前最前沿的方向。`torch.compile` 将 PyTorch 的动态图“编译”成静态优化的代码。底层开发者会为其后端 Inductor 贡献代码，例如：
        *   使用 **Triton** 为特定的模型模式（Pattern）编写更高效的融合 `Kernel` 模板。
        *   改进 Inductor 的代码生成逻辑，使其能更好地利用硬件特性（如 Tensor Cores）。
    4.  **硬件后端集成 (Hardware Backend Integration)：** 如果你的公司研发了一款新的 NPU/AI 加速器，你的任务就是编写一个 PyTorch 的硬件后端。这需要你实现一套完整的算子库，并让 PyTorch 的调度器能够正确地将计算任务分派到你的硬件上。这被称为 "out-of-tree" 后端开发。
*   **所需核心技能：** C++, CUDA/ROCm, Triton, Python C-API, 系统编程，对 PyTorch 内部架构的深刻理解。