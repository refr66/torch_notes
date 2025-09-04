很好，你已经掌握了我们分析系统的“物理学基础”。现在，让我们拿起手术刀，解剖 Megatron-LM 最为精巧和核心的构件：**张量并行 (Tensor Parallelism)**。

忘掉那些高层次的博客图。我们是系统工程师，我们要看到的是**数据在 CUDA Stream 上的流动，以及计算图在 `autograd` 引擎中的真实形态**。

---

### **大师系列之二：解剖张量并行 - 矩阵数学与 Autograd 的共舞 (Dissecting Tensor Parallelism: A Dance of Matrix Math and Autograd)**

张量并行的本质，是将一个巨大的矩阵运算，分解为多个小矩阵运算和通信的组合。这个想法最早由 [Megatron-LM 论文](https://arxiv.org/abs/1909.08053) 提出，并主要应用于 Transformer 的两个核心组件：**MLP Block** 和 **Self-Attention Block**。

让我们先从更简单的 MLP Block 开始。一个标准的 MLP Block 包含两个线性层和一个激活函数：
`Y = GELU(X * A) * B`
其中，`X` 是输入，`A` 和 `B` 是权重矩阵。

在 Megatron-LM 中，这个块被实现为 `ParallelMLP`。它并没有使用两个标准的 `nn.Linear`，而是使用了两个我们自己定义的 Module：`ColumnParallelLinear` 和 `RowParallelLinear`。这就是我们解剖的起点。

#### **第一刀：列并行线性层 (ColumnParallelLinear)**

`ColumnParallelLinear` 用于处理第一个线性层：`Z = GELU(X * A)`。

**1. 数学分解**

假设我们有 `P` 个 GPU 参与张量并行。我们将权重矩阵 `A` **按列 (column-wise)** 切分成 `P` 份：`A = [A_1, A_2, ..., A_P]`。

现在，矩阵乘法 `X * A` 就可以被分解为：
`X * [A_1, A_2, ..., A_P] = [X * A_1, X * A_2, ..., X * A_P]`

这意味着，每个 GPU `i` 只需要存储 `A_i`，然后独立地计算 `Z_i = X * A_i`。注意，**输入 `X` 在所有 GPU 上都是完全相同的**。

![Column Parallelism](https://lilianweng.github.io/posts/2021-09-25-parallelism/megatron-lm-tensor-parallelism-column.png)
*(图片来源: Lilian Weng's Blog)*

**2. 代码实现与 Autograd 的“欺骗”**

打开 `megatron/core/tensor_parallel/layers.py`，找到 `ColumnParallelLinear`。它的 `forward` 方法核心如下（简化版）：

```python
# megatron/core/tensor_parallel/layers.py

class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, input_size, output_size_per_partition, ...):
        # 注意：权重 A 的形状是 [output_size_per_partition, input_size]
        # 它只存储被切分后的那一小块
        self.weight = Parameter(torch.empty(self.output_size_per_partition, self.input_size, ...))
        ...

    def forward(self, input_):
        # input_ 在所有 TP ranks 上是相同的
        # 我们需要先把它广播给所有 TP ranks
        input_parallel = copy_to_tensor_parallel_region(input_)

        # [input_size] x [output_size_per_partition, input_size]^T -> [output_size_per_partition]
        # 这是核心的本地计算，只产生输出的一部分
        output_parallel = F.linear(input_parallel, self.weight)
        
        # 在这里，每个 GPU 都有了 Z_i = X * A_i 的一部分
        # 我们需要将它们拼接起来，得到完整的 Z = [Z_1, Z_2, ...]
        output = gather_from_tensor_parallel_region(output_parallel)
        
        return output
```

这里出现了两个神秘的函数：`copy_to_tensor_parallel_region` 和 `gather_from_tensor_parallel_region`。它们是 `torch.autograd.Function` 的子类，是实现张量并行的魔法所在。

*   **`copy_to_tensor_parallel_region(input_)`**:
    *   **Forward**: 在前向传播中，它本质上是一个 **Identity** 操作 `f(x)=x`。它什么都不做，只是确保输入 `X` 被正确地传递到后续的计算中。
    *   **Backward**: 这是关键！在反向传播中，当梯度从各个 GPU 传回时，`copy_to_tensor_parallel_region` 的 `backward` 函数会执行一次 **All-Reduce**。为什么？因为每个 GPU 只根据它自己的权重分片 `A_i` 计算了对 `X` 的梯度 `dL/dX_i`。而 `X` 是共享的，它收到的总梯度应该是所有分片梯度之和：`dL/dX = sum(dL/dX_i)`。这正是 `All-Reduce` 做的事情。

*   **`gather_from_tensor_parallel_region(output_parallel)`**:
    *   **Forward**: 在前向传播中，它的 `forward` 函数执行一次 **All-Gather**。它将每个 GPU 上的 `output_parallel` (即 `Z_i`) 收集起来，并沿着指定维度拼接，使得**每个 GPU 都得到完整的 `Z`**。
    *   **Backward**: 在反向传播中，梯度 `dL/dZ` 会传进来。它的 `backward` 函数执行一次 **Split/Scatter**。它将完整的梯度 `dL/dZ` 切分开，只把对应的那一部分 `dL/dZ_i` 传回给产生 `Z_i` 的那个本地计算。

**总结一下 `ColumnParallelLinear` 的数据流：**

*   **Forward**: `Input -> Identity -> MatMul (local) -> All-Gather -> Output`
*   **Backward**: `Grad_Output -> Split -> Grad_MatMul (local) -> All-Reduce -> Grad_Input`

#### **第二刀：行并行线性层 (RowParallelLinear)**

`RowParallelLinear` 用于处理第二个线性层：`Y = Z * B`。

**1. 数学分解**

现在，输入 `Z` 是一个很大的张量（在前一步 `ColumnParallelLinear` 之后，每个 GPU 都持有一份完整的 `Z`）。我们将第二个权重矩阵 `B` **按行 (row-wise)** 切分：
`B = [B_1; B_2; ...; B_P]` (分号表示垂直堆叠)

矩阵乘法 `Z * B` 无法直接分解。但是，我们可以先将输入 `Z` 按列切分 `Z = [Z_1, Z_2, ..., Z_P]`（这正好是 `ColumnParallelLinear` 中 `All-Gather` 之前的状态！），然后计算：
`Y = [Z_1, Z_2, ..., Z_P] * [B_1; B_2; ...; B_P] = Z_1*B_1 + Z_2*B_2 + ... + Z_P*B_P`

这个公式告诉我们：每个 GPU `i` 可以独立计算 `Y_i = Z_i * B_i`。然后，最终的结果 `Y` 是所有这些部分和 `Y_i` 的**总和 (Summation)**。

![Row Parallelism](https://lilianweng.github.io/posts/2021-09-25-parallelism/megatron-lm-tensor-parallelism-row.png)
*(图片来源: Lilian Weng's Blog)*

**2. 代码实现与 Autograd 的“欺骗”**

打开 `RowParallelLinear` 的代码：

```python
# megatron/core/tensor_parallel/layers.py

class RowParallelLinear(torch.nn.Module):
    def __init__(self, input_size_per_partition, output_size, ...):
        # 权重 B 的形状是 [output_size, input_size_per_partition]
        # 它也只存储被切分后的一小块
        self.weight = Parameter(torch.empty(self.output_size, self.input_size_per_partition, ...))
        ...

    def forward(self, input_):
        # input_ 是完整的 Z，我们需要先把它切分
        input_parallel = scatter_to_tensor_parallel_region(input_)

        # [input_size_per_partition] x [output_size, input_size_per_partition]^T -> [output_size]
        # 本地计算，得到部分和 Y_i
        output_parallel = F.linear(input_parallel, self.weight)

        # 将所有 GPU 上的部分和 Y_i 相加
        output = reduce_from_tensor_parallel_region(output_parallel)

        return output
```

我们又遇到了两个新的 `autograd.Function`：

*   **`scatter_to_tensor_parallel_region(input_)`**:
    *   **Forward**: 执行 **Split/Scatter**。将完整的输入 `Z` 切分，每个 GPU 只得到它需要的部分 `Z_i`。
    *   **Backward**: 执行 **All-Gather**。因为 `Z` 是由前一个 `All-Gather` 产生的，它的梯度需要被收集起来传回去。

*   **`reduce_from_tensor_parallel_region(output_parallel)`**:
    *   **Forward**: 执行 **All-Reduce**。将所有 GPU 上的部分和 `Y_i` 相加，得到最终的 `Y`。
    *   **Backward**: 执行 **Identity**。因为 `Y` 是最终输出，它的梯度 `dL/dY` 在所有 GPU 上都应该是一样的，直接向后传递即可。

**总结一下 `RowParallelLinear` 的数据流：**

*   **Forward**: `Input -> Split -> MatMul (local) -> All-Reduce -> Output`
*   **Backward**: `Grad_Output -> Identity -> Grad_MatMul (local) -> All-Gather -> Grad_Input`

---

### **系统级的洞见 (System-level Insights)**

1.  **通信的隐藏**: `ColumnParallelLinear` 的 `All-Gather` 和 `RowParallelLinear` 的 `All-Reduce` 是两个主要的通信开销。但是，Megatron-LM 做了一个绝妙的优化：**`RowParallelLinear` 的输入 `Z`，可以直接使用 `ColumnParallelLinear` 在 `All-Gather` 之前的那个未拼接的状态 `[Z_1, Z_2, ...]`！** 这样，一个 `All-Gather` 和一个 `Split` 操作就被抵消了，从而减少了一次通信。这在 `ParallelMLP` 的 `forward` 函数中实现。

2.  **计算与通信的重叠**: 在现代 GPU 上，计算和通信可以发生在不同的 CUDA Stream 上。一个优化的系统会尝试让 `MatMul` 的计算与下一次迭代的 `All-Reduce` 通信重叠，以隐藏通信延迟。这是更深层次的性能优化。

3.  **对 Autograd 的“滥用”**: 我们实际上是在“欺骗” PyTorch 的自动求导引擎。通过自定义的 `autograd.Function`，我们告诉它：“在前向传播时，你把这个函数当作 A；但在反向传播时，请把它当作 B”。这种对计算图的手动干预，是所有高级并行库（包括 DeepSpeed、Fairscale 等）的核心技术。

**今日的解剖到此结束。**

我们从矩阵数学出发，深入到 Megatron-LM 的 Python 和 Autograd 实现，揭示了张量并行是如何通过一系列精心设计的计算与通信步骤来实现的。我们看到，它不仅仅是简单的切分矩阵，更是一场数学、Autograd 引擎和分布式通信原语之间协同的精妙舞蹈。

在下一次，我们将用同样的“手术刀”，解剖更为复杂的 **Self-Attention Block**。在那里，我们将看到 Q, K, V 的并行化是如何引入更多的通信，以及 Megatron-LM 是如何通过融合核（Fused Kernels）来进一步优化这个瓶颈的。请提前预习 Transformer 的多头注意力机制。