configuration_llama.py

# 代码说明

## 0.1 主要功能

该代码定义了一个名为 `LlamaConfig` 的配置类，用于配置 LLaMA（大型语言模型）的参数。这些参数控制了模型的结构和行为，包括词汇大小、隐藏层尺寸、注意力头数量等。该配置类是基于 Hugging Face 的 `PretrainedConfig` 类，并提供了初始化模型所需的所有必要参数。

## 0.1 架构说明

代码的整体架构包括以下几个部分：

- 引入必要的库和模块。
- 定义 `LlamaConfig` 类，继承自 `PretrainedConfig`。
- 在 `LlamaConfig` 类中定义了模型的类型和一些默认配置。
- 提供了一个构造函数，允许用户自定义模型的配置。
- 包含了一些辅助函数和属性，如旋转位置编码（RoPE）的配置验证。
- 定义了默认的张量并行计划。
- 最后，导出了 `LlamaConfig` 类，使其可以被其他模块使用。

## 0.1 关键组件

以下是一些主要的类和函数：

- `LlamaConfig`: 配置类，负责定义和初始化 LLaMA 模型的所有配置选项。
- `PretrainedConfig`: Hugging Face 的预训练模型配置基类，提供了配置模型的基本方法和属性。
- `rope_config_validation`: 旋转位置编码配置验证函数，用于确保 RoPE 相关配置的正确性。

其他关键组件包括：

- 类属性和方法，如 `model_type`, `keys_to_ignore_at_inference`, `base_model_tp_plan`, `base_model_pp_plan` 等。
- 构造函数 `__init__`，它接收多个参数来初始化配置对象。
- `super().__init__` 调用，用于初始化基类 `PretrainedConfig`。
- `__all__` 变量，用于定义模块中导出的符号。

# 类分析

以下是代码中定义的类的详细分析：

LlamaConfig

## 0.1 功能描述

该类`LlamaConfig`是用于存储`LlamaModel`配置信息的配置类。它根据指定的参数实例化LLaMA模型，定义模型架构。使用默认参数实例化的配置将与LLaMA-7B的配置相似。配置对象继承自`PretrainedConfig`，可用于控制模型输出。

## 0.1 参数说明

- `vocab_size` (`int`, 可选, 默认为32000): LLaMA模型的词汇量，定义了`input_ids`传入`LlamaModel`时可以表示的不同标记的数量。
- `hidden_size` (`int`, 可选, 默认为4096): 隐藏表示的维度。
- `intermediate_size` (`int`, 可选, 默认为11008): MLP表示的维度。
- `num_hidden_layers` (`int`, 可选, 默认为32): Transformer解码器中的隐藏层数量。
- `num_attention_heads` (`int`, 可选, 默认为32): Transformer解码器中每个注意力层的注意力头数。
- `num_key_value_heads` (`int`, 可选): 用于实现分组查询注意力的键值头数。如果`num_key_value_heads=num_attention_heads`，模型将使用多头注意力（MHA），如果`num_key_value_heads=1`，模型将使用多查询注意力（MQA），否则使用GQA。
- `hidden_act` (`str` 或 `function`, 可选, 默认为 `"silu"`): 解码器中的非线性激活函数（函数或字符串）。
- `max_position_embeddings` (`int`, 可选, 默认为2048): 模型可能使用的最大序列长度。
- `initializer_range` (`float`, 可选, 默认为0.02): 用于初始化所有权重矩阵的`truncated_normal_initializer`的标准差。
- `rms_norm_eps` (`float`, 可选, 默认为1e-06): rms归一化层使用的epsilon。
- `use_cache` (`bool`, 可选, 默认为`True`): 模型是否应返回最后的键/值注意力（不是所有模型都使用）。
- `pad_token_id` (`int`, 可选): 填充标记id。
- `bos_token_id` (`int`, 可选, 默认为1): 流开始标记id。
- `eos_token_id` (`int`, 可选, 默认为2): 流结束标记id。
- `pretraining_tp` (`int`, 可选, 默认为1): 预训练期间使用的实验特性，张量并行主义等级。
- `tie_word_embeddings` (`bool`, 可选, 默认为`False`): 是否绑定权重嵌入。
- `rope_theta` (`float`, 可选, 默认为10000.0): RoPE嵌入的基础周期。
- `rope_scaling` (`Dict`, 可选): 包含RoPE嵌入缩放配置的字典。
- `attention_bias` (`bool`, 可选, 默认为`False`): 自注意力中是否在查询、键、值和输出投影层使用偏置。
- `attention_dropout` (`float`, 可选, 默认为0.0): 注意力概率的丢弃率。
- `mlp_bias` (`bool`, 可选, 默认为`False`): MLP层中是否在上/下/门投影层使用偏置。
- `head_dim` (`int`, 可选): 注意力头维度。如果为None，默认为`hidden_size // num_attention_heads`。

## 0.1 返回值

该类实例化后返回一个`LlamaConfig`对象，该对象包含定义LLaMA模型所需的所有配置信息。

## 0.1 实现逻辑

- `LlamaConfig`类继承自`PretrainedConfig`类，它提供了配置模型所需的所有参数和默认值。
- 通过设置不同的参数，可以调整模型的大小、复杂性以及行为。
- `rope_scaling`参数是一个复杂的配置，它允许对RoPE嵌入进行不同类型的缩放，以适应不同的序列长度和模型类型。
- `rope_config_validation`函数用于验证RoPE配置的正确性。
- 构造函数`__init__`中设置了所有配置参数，并在需要时提供默认值，然后调用父类的构造函数以完成初始化。
