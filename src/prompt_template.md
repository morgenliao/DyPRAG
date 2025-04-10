# prompt_template.py

# 代码说明

This script is a prompt template for main experiments, likely part of a larger project for a language model or a question-answering system. It is adopted from the PRAG repository and includes functions to generate prompts with or without few-shot examples.

## 0.1 主要功能

The primary functions of this code are to:

- Generate a user prompt for a question-answering task with or without few-shot examples.
- Provide a structure for the assistant's response.
- Process the input question and passages to ensure they are in the correct format.
- Load few-shot examples based on the dataset provided.

## 0.1 架构说明

The code is structured with the following components:

- Import statements at the top to include necessary libraries.
- Global variables to store the current dataset and few-shot examples.
- Functions to handle prompt generation and data loading.
- A main function (`get_prompt`) that ties everything together and returns tokenized inputs for the model.

## 0.1 关键组件

- Functions:
  - `_get_prompt`: Processes the question and passages and ensures the answer is properly formatted.
  - `get_fewshot`: Loads few-shot examples for a given dataset.
  - `get_prompt`: Generates the final prompt with or without few-shot examples, ready to be tokenized.
- Variables:
  - `current_dataset`: Stores the current dataset name.
  - `fewshot`: Stores the few-shot examples.
  - `USER_PROMPT` and `USER_PROMPT_WITH_COT`: Templates for user prompts.
  - `ASSISTANT_PROMPT` and `ASSISTANT_PROMPT_WITH_COT`: Templates for assistant responses.
- Other Components:
  - `tokenizer`: An assumed external component that tokenizes the text.
  - `ROOT_DIR`: A path to the root directory of the project, used to construct file paths.

# 函数分析

以下是代码中定义的函数的详细分析：

_get_prompt

下面是根据您提供的格式对该函数的分析：

## 0.1 功能描述

该函数的主要功能是处理传入的问题、段落和答案，以便生成一个格式化的提示。它确保问题以问号结尾，如果提供了段落，则将这些段落转换为列表形式，并确保答案（如果存在）以句号结尾。

## 0.1 参数说明

- `question`: 一个字符串，表示需要格式化的问题。
- `passages`: 可选参数，可以是一个字符串或字符串列表，表示与问题相关的段落。
- `answer`: 可选参数，一个字符串，表示问题的答案。

## 0.1 返回值

返回一个包含三个元素的元组：

- `question`: 格式化后的问题，确保以问号结尾。
- `passages`: 如果提供了段落，则是一个列表形式的段落。
- `answer`: 格式化后的答案，如果提供了答案，确保以句号结尾。

## 0.1 实现逻辑

1. `question`参数首先被去除了首尾空格，然后检查是否以问号结尾。如果不是，则添加问号。
2. 如果`question`以" ?"结尾，则移除末尾的空格并添加问号。
3. 检查`passages`参数，如果它不是列表类型，则将其转换为包含单个元素的列表。
4. 如果`answer`参数未提供（为`None`），则将其初始化为空字符串。
5. 如果提供了`answer`，则去除首尾空格，并检查是否以句号结尾。如果不是，则添加句号。
6. 最后，函数返回格式化后的`question`、`passages`和`answer`作为一个元组。

get_fewshot

下面是根据您提供的格式对该函数的分析：

## 0.1 功能描述

该函数的主要功能是读取指定数据集的JSON文件，该文件包含问题和答案对，并将其格式化为一个字符串，以便于后续的使用或展示。

## 0.1 参数说明

- `dataset`: 一个字符串，表示要读取的数据集名称。

## 0.1 返回值

该函数没有明确的返回语句，但是它修改了一个全局变量 `fewshot`，该变量包含了格式化后的问题和答案对。因此，该函数的“返回值”实际上是全局变量 `fewshot` 中的内容。

## 0.1 实现逻辑

1. 检查传入的 `dataset` 参数是否以 `_golden` 结尾。如果是，则移除这部分，以获取原始数据集名称。
2. 将处理后的数据集名称赋值给全局变量 `current_dataset`。
3. 使用 `json.load()` 函数读取位于 `fewshot_path` 目录下，名为 `dataset.json` 的文件。这里假设 `fewshot_path` 是一个已经定义好的路径变量。
4. 初始化一个空字符串 `fewshot`。
5. 遍历读取到的JSON对象中的每个元素（即每个问题答案对）。
6. 对于每个问题答案对，将问题和答案格式化为一个字符串，并追加到 `fewshot` 字符串的末尾，每个问题和答案之间用换行符分隔，并在末尾添加两个换行符以分隔不同的问题答案对。

注意：

- 该函数使用了全局变量 `current_dataset` 和 `fewshot`，这在编程实践中通常不推荐，因为全局变量可能导致代码难以理解和维护。
- 函数内部没有处理文件读取错误，如果文件不存在或格式不正确，可能会导致程序崩溃。
- 函数没有返回值，而是通过修改全局变量来传递结果，这限制了函数的可重用性和测试性。

get_prompt

## 0.1 功能描述

该函数 `get_prompt` 的主要功能是构造一个用于对话或问答任务的输入序列，该序列包含了用户提出的问题、相关的段落（如果有的话），以及可能的答案。这个输入序列将被进一步用于训练或测试自然语言处理模型，尤其是那些涉及到上下文理解的模型。

## 0.1 参数说明

- `tokenizer`: 一个分词器对象，用于将文本转换为模型可以理解的输入格式。
- `question`: 用户提出的问题字符串。
- `passages`: 一个包含多个段落的列表，每个段落是一个字符串。这些段落可能包含与问题相关的信息。
- `_get_prompt`: 一个内部函数，用于预处理问题、段落和答案，以便构造最终的输入。
- `answer`: 一个字符串，表示问题的答案。
- `with_cot`: 一个布尔值，指示是否需要包含上下文提示（Context of Thought，COT）。

## 0.1 返回值

返回一个整数列表 `inputs`，这是由分词器处理后的输入序列，可以被模型使用。

## 0.1 实现逻辑

1. 调用内部函数 `_get_prompt` 来预处理问题、段落和答案。
2. 初始化一个空字符串 `contexts`，用于存储所有段落的字符串表示。
3. 如果提供了段落，遍历每个段落，并将它们格式化为带有段落编号的字符串，添加到 `contexts` 中。
4. 根据是否需要包含上下文提示 `with_cot`，选择不同的用户和助手提示模板。
5. 使用这些模板和提供的参数构造用户和助手的内容。
6. 创建一个包含用户内容的消息列表，并将其传递给分词器的 `apply_chat_template` 方法，以生成用户部分的输入序列。
7. 将助手的回答内容编码为整数，并添加到用户输入序列的末尾。
8. 返回最终的整数列表 `inputs`，它可以作为模型的输入。
