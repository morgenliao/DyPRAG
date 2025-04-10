import os
import tempfile
# from zhipuai import ZhipuAI
from openai import OpenAI
from tqdm import tqdm
import ast
import concurrent.futures
import subprocess
import platform

# client = ZhipuAI(api_key="18dfe3a4c78afcbcda4045ac12483c9a.WoUQ7ibrdvRqXDUN")
client = OpenAI(api_key="sk-dceyienvrazgavqubkwwxtlmsbseqlgmswqlowjfbbiyeenl", 
                base_url="https://api.siliconflow.cn/v1")


def split_code(code):
    """
    将代码拆分为类和函数块。

    参数:
        code (str): 要解析的Python代码。

    返回:
        dict: 包含类和函数块的字典，格式为{'class': [(类名, 类代码)], 'function': [(函数名, 函数代码)]}。
    """
    tree = ast.parse(code)
    blocks = {'class': [], 'function': []}

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            blocks['class'].append((node.name, ast.unparse(node)))
        elif isinstance(node, ast.FunctionDef):
            blocks['function'].append((node.name, ast.unparse(node)))

    return blocks

def analyze_python_code(code):
    """
    分析Python代码并生成Markdown格式的代码说明。

    参数:
        code (str): 要分析的Python代码。

    返回:
        str: 分析结果的Markdown格式字符串。
    """
    prompt = """生成以下格式的代码分析：
# 代码说明    
## 主要功能
[描述代码的主要用途]
## 架构说明
[描述代码的整体架构]
## 关键组件
[列出主要的类和函数]"""

    response = client.chat.completions.create(
        # model="glm-4",
        # model="Qwen/Qwen2.5-72B-Instruct",
        model="Pro/deepseek-ai/DeepSeek-V3",
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "好的，我会按照指定格式分析代码。"},
            {"role": "user", "content": code}
        ],
    )
    return response.choices[0].message.content

def analyze_python_block(block, name, block_type):
    """
    分析Python代码块（类或函数）并生成Markdown格式的说明。

    参数:
        block (str): 要分析的代码块。
        name (str): 代码块的名称（类名或函数名）。
        block_type (str): 代码块的类型（类或函数）。

    返回:
        str: 分析结果的Markdown格式字符串。
    """
    prompt = f"""生成以下格式的{block_type}分析：
## 功能描述
[描述{block_type}的主要功能]
## 参数说明
[描述参数及其用途]
## 返回值
[描述返回值类型和含义]
## 实现逻辑
[描述关键实现逻辑]"""

    response = client.chat.completions.create(
        # model="glm-4",
        # model="Qwen/Qwen2.5-72B-Instruct",
        model="Pro/deepseek-ai/DeepSeek-V3",
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"好的，我会按照指定格式分析这个{block_type}。"},
            {"role": "user", "content": block}
        ],
    )
    return f"{name}\n\n" + response.choices[0].message.content

def format_markdown(file_path):
    """
    使用markdownlint格式化Markdown文件。

    参数:
        file_path (str): 要格式化的Markdown文件路径。
    """
    try:
        # Windows 系统使用 npm 全局安装的 markdownlint-cli
        if platform.system() == 'Windows':
            cmd = ['npx', 'markdownlint', '-f', file_path]
        else:
            cmd = ['markdownlint', '-f', file_path]
            
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',  # 指定编码为 UTF-8
            shell=True if platform.system() == 'Windows' else False,
            env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}  # 设置 Python IO 编码
        )
        print(f"已完成 {file_path} 的格式化")
    except subprocess.CalledProcessError as e:
        print(f"格式化 {file_path} 时发生错误，但将继续处理")
    except FileNotFoundError:
        print("未找到 markdownlint，请确保已通过 npm 安装 markdownlint-cli：")
        print("npm install -g markdownlint-cli")
    except Exception as e:
        print(f"格式化过程中出现其他错误：{str(e)}，但将继续处理")

def analyze_code(code, language):
    """
    分析代码并生成Markdown格式的代码说明。

    参数:
        code (str): 要分析的代码。
        language (str): 代码的编程语言。

    返回:
        str: 分析结果的Markdown格式字符串。
    """
    prompt = f"""生成以下格式的{language}代码分析：
# 代码说明    
## 主要功能
[描述代码的主要用途]
## 架构说明
[描述代码的整体架构]
## 关键组件
[列出主要的类和函数]"""

    response = client.chat.completions.create(
        # model="glm-4",
        # model="Qwen/Qwen2.5-72B-Instruct",
        model="Pro/deepseek-ai/DeepSeek-V3",
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"好的，我会按照指定格式分析{language}代码。"},
            {"role": "user", "content": code}
        ],
    )
    return response.choices[0].message.content

def analyze_block(block, name, block_type, language):
    """
    分析代码块并生成Markdown格式的说明。

    参数:
        block (str): 要分析的代码块。
        name (str): 代码块的名称。
        block_type (str): 代码块的类型。
        language (str): 代码的编程语言。

    返回:
        str: 分析结果的Markdown格式字符串。
    """
    prompt = f"""生成以下格式的{language}{block_type}分析：
                    ### 功能描述
                    [描述{block_type}的主要功能]
                    ### 参数说明
                    [描述参数及其用途]
                    ### 返回值
                    [描述返回值类型和含义]
                    ### 实现逻辑
                    [描述关键实现逻辑]"""

    response = client.chat.completions.create(
        # model="glm-4",
        # model="Qwen/Qwen2.5-72B-Instruct",
        model="Pro/deepseek-ai/DeepSeek-V3",
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"好的，我会按照指定格式分析这个{language}{block_type}。"},
            {"role": "user", "content": block}
        ],
    )
    return f"## {name}\n\n" + response.choices[0].message.content


def add_header_numbers(content):
    """
    为Markdown标题添加序号。

    参数:
        content (str): Markdown内容。

    返回:
        str: 添加序号后的Markdown内容。
    """
    lines = content.split('\n')
    counters = [0] * 6  # h1-h6 的计数器
    numbered_lines = []
    file_title_processed = False  # 标记是否已处理过文件标题

    for line in lines:
        if line.strip().startswith('#'):
            # 计算标题级别
            level = len(line.split()[0])
            title_text = line.split(' ', 1)[1] if len(line.split()) > 1 else ""
            
            # 跳过文件标题（第一个出现的一级标题）
            if level == 1 and not file_title_processed:
                numbered_lines.append(line)
                file_title_processed = True
                continue
            
            # 对于不同层级的标题处理
            if level == 1:  # 一级标题（代码说明、类分析、函数分析）
                counters[0] += 1
                for i in range(1, 6):
                    counters[i] = 0  # 重置所有低级别计数器
                numbered_lines.append(f"# {counters[0]}. {title_text}")
            elif level == 2:  # 二级标题
                counters[1] += 1
                for i in range(2, 6):
                    counters[i] = 0  # 重置所有低级别计数器
                numbered_lines.append(f"## {counters[0]}.{counters[1]} {title_text}")
            elif level == 3:  # 三级标题
                counters[2] += 1
                for i in range(3, 6):
                    counters[i] = 0  # 重置所有低级别计数器
                numbered_lines.append(f"### {counters[0]}.{counters[1]}.{counters[2]} {title_text}")
            elif level == 4:  # 四级标题
                counters[3] += 1
                numbered_lines.append(f"#### {counters[0]}.{counters[1]}.{counters[2]}.{counters[3]} {title_text}")
            elif level == 5:  # 五级标题
                counters[4] += 1
                numbered_lines.append(f"##### {counters[0]}.{counters[1]}.{counters[2]}.{counters[3]}.{counters[4]} {title_text}")
            else:  # 六级标题
                counters[5] += 1
                numbered_lines.append(f"###### {counters[0]}.{counters[1]}.{counters[2]}.{counters[3]}.{counters[4]}.{counters[5]} {title_text}")
        else:
            numbered_lines.append(line)

    return '\n'.join(numbered_lines)


def process_python_file(file_path):
    """
    分析Python文件并生成Markdown格式的分析结果。

    参数:
        file_path (str): 要分析的Python文件路径。
    """
    try:
        markdown_file_path = os.path.splitext(file_path)[0] + '.md'
        if (os.path.exists(markdown_file_path)):
            print(f"\n文件 {file_path} 已经分析过，跳过。")
            return

        print(f"\n正在分析文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        file_name = os.path.basename(file_path)
        doc_title = f"# {file_name}\n\n"
        
        overall_analysis = analyze_python_code(content)
        
        blocks = split_code(content)
        
        class_analyses = []
        if (blocks['class']):
            class_analyses.append("# 类分析\n\n以下是代码中定义的类的详细分析：\n")
            for class_name, class_code in blocks['class']:
                analysis = analyze_python_block(class_code, class_name, "类")
                class_analyses.append(analysis)
        
        function_analyses = []
        if (blocks['function']):
            function_analyses.append("# 函数分析\n\n以下是代码中定义的函数的详细分析：\n")
            for func_name, func_code in blocks['function']:
                analysis = analyze_python_block(func_code, func_name, "函数")
                function_analyses.append(analysis)

        sections = [doc_title, overall_analysis]
        if (class_analyses):
            sections.extend(class_analyses)
        if (function_analyses):
            sections.extend(function_analyses)
        
        full_analysis = "\n\n".join(sections)
        
        # 添加标题序号
        numbered_analysis = add_header_numbers(full_analysis)

        # 保存文件时指定 UTF-8 编码
        with open(markdown_file_path, 'w', encoding='utf-8') as file:
            file.write(numbered_analysis)

        try:
            format_markdown(markdown_file_path)
        except Exception as e:
            print(f"格式化失败，但文件已保存：{str(e)}")

        print(f"分析结果已保存为: {markdown_file_path}")
    except Exception as e:
        print(f"处理文件 {file_path} 时发生错误: {str(e)}")


def get_file_type_choice():
    """
    获取用户选择的文件类型。

    返回:
        list: 用户选择的文件扩展名列表。
    """
    file_types = {
        '1': ('py', 'Python'),
        '2': ('js', 'JavaScript'),
        '3': ('ts', 'TypeScript'),
        '4': ('rs', 'Rust'),
        '5': ('cpp', 'C++'),
        '6': ('c', 'C'),
        '7': ('go', 'Go')
    }
    
    print("\n请选择要分析的代码类型：")
    for key, (ext, name) in file_types.items():
        print(f"{key}. {name} (.{ext})")
    
    while True:
        choice = input("\n请输入数字(1-7)，多个类型用逗号分隔（如：1,2,3）: ").strip()
        try:
            selections = [s.strip() for s in choice.split(',')]
            extensions = []
            for s in selections:
                if s in file_types:
                    extensions.append(file_types[s][0])
                else:
                    raise ValueError
            return extensions
        except ValueError:
            print("输入无效，请重新输入！")

def find_code_files(directory, extensions, max_depth=3):
    """
    查找指定扩展名的代码文件。

    参数:
        directory (str): 要搜索的目录。
        extensions (list): 要搜索的文件扩展名列表。
        max_depth (int): 搜索的最大目录深度。

    返回:
        list: 找到的代码文件路径列表。
    """
    code_files = []
    for root, dirs, files in os.walk(directory):
        if root[len(directory):].count(os.sep) < max_depth:
            for file in files:
                if any(file.endswith(f'.{ext}') for ext in extensions) and \
                   file != "trans_code.py" and file != "annot_py_batch.py":
                    code_files.append(os.path.join(root, file))
    return code_files

def get_language_prompt(file_ext):
    """
    根据文件扩展名返回相应的语言提示。

    参数:
        file_ext (str): 文件扩展名。

    返回:
        str: 对应的语言名称。
    """
    prompts = {
        'py': "Python",
        'js': "JavaScript",
        'ts': "TypeScript",
        'rs': "Rust",
        'cpp': "C++",
        'c': "C",
        'go': "Go"
    }
    return prompts.get(file_ext, "Unknown")

# 修改主程序
if __name__ == "__main__":
    # 获取用户选择的文件类型
    selected_extensions = get_file_type_choice()
    
    # 获取当前工作目录
    current_directory = os.getcwd()
    
    # 查找所有指定类型的代码文件
    code_files = find_code_files(current_directory, selected_extensions)
    total_files = len(code_files)

    if total_files == 0:
        print(f"\n当前目录下未找到任何{', '.join(selected_extensions)}文件！")
    else:
        print(f"\n找到 {total_files} 个代码文件")
        
        # 使用线程池处理文件
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {executor.submit(process_python_file, file_path): file_path 
                            for file_path in code_files}
            
            with tqdm(total=total_files, desc="总体进度") as pbar:
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        future.result()
                    except Exception as exc:
                        print(f'{file_path} 生成过程中产生了异常: {exc}')
                    finally:
                        pbar.update(1)

        print("\n所有代码文件都已处理完毕。")