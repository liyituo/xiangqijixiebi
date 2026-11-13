# FlexArm 大模型工具包 (FlexArm LLM Toolkit)

面向工业级 6 轴机械臂（myCobot Pro 450）与智能自适应电动夹爪的 **大模型 Function Calling 专用工具包**。
支持标准 OpenAI、Google Gemini、Anthropic Claude、LangChain 及 AutoGen 框架无缝接入。

---

## 🌟 核心特性

1. **8 大原子工具全覆盖**：状态感知、5重安全审查、11步动作规划、空中检查点挂起、最小代价改道重规划、数学代价推演、直接力控夹爪、最高优先级急停。
2. **三层物理安全防火墙**：
   - 规划层：工件与工位白名单、工作空间限位约束、无碰撞干涉；
   - 执行层：空中安全检查点 (ACT-06) 悬停挂起；
   - 控制层：大角度自动分步平滑插补 ($\le 20^\circ$)，防蠕动硬门禁（速度 $<5$ 强制提升至 $10$）。
3. **真实/仿真双轨运行**：支持 TCP Socket (`192.168.0.232:4500`) 驱动真机，离线或无设备时无缝回退至仿真推演。
4. **零依赖启发式 ReAct 引擎**：无需配置 API Key 即可本地跑通智能体自主推理闭环。

---

## 📦 工具列表 (Tool Schemas)

| 工具名称 | 功能描述 | 核心输入参数 |
| :--- | :--- | :--- |
| `flexarm_get_status` | 获取各关节角度、末端坐标、夹爪力控反馈、工位状态 | 无 |
| `flexarm_safety_check` | 5 重物理硬约束安全审查（拦截未知工位、工件、越界、冲突） | `item_id`, `source_station`, `target_station` |
| `flexarm_plan_task` | 生成带空中安全检查点 (ACT-06) 的 11 步无视觉动作规划 | `item_id`, `source_station`, `target_station`, `speed` |
| `flexarm_execute_plan` | 驱动执行动作序列，支持检查点悬停挂起或预置改道 | `interactive_checkpoint`, `auto_replan_target` |
| `flexarm_replan_destination` | **【核心重规划】** 工件悬空时热替换目标工位，以标准航速送达 | `new_target_station`, `speed` |
| `flexarm_compare_replan_costs`| 数学加权代价评估（Baseline vs FlexArm 补丁，降低 80.7% 代价）| `new_target_station` |
| `flexarm_control_gripper` | 底层直接驱动电动夹爪开闭，读取力控代码 | `action` ('OPEN'/'CLOSE'), `speed` |
| `flexarm_emergency_stop` | **【最高优先级】** 立即紧急制动停止机械臂并断开网络 | `reason` |

---

## 🚀 快速上手 (Quickstart)

### 1. 基础 Python 调用

```python
from flexarm_toolkit import FlexArmToolkit

# 初始化工具包 (simulation=False 为真实机械臂，True 为仿真模式)
toolkit = FlexArmToolkit(simulation=False)

# 1. 查询状态
status = toolkit.execute_tool("flexarm_get_status", {})
print("当前关节角度:", status["joint_angles_deg"])

# 2. 安全审查
safe_res = toolkit.execute_tool("flexarm_safety_check", {
    "item_id": "Object_A",
    "source_station": "P1",
    "target_station": "P3"
})
print("安全审查是否通过:", safe_res["safety_passed"])

# 3. 生成 11 步搬运任务 (含 ACT-06 检查点)
plan_res = toolkit.execute_tool("flexarm_plan_task", {
    "item_id": "Object_A",
    "source_station": "P1",
    "target_station": "P3",
    "speed": 10
})

# 4. 在空中检查点动态改道至 P4 (执行感知与最小代价重规划)
replan_res = toolkit.execute_tool("flexarm_replan_destination", {
    "new_target_station": "P4",
    "speed": 10
})

# 5. 执行规划闭环
exec_res = toolkit.execute_tool("flexarm_execute_plan", {
    "interactive_checkpoint": False,
    "auto_replan_target": "P4"
})
print("任务执行完成:", exec_res["success"])
```

---

### 2. 接入 OpenAI 官方 SDK (Function Calling)

```python
from openai import OpenAI
from flexarm_toolkit import FlexArmToolkit, TOOL_SCHEMAS, SYSTEM_PROMPT
import json

client = OpenAI(api_key="your-api-key")
toolkit = FlexArmToolkit(simulation=False)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "把工件A从P1搬运到P3，但到达检查点后改放到P4工位"}
]

# 发起带工具声明的请求
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=TOOL_SCHEMAS,
    tool_choice="auto"
)

# 处理大模型工具调用
msg = response.choices[0].message
if msg.tool_calls:
    for tool_call in msg.tool_calls:
        fn_name = tool_call.function.name
        fn_args = json.loads(tool_call.function.arguments)
        
        # 统一调度底层机械臂物理工具
        tool_result = toolkit.execute_tool(fn_name, fn_args)
        
        # 将物理执行结果反馈给大模型
        messages.append(msg)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(tool_result, ensure_ascii=False)
        })

# 最终大模型给出物理闭环总结回复
final_resp = client.chat.completions.create(model="gpt-4o", messages=messages)
print(final_resp.choices[0].message.content)
```

---

### 3. 使用内置自主 ReAct 智能体

若本地没有配置 API Key，可直接运行自主闭环智能体：

```python
from flexarm_toolkit import FlexArmAgent

agent = FlexArmAgent(simulation=False)
result = agent.chat_and_execute("把物体A从工位P1搬运到P3，在半空检查点改放到工位P4")
print("Agent 报告:", result["final_answer"])
```

---

## 🧪 自动化测试套件

在项目根目录下执行单元测试：

```powershell
python test_toolkit.py
```

9 项测试用例 100% 通过（涵盖 Schema 校验、状态读取、5重拦截、任务规划、代价对比、改道重规划与夹爪控制）。
