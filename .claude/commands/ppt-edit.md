---
description: "Consume a PPT Master editor local AI handoff bundle, update slide_state.json in-place, then render compatible SVG output."
---

# PPT Edit — 编辑器本地 AI Handoff 执行器

## User Onboarding (IMPORTANT)

当用户没有提供项目路径或请求文件路径，只输入 `/ppt-edit` 或询问帮助时，先展示以下说明：

---

**PPT Edit** - 消费 PPT Master Editor 导出的本地 AI handoff bundle

适用输入：

- `/ppt-edit /绝对路径/到/项目目录`
- `/ppt-edit /绝对路径/到/项目目录/design_patch.ai-request.json`
- `/ppt-edit /绝对路径/到/项目目录/.cache/ai_handoff/design_patch.ai-request.json`
- `/ppt-edit /绝对路径/到/项目目录/design_patch.json`（仅当该文件仍包含 `aiCommand` 时）

执行内容：

1. 读取 `slide_state.json`
2. 读取编辑器导出的本地 AI handoff 文件
3. 修改 `slide_state.json`
4. 运行 `python3 tools/slide_state_bridge.py render <项目路径>`
5. 验证项目结构

注意：这是给 Claude Code 本地 command / skill 使用的工作流。浏览器编辑器只负责导出文件；仓库本身不提供浏览器直连或服务端模型 API。
若编辑器已绑定项目，默认生成位置是 `<项目>/.cache/ai_handoff/`。

---

在用户给出具体路径前，不要继续执行。

## Execution

当用户给出项目路径或 handoff 文件路径时，调用 `ppt-edit` skill，按 skill 中定义的本地流程执行。不要在这里重复整个工作流。
