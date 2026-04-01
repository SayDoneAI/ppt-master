# Task Plan: 直接编辑 SVG + 预设系统

## Goal
将编辑器从"slide_state 渲染 SVG"改为"直接加载 AI 生成的真实 SVG"，并加入面向普通人的预设系统（配色/字体一键切换），让不懂设计的人也能做出专业效果。

## 核心转向
```
之前: slide_state.json → slideToSvg() → 浏览器渲染（质量低）
之后: AI 生成高质量 SVG → 编辑器直接加载 → 用户可视化微调 + 预设一键换风格
```

## 为什么 slide_state 渲染不行
- slide_state 是低级图元堆叠（rect + text + path），没有布局语义
- AI 直接写 SVG 时有完整的视觉设计能力（卡片、多栏、间距、层次），slide_state 丢失了这些
- Pretext 只管"一个文本框内断行"，不管元素间关系
- 用户打开编辑器看到的效果远不如 AI 原始产出

## Current Phase
Phase 4（AI 回合闭环待继续；本轮已完成研究型配色收口、颜色条按钮和字体按钮溢出修复，editor test/build 已重新恢复绿色）

## Phases

### Phase 1: 编辑器改为直接加载 SVG
- [x] Step 1/5：raw SVG 直接加载渲染
  - `editor/src/app.ts` 新增 `rawSvgStrings: string[]`，SVG 文件 / `?svg=` / 默认 demo 真实 SVG 都会保留原始字符串
  - `renderCanvas()` / `renderThumbnails()` 优先渲染 raw SVG，缺失时才 fallback `slideToSvg()`
  - raw SVG 在进入编辑器前统一标准化：补 `data-element-id`，并对 demo 的相对图片路径做最小修正
  - 当前画布导出改为导出 live DOM 的真实 SVG，而不是从 `SlideState` 重新生成
  - `editor/vite.config.ts` 增加 `/examples/...` 静态暴露，默认 demo 在 dev/build/preview 下都可访问
- [ ] 新建 `editor/src/svg_editor.ts`，直接操作 SVG DOM（替代 slideToSvg 渲染路径）
  - 加载方式：`innerHTML` 注入 SVG 到画布，或 `<object>` / inline SVG
  - 保留元素交互：hover 高亮、点击选中、双击编辑文字
  - 保留拖拽移动
  - 直接操作 SVG DOM 属性（不再经过 slide_state 转换）
- [ ] 修改 `editor/src/app.ts` 入口：
  - 启动时如果 `?svg=` 或拖入 SVG 文件，走 SVG 直接编辑路径
  - 默认用一个真实示例 SVG 做 demo（从 examples/ 取）
  - slide_state 路径保留但降级为兼容模式
- [x] 属性面板改为普通人友好：
  - text 元素：「文字内容」「字号」「颜色」「粗细」
  - rect 元素：「颜色」「圆角」「边框」
  - image 元素：「替换图片」
  - 隐藏技术细节（x/y 坐标只在"高级"折叠中显示）
- [x] SVG 保存/导出：修改后的 SVG 直接下载（`outerHTML` 序列化）
- [x] 缩略图生成：多页 SVG 时，每页独立缩略图

### Phase 2: 预设系统（配色一键切换）
- [ ] 定义 SVG 语义标记协议：
  ```
  data-color-role="primary"     → 主色（标题、强调）
  data-color-role="secondary"   → 辅色（副标题、图标底色）
  data-color-role="accent"      → 强调色（CTA 按钮、高亮）
  data-color-role="text-dark"   → 深色文字
  data-color-role="text-light"  → 浅色文字（暗色背景上）
  data-color-role="text-muted"  → 次要文字
  data-color-role="background"  → 背景色
  data-color-role="background-alt" → 交替背景
  ```
- [x] 收敛为 16 套更通用的 curated 配色方案到 `editor/src/presets/colors.ts`
- [x] 配色预设面板 UI：色卡网格，点击即生效
  - 实现：遍历 SVG DOM 中带 `data-color-role` 的元素，替换 fill/stroke
  - 预览：hover 色卡时实时预览效果
  - 当前按钮形态：不显示文字，只显示 3 条颜色条；名称通过 tooltip / aria-label 暴露
- [x] 更新 Executor 角色定义：AI 生成 SVG 时必须打 `data-color-role` 标记
- [x] 兼容无标记 SVG：通过颜色聚类自动推断角色（fallback）

### Phase 3: 字体预设 + 更多普通人友好交互
- [x] 字体预设方案已切到 6 套免费商用字体：
  - Noto Sans / 思源黑体 / 思源宋体 / 阿里巴巴普惠体 / OPPO Sans / HarmonyOS Sans
- [x] SVG 字体标记：`data-font-role="title" | "body" | "caption" | "label"`（编辑器已消费）
- [x] 颜色选择器组件（替代 HEX 输入）
- [x] 字号 slider（替代数字输入）
- [x] 工具栏导出补丁：
  - 新增“导出 PNG”和“保存模板”，并复用当前 SVG 直出链路
- [x] 字体按钮溢出修复：
  - 当前侧栏宽度下改为更紧凑的 2 列字体网格，长名字走省略号并保留 tooltip
- [x] 默认 demo 清理：
  - `slide_01_cover.svg` 已移除 GitHub / MIT 标注
- [x] 文本 resize 语义收敛：
  - 文本 `e / w` 手柄只改文本框宽度与 x 锚点，不再同步放大字号
  - 文本仅保留四角 + 左右手柄，移除纯 `n / s`
  - 拖拽中的 live preview 会按最新 `width` 实时重排文本
- [x] 顶部比例控件收敛为海报画幅入口：
  - 显示条件从“单页即可”改为“单页且当前画布命中 `square/poster/story` 海报预设”
  - 单页 PPT `1280x720` / `1024x768` 与多页场景不再显示
  - 现有 `resizeSlideStateCanvas()` 缩放逻辑保持不变
- [ ] 元素对齐工具（居中、左对齐等）

### Phase 4: AI 回合闭环
- [ ] 用户修改后 → 导出当前 SVG + 修改描述 → 交给 AI skill
- [ ] AI 返回新 SVG → 编辑器自动刷新
- [ ] 保留 design_patch 作为"告诉 AI 改了什么"的辅助格式

## Key Decisions
| Decision | Rationale |
|----------|-----------|
| SVG DOM 作为真相源 | AI 生成的 SVG 质量已经很高，不需要中间层 |
| data-color-role 语义标记 | 让编辑器能按角色批量换配色，不需要理解每个元素 |
| 属性面板用人话 | "字号" vs "font-size"、"颜色" vs "fill"，降低认知门槛 |
| 收敛为 16 套 curated 配色 | 比旧的 19 套 follow-up spec 更贴近普通用户认可的常用风格，同时让 UI 更容易收纳 |
| 海报比例控件只在海报预设显示 | 避免单页 PPT 误出现 `1:1 / 4:5 / 9:16`，把用户心智收敛成“海报画幅”入口 |
| slide_state 降级不废弃 | AI 回传场景仍可用，保持兼容 |

## 普通人的使用流程
```
1. 告诉 AI "做一张活动海报" → AI skill 生成 SVG
2. 编辑器自动打开，看到高质量预览
3. 想改文字 → 双击文字直接改
4. 想改颜色 → 右侧面板选预设配色方案（如"科技蓝"、"金融深蓝"）
5. 想微调位置 → 拖拽移动
6. 满意 → 点"导出"下载 SVG / PPTX
7. 不满意 → 告诉 AI 要改什么 → AI 返回新版本 → 自动刷新
```

## 从现有代码中保留的能力
- hover 高亮 + 点击选中 + overlay 控制手柄（改为操作 SVG DOM）
- 双击文字编辑（foreignObject + textarea 方案不变）
- Pretext 文本重排（只在修改文字后触发，而非初次渲染）
- 缩略图条 + 多页导航
- 拖拽移动
- dev server + HMR 监听

## 从现有代码中废弃/降级的能力
- slideToSvg() 渲染路径 → 降级为兼容模式
- slide_state 作为真相源 → 降级为 AI 传输格式
- createDemoState() 硬编码 demo → 替换为加载真实 SVG
- 属性面板的开发者字段（x/y/width/d 等）→ 折叠到"高级"

## Constraints
- finalize_svg.py 不删 data-* 属性（已验证）
- svg_to_pptx.py 也不操作 data-* 属性（已验证）
- 现有后处理链路（finalize → export）保持不变
