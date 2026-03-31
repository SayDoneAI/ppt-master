---
name: PPT Master Preview
description: 当需要在本地预览 PPT Master 的模板库、示例库、Gallery 页面或查看正确的预览入口时使用。优先启动仓库根目录 HTTP 服务，并以 `gallery.html` 作为总览入口；需要逐页浏览示例时再使用 `viewer.html`，需要查看单个项目最终页时再进入 `<项目路径>/svg_final/`。
---

# PPT Master 本地预览

本技能用于统一 PPT Master 仓库的本地预览入口，避免只打开单个 `svg_final/` 目录而错过仓库自带的 Gallery 和示例浏览页。

## 使用顺序

1. **总览模板与示例时，优先预览仓库根目录**

   ```bash
   python3 -m http.server -d . 8000
   ```

   默认入口：

   - `http://localhost:8000/gallery.html` — 模板与案例总览，**首选**
   - `http://localhost:8000/viewer.html` — 示例逐页浏览器
   - `http://localhost:8000/` — 首页卡片入口

2. **如果模板库或案例索引刚更新，先重新生成 Gallery**

   ```bash
   python3 tools/template_gallery.py --port 8000
   ```

   该命令会生成/刷新仓库根目录的 `gallery.html`，并直接启动预览服务。

3. **只检查单个项目最终页时，再预览该项目目录**

   ```bash
   python3 -m http.server -d <项目路径>/svg_final 8000
   ```

## 排障提示

- `gallery.html` 中部分卡片显示“无预览”是正常情况，表示该案例本身没有绑定预览图。
- `gallery.html` 中部分图片来自外部链接（如 `raw.githubusercontent.com`）；如果网络受限，可能加载失败，但这不代表仓库本地示例文件丢失。
- `svg_output/` 中的 SVG 可能仍引用相对路径图片；用于最终展示时优先检查 `svg_final/`，因为后处理会嵌入图片和图标。
