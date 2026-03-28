---
description: "SVG-based visual content generator — PPT, posters, social media images. Powered by ppt-master engine. Use when the user wants to create presentations, posters, social media graphics (WeChat Moments, Xiaohongshu, Stories, Banners, etc.)."
---

# Poster — SVG 视觉内容生成器

## User Onboarding (IMPORTANT)

**When the user provides NO content or arguments (just `/poster`), or asks for help, you MUST present the following introduction instead of starting the workflow:**

Present this to the user:

---

**Poster** - PPT / 海报 / 社交图片生成器

基于 ppt-master 引擎，支持多种格式：

| 格式 | 代码 | 尺寸 | 适用场景 |
|------|------|------|----------|
| PPT 16:9 | `ppt169` | 1280x720 | 演示汇报、培训课件 |
| PPT 4:3 | `ppt43` | 1024x768 | 传统投影 |
| 朋友圈海报 | `moments` | 1080x1080 | 朋友圈、Instagram |
| 小红书 | `xhs` | 1242x1660 | 小红书图文、知识分享 |
| 竖版 Story | `story` | 1080x1920 | 抖音/Instagram Story |
| 公众号头图 | `wechat` | 900x383 | 微信公众号文章配图 |
| 横版 Banner | `banner` | 1920x1080 | 网页横幅、大屏展示 |
| A4 打印 | `a4` | 1240x1754 | 打印海报、宣传单页 |

**使用示例：**

- `/poster 做一张朋友圈海报：香港储蓄险的5个优势 --format moments`
- `/poster 把这篇文章做成PPT：report.pdf`
- `/poster 做一组小红书图文：保费融资 --format xhs`
- `/poster https://example.com/article --format ppt169`

也可以直接描述你想做什么，我会帮你选择合适的格式。

**需要做讲解视频？** 请使用 `/xingbao`。

---

**After presenting the introduction, wait for the user to provide their content/request. Do NOT proceed with any workflow until the user gives specific content.**

## Execution

When the user provides content, invoke the **poster** skill. The full workflow is defined in the skill — do NOT duplicate it here.
