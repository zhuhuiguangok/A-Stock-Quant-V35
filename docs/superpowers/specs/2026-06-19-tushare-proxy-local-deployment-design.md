# Tushare 代理接入与本地部署设计

## 目标

在不重构项目架构的前提下，让项目中所有 Tushare Pro 客户端都通过
`http://lianghua.nanyangqiankun.top` 访问，并在 Windows 本机完成安装、启动和真实数据验证。

## 实现方案

采用逐处修改的最小方案。每个 `ts.pro_api(...)` 调用后立即设置对应客户端的私有 HTTP 地址：

```python
pro._DataApi__http_url = 'http://lianghua.nanyangqiankun.top'
```

对于变量名为 `_pro` 的客户端使用同样赋值方式：

```python
_pro._DataApi__http_url = 'http://lianghua.nanyangqiankun.top'
```

覆盖范围为 `stock_app/views.py` 中四处客户端初始化，以及
`stock_app/risk_factor_builder.py` 中一处客户端初始化。Token 继续沿用项目现有的
`TUSHARE_TOKEN` 环境变量和 Web UI 保存机制，不把 Token 写入受版本控制的源码。

## 测试

先增加静态回归测试，确认每个 `ts.pro_api(...)` 初始化点后均设置代理地址。测试先在修改源码前失败，
再加入代理配置使其通过。

随后执行：

1. Python 依赖安装；
2. Django `check`；
3. 数据库迁移；
4. 使用指定代理请求 `000001.SZ` 日线数据；
5. 本地启动 Django，验证 `http://127.0.0.1:8000/` 可访问。

## 本地配置

项目根目录使用被 `.gitignore` 排除的 `.env` 保存 `TUSHARE_TOKEN`。服务仅绑定本机地址，
不进行公网发布。NLP 模型不是启动和 Tushare 代理验证的必要条件，因此本次不下载约 400MB 的模型。

## 成功标准

- 五处 Tushare Pro 客户端均明确设置代理地址；
- 自动化测试、Django 检查和迁移成功；
- 指定代理可以返回真实日线数据；
- 本地 Django 首页返回成功响应。
