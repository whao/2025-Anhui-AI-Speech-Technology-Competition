# 讯飞API认证URL生成 - Python实现

这是根据提供的Go代码转换而来的Python实现，用于生成讯飞（iFlytek）语音听写流式API的认证URL。

## 🔑 你的个人信息

```
APPID: 27c57829
APISecret: OWYwMjUyMjgzNWY2OGM3N2E0MjI5M2U5
APIKey: 12ba58ebdf50c172e6f03d2bbc830c39
```

## 📁 生成的文件

### 1. `xunfei_auth_simple.py` - 简单版本
直接对应Go代码的Python实现，只包含URL生成功能。

**主要函数：**
- `hmac_with_sha_to_base64()` - 对应Go的`HmacWithShaTobase64`函数
- `assemble_auth_url()` - 对应Go的`assembleAuthUrl`函数

### 2. `xunfei_asr_client.py` - 完整版本  
包含完整的WebSocket客户端实现，可以实际进行语音识别。

**主要功能：**
- URL认证生成
- WebSocket连接
- 音频数据发送
- 识别结果接收和解析

### 3. `xunfei_auth.py` - 基础版本
最初的实现版本，功能与简单版本相似。

## 🚀 使用方法

### 简单使用 - 只生成认证URL

```python
from xunfei_auth_simple import assemble_auth_url

# 你的信息
API_KEY = "12ba58ebdf50c172e6f03d2bbc830c39"
API_SECRET = "OWYwMjUyMjgzNWY2OGM3N2E0MjI5M2U5"
HOST_URL = "wss://iat-api.xfyun.cn/v2/iat"

# 生成认证URL
auth_url = assemble_auth_url(HOST_URL, API_KEY, API_SECRET)
print(auth_url)
```

### 完整使用 - WebSocket连接

```python
import asyncio
from xunfei_asr_client import XunfeiASR

async def main():
    # 创建客户端
    client = XunfeiASR(
        app_id="27c57829",
        api_key="12ba58ebdf50c172e6f03d2bbc830c39", 
        api_secret="OWYwMjUyMjgzNWF2OGM3N2E0MjI5M2U5"
    )
    
    # 生成认证URL
    auth_url = client.generate_auth_url()
    print(f"认证URL: {auth_url}")
    
    # 如果有PCM音频文件，可以进行识别
    # results = await client.recognize_audio_file("audio.pcm")
    # text = client.parse_results(results)
    # print(f"识别结果: {text}")

# 运行
asyncio.run(main())
```

## 🔧 Go代码对应关系

| Go函数/变量 | Python对应 | 说明 |
|------------|------------|------|
| `assembleAuthUrl()` | `assemble_auth_url()` | 主要的URL生成函数 |
| `HmacWithShaTobase64()` | `hmac_with_sha_to_base64()` | HMAC-SHA256签名函数 |
| `url.Parse(hosturl)` | `urlparse(hosturl)` | URL解析 |
| `time.Now().UTC().Format(time.RFC1123)` | `datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')` | RFC1123时间格式 |
| `strings.Join(signString, "\n")` | `"\n".join(sign_string_parts)` | 字符串拼接 |
| `base64.StdEncoding.EncodeToString()` | `base64.b64encode().decode('utf-8')` | Base64编码 |
| `url.Values{}` + `v.Encode()` | `urlencode()` | URL参数编码 |

## 🌐 API文档参考

- **官方文档**: https://www.xfyun.cn/doc/asr/voicedictation/API.html
- **WebSocket地址**: `wss://iat-api.xfyun.cn/v2/iat`
- **认证方式**: HMAC-SHA256签名
- **支持格式**: PCM、SPEEX、MP3（部分支持）

## ⚠️ 注意事项

1. **时间同步**: 服务器会检查时钟偏移，最大允许300秒偏差
2. **音频格式**: 
   - PCM: 16k/8k采样率，16bit，单声道
   - 每帧建议1280字节（40ms音频）
3. **会话时长**: 最长60秒
4. **并发限制**: 默认50路并发

## 🧪 测试结果

运行 `python3 xunfei_auth_simple.py` 会输出：

```
============================================================
讯飞API认证URL生成结果
============================================================
输入参数:
  APPID: 27c57829
  API_KEY: 12ba58ebdf50c172e6f03d2bbc830c39
  API_SECRET: OWYwMjUyMjgzNWY2OGM3N2E0MjI5M2U5
  HOST_URL: wss://iat-api.xfyun.cn/v2/iat

生成的认证URL:
wss://iat-api.xfyun.cn/v2/iat?host=iat-api.xfyun.cn&date=...&authorization=...

URL参数详情:
  Host: iat-api.xfyun.cn
  Date: Sun, 27 Jul 2025 14:26:53 GMT
  Authorization (base64): ...
  Authorization (decoded): api_key="...", algorithm="hmac-sha256", headers="host date request-line", signature="..."
============================================================
```

## 📦 依赖包

标准库即可，无需额外安装：
- `hmac`, `hashlib` - 签名计算
- `base64` - Base64编码  
- `urllib.parse` - URL处理
- `datetime` - 时间处理
- `websockets` - WebSocket连接（仅完整版本需要）

如需WebSocket功能，安装：
```bash
pip install websockets
```
