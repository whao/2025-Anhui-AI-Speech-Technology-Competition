#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
讯飞（iFlytek）语音听写流式API认证工具
根据Go代码转换而来，用于生成WebSocket连接的认证URL
"""

import hmac
import hashlib
import base64
from urllib.parse import urlparse, urlencode
from datetime import datetime, timezone


def hmac_with_sha_to_base64(algorithm: str, data: str, key: str) -> str:
    """
    使用HMAC-SHA256算法对数据进行签名并返回base64编码的结果
    
    Args:
        algorithm: 算法名称（固定为"hmac-sha256"）
        data: 要签名的数据
        key: 签名密钥（API Secret）
    
    Returns:
        base64编码的签名结果
    """
    if algorithm != "hmac-sha256":
        raise ValueError("Only hmac-sha256 algorithm is supported")
    
    # 使用HMAC-SHA256进行签名
    signature = hmac.new(
        key.encode('utf-8'),
        data.encode('utf-8'),
        hashlib.sha256
    ).digest()
    
    # 返回base64编码的结果
    return base64.b64encode(signature).decode('utf-8')


def assemble_auth_url(host_url: str, api_key: str, api_secret: str) -> str:
    """
    构建讯飞API的认证URL
    
    Args:
        host_url: WebSocket API地址，如 "wss://iat-api.xfyun.cn/v2/iat"
        api_key: API密钥
        api_secret: API密钥对应的Secret
    
    Returns:
        包含认证参数的完整URL
    """
    # 解析URL
    parsed_url = urlparse(host_url)
    
    # 生成RFC1123格式的UTC时间戳
    # 注意：这里使用UTC时间，格式必须是RFC1123
    date = datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S GMT')
    
    # 构建参与签名的字符串
    # 格式：host: {host}\ndate: {date}\n{request_line}
    sign_string_parts = [
        f"host: {parsed_url.hostname}",
        f"date: {date}",
        f"GET {parsed_url.path} HTTP/1.1"
    ]
    
    # 拼接签名字符串（使用换行符连接）
    sign_string = "\n".join(sign_string_parts)
    
    # 使用HMAC-SHA256对签名字符串进行签名
    signature = hmac_with_sha_to_base64("hmac-sha256", sign_string, api_secret)
    
    # 构建authorization原始字符串
    auth_string = (
        f'api_key="{api_key}", '
        f'algorithm="hmac-sha256", '
        f'headers="host date request-line", '
        f'signature="{signature}"'
    )
    
    # 对authorization字符串进行base64编码
    authorization = base64.b64encode(auth_string.encode('utf-8')).decode('utf-8')
    
    # 构建URL参数
    url_params = {
        'host': parsed_url.hostname,
        'date': date,
        'authorization': authorization
    }
    
    # 拼接最终的URL
    call_url = f"{host_url}?{urlencode(url_params)}"
    
    return call_url


# 测试代码和使用示例
if __name__ == "__main__":
    # 你的个人信息
    APPID = "27c57829"
    API_SECRET = "OWYwMjUyMjgzNWY2OGM3N2E0MjI5M2U5"
    API_KEY = "12ba58ebdf50c172e6f03d2bbc830c39"
    
    # 讯飞中英识别大模型API的WebSocket地址 (新版API)
    HOST_URL = "wss://iat.xf-yun.com/v1"
    
    # 生成认证URL
    auth_url = assemble_auth_url(HOST_URL, API_KEY, API_SECRET)
    
    print("生成的认证URL:")
    print(auth_url)
    print("\n" + "="*80)
    
    # 分解URL以便查看各个参数
    parsed = urlparse(auth_url)
    from urllib.parse import parse_qs
    query_params = parse_qs(parsed.query)
    
    print("URL参数解析:")
    print(f"Host: {query_params.get('host', [''])[0]}")
    print(f"Date: {query_params.get('date', [''])[0]}")
    print(f"Authorization (base64): {query_params.get('authorization', [''])[0]}")
    
    # 解码authorization参数查看内容
    try:
        auth_decoded = base64.b64decode(query_params.get('authorization', [''])[0]).decode('utf-8')
        print(f"Authorization (decoded): {auth_decoded}")
    except (ValueError, IndexError) as e:
        print(f"Authorization解码失败: {e}")
