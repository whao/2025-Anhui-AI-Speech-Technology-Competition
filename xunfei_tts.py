#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
讯飞超拟人语音合成API客户端
基于WebSocket实现流式TTS功能
"""

import json
import base64
import websocket
import threading
import time
from dotenv import load_dotenv
import os
from xunfei_auth import assemble_auth_url


class XunfeiTTS:
    def __init__(self, app_id: str, api_key: str, api_secret: str):
        """
        初始化讯飞TTS客户端
        
        Args:
            app_id: 应用ID
            api_key: API密钥
            api_secret: API密钥Secret
        """
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        
        # 超拟人TTS API地址
        self.host_url = "wss://cbm01.cn-huabei-1.xf-yun.com/v1/private/mcd9m97e6"
        
        # 音频数据存储
        self.audio_data = []
        
        # WebSocket连接
        self.ws = None
        
        # 连接状态
        self.connected = False
        
    def create_request_data(self, text: str, voice: str = "x5_lingfeiyi_flow", 
                          audio_encoding: str = "lame", sample_rate: int = 24000) -> dict:
        """
        创建TTS请求数据
        
        Args:
            text: 要合成的文本
            voice: 发音人，默认为聆飞逸
            audio_encoding: 音频编码格式，默认为lame(mp3)
            sample_rate: 采样率，默认24000
            
        Returns:
            请求数据字典
        """
        request_data = {
            "header": {
                "app_id": self.app_id,
                "status": 2  # 2表示结束（一次性合成）
            },
            "parameter": {
                "oral": {
                    "oral_level": "mid"  # 口语化等级：高(high), 中(mid), 低(low)
                },
                "tts": {
                    "vcn": voice,  # 发音人
                    "speed": 50,   # 语速：0-100
                    "volume": 50,  # 音量：0-100
                    "pitch": 50,   # 语调：0-100
                    "bgs": 0,      # 背景音
                    "reg": 0,      # 英文发音方式
                    "rdn": 0,      # 数字发音方式
                    "rhy": 0,      # 是否返回拼音
                    "audio": {
                        "encoding": audio_encoding,  # 音频编码
                        "sample_rate": sample_rate,  # 采样率
                        "channels": 1,               # 声道数
                        "bit_depth": 16,             # 位深
                        "frame_size": 0              # 帧大小
                    }
                }
            },
            "payload": {
                "text": {
                    "encoding": "utf8",
                    "compress": "raw",
                    "format": "plain",
                    "status": 2,  # 2表示结束（一次性合成）
                    "seq": 0,     # 数据序号
                    "text": base64.b64encode(text.encode('utf-8')).decode('utf-8')
                }
            }
        }
        
        return request_data
    
    def on_message(self, _ws, message):
        """WebSocket消息处理"""
        try:
            response = json.loads(message)
            
            # 检查响应状态
            if response.get("header", {}).get("code") != 0:
                error_msg = response.get("header", {}).get("message", "Unknown error")
                print(f"TTS API错误: {error_msg}")
                return
            
            # 提取音频数据
            payload = response.get("payload", {})
            audio_info = payload.get("audio", {})
            
            if "audio" in audio_info:
                audio_base64 = audio_info["audio"]
                audio_bytes = base64.b64decode(audio_base64)
                self.audio_data.append(audio_bytes)
                print(f"接收到音频数据块，大小: {len(audio_bytes)} 字节")
            
            # 检查是否接收完毕
            status = audio_info.get("status", 0)
            if status == 2:  # 2表示结束
                print("音频数据接收完毕")
                if self.ws:
                    self.ws.close()
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"处理响应消息时出错: {e}")
    
    def on_error(self, _ws, error):
        """WebSocket错误处理"""
        print(f"WebSocket错误: {error}")
    
    def on_close(self, _ws, _close_status_code, _close_msg):
        """WebSocket连接关闭处理"""
        print("WebSocket连接已关闭")
        self.connected = False
    
    def on_open(self, _ws):
        """WebSocket连接建立处理"""
        print("WebSocket连接已建立")
        self.connected = True
    
    def synthesize_speech(self, text: str, output_file: str = "output.mp3", 
                         voice: str = "x5_lingfeiyi_flow") -> bool:
        """
        合成语音
        
        Args:
            text: 要合成的文本
            output_file: 输出文件路径
            voice: 发音人
            
        Returns:
            是否成功
        """
        try:
            # 清空之前的音频数据
            self.audio_data = []
            
            # 生成认证URL
            auth_url = assemble_auth_url(self.host_url, self.api_key, self.api_secret)
            print(f"连接URL: {auth_url}")
            
            # 创建WebSocket连接
            websocket.enableTrace(False)  # 可以设置为True来调试
            self.ws = websocket.WebSocketApp(
                auth_url,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            
            # 在新线程中运行WebSocket
            def run_websocket():
                if self.ws:
                    self.ws.run_forever()
            
            ws_thread = threading.Thread(target=run_websocket)
            ws_thread.daemon = True
            ws_thread.start()
            
            # 等待连接建立
            timeout = 10  # 10秒超时
            start_time = time.time()
            while not self.connected and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            if not self.connected:
                print("连接超时")
                return False
            
            # 发送TTS请求
            request_data = self.create_request_data(text, voice)
            if self.ws:
                self.ws.send(json.dumps(request_data))
                print(f"已发送TTS请求，文本: {text}")
            
            # 等待音频数据接收完毕
            start_time = time.time()
            while self.connected and time.time() - start_time < 30:  # 30秒超时
                time.sleep(0.1)
            
            # 保存音频文件
            if self.audio_data:
                audio_content = b''.join(self.audio_data)
                with open(output_file, 'wb') as f:
                    f.write(audio_content)
                print(f"音频文件已保存: {output_file}")
                print(f"音频大小: {len(audio_content)} 字节")
                return True
            else:
                print("未接收到音频数据")
                return False
                
        except (ConnectionError, OSError, websocket.WebSocketException) as e:
            print(f"语音合成失败: {e}")
            return False


def main():
    """
    主函数 - 演示TTS功能
    """
    # 从 .env 获取的个人信息
    load_dotenv()  # 加载环境变量

    APPID = os.getenv("XF_APPID")
    API_SECRET = os.getenv("XF_API_SECRET")
    API_KEY = os.getenv("XF_API_KEY")

    # 创建TTS客户端
    tts_client = XunfeiTTS(APPID, API_KEY, API_SECRET)
    
    # 要合成的文本
    test_text = "欢迎使用讯飞超拟人语音合成技术！这是一个测试样例。"
    
    # 选择发音人（需要在控制台开通权限）
    # 可用发音人: x5_lingfeiyi_flow(聆飞逸-成年男声), x5_lingyuyan_flow(聆玉言-成年女声) 等
    selected_voice = "x5_lingfeiyi_flow"  # 默认聆飞逸
    
    print("开始语音合成...")
    print(f"文本: {test_text}")
    print(f"发音人: {selected_voice}")
    
    # 生成输出文件名
    output_file = "tts_sample.mp3"
    
    # 执行语音合成
    success = tts_client.synthesize_speech(
        text=test_text,
        output_file=output_file,
        voice=selected_voice
    )
    
    if success:
        print(f"\n语音合成成功！输出文件: {output_file}")
        
        # 检查文件是否存在
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"文件大小: {file_size} 字节")
            
            # 如果是Linux系统，尝试播放音频
            if os.name == 'posix':  # Unix/Linux系统
                print("\n尝试播放音频文件...")
                try:
                    os.system(f"which mpg123 && mpg123 {output_file} || echo '请安装mpg123播放器: sudo apt-get install mpg123'")
                except OSError:
                    print("无法播放音频，请手动播放文件")
        else:
            print("文件未找到，可能生成失败")
    else:
        print("语音合成失败")


if __name__ == "__main__":
    main()
