#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
iFlytek 中英识别大模型 ASR API 客户端 (优化版)
大幅提升处理速度，减少不必要的延迟
"""

import asyncio
import websockets
import json
import base64
import wave
import time
from xunfei_auth import assemble_auth_url


class iFlyTekASR:
    def __init__(self, app_id: str, api_key: str, api_secret: str):
        """
        初始化iFlytek ASR客户端 (优化版)
        
        Args:
            app_id: 应用ID
            api_key: API密钥
            api_secret: API密钥对应的Secret
        """
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        
        # 新版API地址
        self.host_url = "wss://iat.xf-yun.com/v1"
        
        # 优化的音频配置
        self.audio_config = {
            "encoding": "raw",  # PCM格式
            "sample_rate": 16000,  # 16kHz采样率
            "channels": 1,  # 单声道
            "bit_depth": 16,  # 16位深度
        }
        
        # 识别结果
        self.result_text = ""
        self.final_results = {}  # 存储最终结果
        self.partial_results = {}  # 存储部分结果
        
    def get_auth_url(self) -> str:
        """生成认证URL"""
        return assemble_auth_url(self.host_url, self.api_key, self.api_secret)
    
    def create_first_frame(self, seq: int = 1) -> str:
        """创建第一帧数据包"""
        frame = {
            "header": {
                "app_id": self.app_id,
                "res_id": "hot_words",
                "status": 0  # 首帧
            },
            "parameter": {
                "iat": {
                    "domain": "slm",  # 大模型领域
                    "language": "zh_cn",  # 中文
                    "accent": "mandarin",  # 普通话
                    "eos": 6000,  # 静音6秒停止
                    "vinfo": 1,  # 句子级别帧对齐
                    "dwa": "wpgs",  # 流式识别PGS，返回速度更快
                    "result": {
                        "encoding": "utf8",
                        "compress": "raw",
                        "format": "json"
                    }
                }
            },
            "payload": {
                "audio": {
                    "encoding": self.audio_config["encoding"],
                    "sample_rate": self.audio_config["sample_rate"],
                    "channels": self.audio_config["channels"],
                    "bit_depth": self.audio_config["bit_depth"],
                    "seq": seq,
                    "status": 0,  # 开始
                    "audio": ""  # 第一帧不包含音频数据
                }
            }
        }
        return json.dumps(frame, ensure_ascii=False)
    
    def create_audio_frame(self, audio_data: bytes, seq: int, status: int) -> str:
        """创建音频数据帧"""
        # 将音频数据编码为base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        if status == 1:  # 中间帧
            frame = {
                "header": {
                    "app_id": self.app_id,
                    "res_id": "hot_words",
                    "status": 1  # 中间帧
                },
                "payload": {
                    "audio": {
                        "encoding": self.audio_config["encoding"],
                        "sample_rate": self.audio_config["sample_rate"],
                        "channels": self.audio_config["channels"],
                        "bit_depth": self.audio_config["bit_depth"],
                        "seq": seq,
                        "status": 1,  # 继续
                        "audio": audio_base64
                    }
                }
            }
        else:  # 最后一帧
            frame = {
                "header": {
                    "app_id": self.app_id,
                    "res_id": "hot_words",
                    "status": 2  # 最后一帧
                },
                "payload": {
                    "audio": {
                        "encoding": self.audio_config["encoding"],
                        "sample_rate": self.audio_config["sample_rate"],
                        "channels": self.audio_config["channels"],
                        "bit_depth": self.audio_config["bit_depth"],
                        "seq": seq,
                        "status": 2,  # 结束
                        "audio": ""  # 最后一帧不包含音频数据
                    }
                }
            }
        
        return json.dumps(frame, ensure_ascii=False)
    
    def read_wav_file(self, file_path: str) -> tuple:
        """读取WAV文件并转换为16kHz单声道PCM"""
        with wave.open(file_path, 'rb') as wav_file:
            # 获取音频参数
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.getnframes()
            
            print(f"原始音频参数: {sample_rate}Hz, {channels}声道, {sample_width*8}位, 时长: {frames/sample_rate:.2f}秒")
            
            # 读取音频数据
            audio_data = wav_file.readframes(frames)
            
            # 如果是44.1kHz，需要降采样到16kHz
            if sample_rate == 44100:
                import numpy as np
                
                # 转换为numpy数组
                if sample_width == 2:  # 16位
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                else:
                    raise ValueError(f"不支持的位深度: {sample_width*8}位")
                
                # 如果是立体声，转换为单声道
                if channels == 2:
                    audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(np.int16)
                
                # 降采样到16kHz
                resample_ratio = 16000 / 44100
                target_length = int(len(audio_array) * resample_ratio)
                indices = np.linspace(0, len(audio_array) - 1, target_length).astype(int)
                resampled_audio = audio_array[indices]
                
                audio_data = resampled_audio.tobytes()
                sample_rate = 16000
                channels = 1
                
                print(f"转换后音频参数: {sample_rate}Hz, {channels}声道, 16位, 数据大小: {len(audio_data)}字节")
            
            elif sample_rate != 16000:
                raise ValueError(f"不支持的采样率: {sample_rate}Hz，请使用16kHz或44.1kHz")
            
            return audio_data, sample_rate, channels, sample_width
    
    def decode_result_text(self, text_base64: str) -> str:
        """解码识别结果文本"""
        try:
            # 解码base64
            decoded_bytes = base64.b64decode(text_base64)
            decoded_str = decoded_bytes.decode('utf-8')
            
            # 解析JSON
            result_json = json.loads(decoded_str)
            
            # 提取文本
            text_parts = []
            if 'ws' in result_json:
                for word_segment in result_json['ws']:
                    if 'cw' in word_segment:
                        for chinese_word in word_segment['cw']:
                            if 'w' in chinese_word:
                                text_parts.append(chinese_word['w'])
            
            return ''.join(text_parts)
            
        except Exception as e:
            print(f"解码结果时出错: {e}")
            return ""
    
    async def send_audio_data(self, websocket, audio_data):
        """发送音频数据 (异步，不等待响应)"""
        # 优化的帧大小：增加到6400字节 (200ms @ 16kHz)
        # 这样可以减少网络包的数量，提高效率
        frame_size = 6400  # 200ms音频数据
        seq = 2
        
        print(f"开始发送音频数据，总大小: {len(audio_data)}字节，帧大小: {frame_size}字节")
        
        for i in range(0, len(audio_data), frame_size):
            chunk = audio_data[i:i + frame_size]
            
            # 判断是否是最后一块
            is_last = (i + frame_size >= len(audio_data))
            status = 2 if is_last else 1
            
            # 创建并发送音频帧
            audio_frame = self.create_audio_frame(chunk, seq, status)
            await websocket.send(audio_frame)
            
            if is_last:
                print(f"已发送最后一帧 (序号: {seq}, 总帧数: {seq-1})")
            else:
                if seq % 10 == 0:  # 每10帧打印一次进度
                    progress = (i + frame_size) / len(audio_data) * 100
                    print(f"发送进度: {progress:.1f}% (序号: {seq})")
            
            seq += 1
            
            # 大幅减少延迟：仅在必要时添加很小的延迟
            if not is_last:
                await asyncio.sleep(0.001)  # 1ms延迟，而不是40ms
    
    async def receive_results(self, websocket):
        """接收识别结果 (异步)"""
        try:
            while True:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                response_json = json.loads(response)
                
                # 检查响应状态
                header = response_json.get('header', {})
                if header.get('code') != 0:
                    print(f"错误响应: {response}")
                    continue
                
                # 提取识别结果
                if 'payload' in response_json and 'result' in response_json['payload']:
                    result = response_json['payload']['result']
                    if 'text' in result:
                        text = self.decode_result_text(result['text'])
                        seq_num = result.get('seq', 0)
                        status = result.get('status', 0)
                        
                        if text:
                            if status == 2:  # 最终结果
                                self.final_results[seq_num] = text
                                print(f"✅ 最终结果 (seq {seq_num}): {text}")
                            else:  # 部分结果
                                self.partial_results[seq_num] = text
                                print(f"⏳ 部分结果 (seq {seq_num}): {text}")
                
                # 检查是否完成
                if header.get('status') == 2:
                    print("🎉 识别完成!")
                    break
                    
        except asyncio.TimeoutError:
            print("⚠️ 接收结果超时，但继续等待...")
        except websockets.exceptions.ConnectionClosed:
            print("📡 WebSocket连接已关闭")
    
    async def transcribe_audio(self, audio_file_path: str) -> str:
        """转录音频文件 (优化版)"""
        start_time = time.time()
        
        # 读取音频文件
        audio_data, sample_rate, channels, sample_width = self.read_wav_file(audio_file_path)
        
        # 获取认证URL
        auth_url = self.get_auth_url()
        print(f"🔗 连接到: {auth_url}")
        
        # 连接WebSocket
        async with websockets.connect(auth_url) as websocket:
            print("✅ WebSocket连接成功")
            
            # 发送第一帧
            first_frame = self.create_first_frame(seq=1)
            await websocket.send(first_frame)
            print("📤 已发送第一帧")
            
            # 接收第一帧响应
            response = await websocket.recv()
            response_json = json.loads(response)
            if response_json.get('header', {}).get('code') != 0:
                raise Exception(f"第一帧响应错误: {response}")
            print("📥 收到第一帧响应")
            
            # 并发发送音频数据和接收结果
            send_task = asyncio.create_task(self.send_audio_data(websocket, audio_data))
            receive_task = asyncio.create_task(self.receive_results(websocket))
            
            # 等待两个任务完成
            await asyncio.gather(send_task, receive_task)
        
        # 合并结果
        all_results = {}
        
        # 收集所有结果（部分结果通常更完整）
        all_results.update(self.partial_results)
        all_results.update(self.final_results)
        
        if all_results:
            # 取最长的完整识别结果，排除纯标点符号
            best_result = ""
            for seq_num in sorted(all_results.keys()):
                result = all_results[seq_num].strip()
                # 排除空结果和纯标点符号
                if len(result) > len(best_result) and result not in ['。', '，', '']:
                    best_result = result
            
            self.result_text = best_result
        
        end_time = time.time()
        processing_time = end_time - start_time
        audio_duration = len(audio_data) / (16000 * 2)  # 16kHz, 16bit
        speed_ratio = audio_duration / processing_time
        
        print(f"⏱️ 处理时间: {processing_time:.2f}秒")
        print(f"🎵 音频时长: {audio_duration:.2f}秒") 
        print(f"🚀 处理速度: {speed_ratio:.2f}x 实时速度")
        
        return self.result_text


async def main():
    """主函数"""
    # iFlytek API配置
    APPID = "27c57829"
    API_SECRET = "OWYwMjUyMjgzNWY2OGM3N2E0MjI5M2U5"
    API_KEY = "12ba58ebdf50c172e6f03d2bbc830c39"
    
    # 音频文件路径
    AUDIO_FILE = "/home/whao/Developer/2025-Anhui-AI-Speech-Technology-Competition/data/asr.wav"
    
    # 创建ASR客户端
    asr_client = iFlyTekASR(APPID, API_KEY, API_SECRET)
    
    print("🚀 开始音频转录 (优化版)...")
    print(f"📁 音频文件: {AUDIO_FILE}")
    print("-" * 60)
    
    try:
        # 执行转录
        result = await asr_client.transcribe_audio(AUDIO_FILE)
        
        print("-" * 60)
        print(f"🎯 转录完成!")
        print(f"📝 最终结果: {result}")
        
        # 保存结果到文件
        output_file = "/home/whao/Developer/2025-Anhui-AI-Speech-Technology-Competition/asr_result.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"💾 结果已保存到: {output_file}")
        
    except Exception as e:
        print(f"❌ 转录过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
