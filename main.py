import json
import logging
import os
import re
import subprocess
from os import mkdir, makedirs

import json5
import yt_dlp
import dotenv
import requests

from pathlib import Path

from openai import OpenAI
from faster_whisper import WhisperModel, BatchedInferencePipeline

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
logging.getLogger('httpx').setLevel(logging.WARNING)

ffmpeg_dir="D:/ffmpeg-6.1.1-essentials_build"

model_llama3 = "llama3.1:8b"
model_qwen25_7b = "qwen2.5:7b"
model_qwen25_14b = "qwen2.5:14b"
model_deepseek_7b = "deepseek-r1:7b"
model_deepseek_14b = "deepseek-r1:14b"
model=model_qwen25_14b

client = OpenAI(
    base_url='http://192.168.31.10:11434/v1',
    api_key='ollama',
)

def ollama_translate(sentences: list, language_from='english', language_to='chinese') -> list:
    json_text = json5.dumps(sentences, ensure_ascii=False)
    messages=[
        {
            "role": "system",
            "content": """
                    You are a professional, real translation engine. Strictly follow the rules given in <rule>. You will be given input and steps to follow to translate the text without any explanation.
                    Please provide your answer in the following <output-format> and without any additional information.
            """
        },
        {
            "role": "user",
            "content": f"""
                 <input>
                     {json_text}
                 </input>
                 
                 <output-format>
                     [
                        {{
                            "text": "<<original text in {language_from}>>",
                            "step1": "<<translated text in {language_to}>>",
                            "step2": "<<refined translation in {language_to}>>"
                        }},
                        {{
                            "text": "<<original text in {language_from}>>",
                            "step1": "<<translated text in {language_to}>>",
                            "step2": "<<refined translation in {language_to}>>"
                        }},
                        ...
                     ]
                 </output-format>
                 
                 <steps>
                    1. Extract the content from the "text" field in the provided json object. Keep the original {language_from} text in the "text" field.
                    2. Translate the extracted content into {language_to} and place into the "step1" field.
                    3. Refine the initial translation from "step1" to make it more natural and understandable in {language_to}. Place this refined translation into the "step2" field.
                 </steps>
                 
                 <rules>
                    1. Do output LaTeX. 
                    2. Do not output something like '\\(x\\)'. 
                    3. Do not split professional vocabulary. 
                    4. Do not translate or modify proprietary words, program variables, function names. such as 'y_hat', 'f(x)', 'f(x)=wx+b'. 
                 </rules>
            """
        },
    ]

    # 如果报错, 重试3次
    for i in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                # temperature=0.5,
                messages=messages,
            )
            logging.info(response.choices[0].message.content)
            return json5.loads(response.choices[0].message.content)
        except Exception as e:
            logging.error("execute ollama_translate failed, retry %d, error %s" % (i, e))


def ollama_segments_analysis(text: str) -> str:
    messages=[
        {
            "role": "system",
            "content": """
                Your task is to split the provided sentence into shorter sentences so that they can be displayed in the video subtitles.
                You need to split the text given in the <input> tag according to punctuation or sentence meaning. 
                The key is to ensure that each sentence has less than 15 words, but it is more important to keep the semantic integrity of a single sentence.
                Several examples are provided for your reference in <example>.
                Please provide your answer in the following <output-format> and without any additional information.
            """
        },
        {
            "role": "user",
            "content": f"""
                <input>
                    {text}
                </input>
                <output-format>
                    {{
                        "split": ["<<Output each sentence in sequence.>>"]
                    }}
                </output-format>
                <rules>
                    0. Your task is to split the provided sentence into shorter sentences so that they can be displayed in the video subtitles.
                    1. First segment the text at punctuation marks, such as period, comma, question mark, etc. (such as ",", ".", "?", "!" etc.)
                    2. You can also segment at conjunctions (such as "and", "but", "because", "when", "then", "if", "so", "that"). 
                    3. It can also be divided according to sentence structure, such as long inverted sentences, subordinate clauses, etc.
                    4. Output each sentence in sequence. For example, ["this is the first part,", "this is the second part."]
                    5. Please provide your answer in the following <output-format> and without any additional information.
                </rules>
                <examples>
                    <example>
                        <input>
                            By now you know that I am the developer of the ConfUI IP Adapter extension and now also of the Instant ID one. Instant ID is a style transfer model targeted to people's portraits. Be careful because there are many Instant ID extensions in the manager, but mine is the only native one, meaning that it is fully
                        </input>
                        <output>
                            {{
                                "split": ["By now you know that", "I am the developer of the ConfUI IP Adapter extension", "and now also of the Instant ID one.", "Instant ID is a style transfer model targeted to people's portraits.", "Be careful because there are many Instant ID extensions in the manager,", "but mine is the only native one,", "meaning that it is fully"]
                            }}
                        </output>
                    </example>
                    <example>
                        <input>
                            By now you know that I am the developer of the ConfUI IP Adapter extension and now also of the Instant ID one. Instant ID is a style transfer model targeted to people's portraits. Be careful because there are many Instant ID extensions in the manager, but mine is the only native one, meaning that it is fully
                        </input>
                        <output>
                            {{
                                "split": ["By now you know that", "I am the developer of the ConfUI IP Adapter extension", "and now also of the Instant ID one.", "Instant ID is a style transfer model targeted to people's portraits.", "Be careful because there are many Instant ID extensions in the manager,", "but mine is the only native one,", "meaning that it is fully"]
                            }}
                        </output>
                    </example>
                </examples>
            """.strip()
        },
    ]
    for i in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                # temperature=0.8,
                messages=messages,
            )
            content = response.choices[0].message.content
            logging.info(content)

            # # 把<think></think>标签中的内容和标签都删除
            # content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            # content = re.sub(r"```json(.*?)```", r"\1", content, flags=re.DOTALL)
            # logging.info(content)

            return json5.loads(content)
        except Exception as e:
            logging.error("execute ollama_translate failed, retry %d, error %s" % (i, e))

def get_whisper_segments(audio: str, language_from="english"):
    language_map = {
        "english": "en",
        "japanese": "ja"
    }
    language = language_map.get(language_from, "en")

    model_map = {
        "english": "./models/faster-distil-whisper-large-v3",
        "japanese": "./models/faster-whisper-large-v3"
    }
    model_path = model_map.get(language_from, "./models/faster-whisper-large-v3")

    # Run on GPU with FP16
    model = WhisperModel(
        model_size_or_path=model_path,
        device="cuda",
        compute_type="float16",
        local_files_only=True,
        # num_workers=4,
    )

    batched_model = BatchedInferencePipeline(model=model, )
    segments, info = batched_model.transcribe(
        audio=audio,
        word_timestamps=True,
        batch_size=16,
        language=language,
        length_penalty=1,
        beam_size=5,
        # max_new_tokens=10,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    logging.info("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    whisper_result = [
        {
            "text": segment.text,
            "start": segment.start,
            "end": segment.end,
            "words": [
                {
                    "word": word.word,
                    "start": word.start,
                    "end": word.end
                } for word in segment.words]
        } for segment in segments]
    logging.info("whisper finish, segments count: %d" % len(whisper_result))
    return whisper_result

def load_or_execute(file_name, func, *args, **kwargs):
    """
    加载或创建JSON文件。

    参数:
    file_name (str): 文件名。
    func (callable): 闭包或方法，当文件不存在时调用。
    *args: 传递给闭包的 positional arguments。
    **kwargs: 传递给闭包的 keyword arguments。

    返回:
    dict: 文件内容或闭包的返回结果。
    """
    if os.path.exists(file_name):
        logging.info("file exists, load %s" % file_name)
        # 文件存在，读取并反序列化
        with open(file_name, 'r', encoding='utf-8') as file:
            return json5.load(file)
    else:
        logging.info("file not exists, execute %s" % func.__name__)
        # 文件不存在，执行闭包并保存结果
        # 报错就重试, 3次
        result = func(*args, **kwargs)
        with open(file_name, 'w', encoding='utf-8') as file:
            json.dump(result, file, ensure_ascii=False, indent=4)
        return result

# 定位每段的开始和结束
def find_sentences_times(sentences, words):
    word_index = 0
    for sentence in sentences:
        start_time = None
        end_time = None
        sentence_pos = 0
        sentence_text = sentence["text"]

        while word_index < len(words) and sentence_pos < len(sentence_text):
            word_time = words[word_index]
            word = word_time["word"].strip(" ,.!?")

            # 查找单词在句子中的位置
            word_pos = sentence_text.find(word, sentence_pos)

            if word_pos != -1:
                if start_time is None:
                    start_time = word_time["start"]
                end_time = word_time["end"]

                # 更新句子位置指针，跳过已匹配的部分
                sentence_pos = word_pos + len(word)
                word_index += 1
            elif start_time is None:
                # 大模型拆句子有可能把某些句子或者单词扔掉了, 先忽略
                word_index += 1
            else:
                break

        if start_time is not None and end_time is not None:
            sentence["start"] = start_time
            sentence["end"] = end_time
        else:
            logging.info(f"can't find sentence start or end: {sentence}")
    return sentences

def seconds_to_srt_time(seconds):
    # 分离小时
    hours = int(seconds // 3600)
    # 剩余秒数
    seconds %= 3600
    # 分离分钟
    minutes = int(seconds // 60)
    # 剩余秒数
    seconds %= 60
    # 分离毫秒
    milliseconds = int((seconds - int(seconds)) * 100)
    # 格式化输出
    return f"{hours:01d}:{minutes:02d}:{int(seconds):02d}.{milliseconds:02d}"

def write_subtitle(segments_analysis, subtitle_file):
    global index, analysis
    with open(f"{subtitle_file}", 'w', encoding='utf-8') as f:
        f.write("""[Script Info]
; This is an Advanced Substation Alpha v4+ script.
Title: Untitled
ScriptType: v4.00+
Collisions: Normal
PlayDepth: 0
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Chinese,SimHei,16,&H0000D9FC,&H00000000,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,4,1,2,10,10,10,1
Style: English,Arial,12,&H00FFFFFF,&H00000000,&H00000000,&H80000000,0,1,0,0,100,100,0,0,1,4,1,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""")
        for index, analysis in enumerate(segments_analysis):
            for i, sub in enumerate(analysis['sentences']):
                start_time = seconds_to_srt_time(sub["start"])
                if i + 1 < len(analysis['sentences']):
                    end_time = seconds_to_srt_time(analysis['sentences'][i + 1]["start"])
                elif index + 1 < len(segments_analysis):
                    end_time = seconds_to_srt_time(segments_analysis[index + 1]['sentences'][0]["start"])
                else:
                    end_time = seconds_to_srt_time(sub["end"])
                english_text = sub["text"]
                chinese_text = sub["step1"] if sub.get("step2", "") == "" else sub["step2"]
                # 将标点符号后面追加 \N
                chinese_text = re.sub(r'([，。、！？）：,.!?:’])(?=.)', '\\1 ', chinese_text)
                mid_index = len(chinese_text) // 2
                chinese_text = chinese_text[:mid_index] + ' ' + chinese_text[mid_index:] if len(chinese_text) > 35 and ' ' not in chinese_text else chinese_text

                f.write(f"Dialogue: 0,{start_time},{end_time},English,,0,0,0,,{{\\fad(100,100)}}{english_text}\n")
                f.write(f"Dialogue: 0,{start_time},{end_time},Chinese,,0,0,0,,{{\\fad(100,100)}}{chinese_text}\n")

def embed_subtitles(input_video, subtitle_file, output_video, skip_time):
    # 构建 ffmpeg 命令
    command = [
        f'{ffmpeg_dir}/bin/ffmpeg', '-y',
        '-hwaccel', 'cuda',
        '-i', f'{input_video}',
        '-ss', f'{skip_time}',
        '-vf', f"subtitles=\\'{subtitle_file}\\'",
        '-threads', '24',
        '-c:v', 'h264_nvenc',
        '-c:a', 'copy',
        '-stats',
        f'{output_video}'
    ]

    # 调用 ffmpeg 命令
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    for line in process.stderr:
        logging.info(line.strip())
    process.wait()
    logging.info("ffmpeg process completed.")

def crop_video(input_video, crop):
    output_video = os.path.splitext(input_video)[0] + "_cropped.mp4"
    # 构建 ffmpeg 命令
    command = [
        f'{ffmpeg_dir}/bin/ffmpeg', '-y',
        '-hwaccel', 'cuda',
        '-i', f'{input_video}',
        '-vf', f"crop={crop},scale=1280:720",
        '-threads', '24',
        '-c:v', 'h264_nvenc',
        '-c:a', 'copy',
        '-stats',
        f'{output_video}'
    ]

    # 调用 ffmpeg 命令
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    for line in process.stderr:
        logging.info(line.strip())
    process.wait()
    logging.info("ffmpeg process completed.")
    return output_video

def download_video(vieo_id, output_dir):
    url = f"https://www.youtube.com/watch?v={vieo_id}"
    proxy = os.getenv("YT_DLP_PROXY")

    # 配置 yt-dlp 用于提取元信息
    ydl_opts = {
        'skip_download': True,  # 只提取信息，不下载
        'proxy': proxy,
    }

    filename = None
    thumbnail_url = None
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)  # 获取视频信息
        logging.info({
            "title": info['title'],
            "thumbnail": info['thumbnail'],
        })

        thumbnail_url = info['thumbnail']

        # 根据模板生成文件名
        filename = f"{info['title']}.{info['ext'] or 'mp4'}"
        filename = ydl.prepare_filename(outtmpl= rf"{output_dir}\{filename}", info_dict=info)

    thumbnail_suffix = os.path.splitext(thumbnail_url)[1]
    thumbnail_suffix = thumbnail_suffix.split('?')[0]
    thumbnail = rf"{output_dir}\{"thumbnail" + thumbnail_suffix}"

    # 不存在则下载
    if not os.path.exists(thumbnail):
        logging.info(f"下载视频缩略图 {thumbnail_url}")
        response = requests.get(thumbnail_url, proxies={'https': proxy})
        if response.status_code == 200:
            with open(thumbnail, "wb") as f:
                f.write(response.content)
            logging.info(f"封面已下载为 {thumbnail}")
        else:
            logging.info(f"下载视频缩略图失败: {response}")

    if os.path.exists(filename):
        logging.info(f"文件 {filename} 已存在，跳过下载。")
        return filename

    # 存放下载后文件名
    downloaded_file = {'filename': None}
    # 定义回调函数
    def progress_hook(d):
        if d['status'] == 'finished':  # 下载完成时获取文件名
            downloaded_file['filename'] = d['info_dict']['filename']

    resolution = 720
    yt_options = {
        'outtmpl': filename,
        'format': f'bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]',
        "extractor_retries": 1,
        "proxy": proxy,
        'progress_hooks': [progress_hook],
        'postprocessor_hooks': [progress_hook],
    }
    with yt_dlp.YoutubeDL(yt_options) as ydl:
        ydl.download([url])

    # 返回最终下载文件的路径
    return downloaded_file['filename']

def flat_playlist(playlist_id):
    playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"
    proxy = os.getenv("YT_DLP_PROXY")

    # 配置 yt-dlp 用于提取元信息
    ydl_opts = {
        'skip_download': True,  # 只提取信息，不下载
        'extract_flat': True,
        "proxy": proxy,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        playlist_info = ydl.extract_info(playlist_url, download=False)
        if 'entries' in playlist_info:
            return [entry for entry in playlist_info['entries']]

def process_video(video_id, language_from='english', start_time=0):
    # 每个视频一个缓存文件夹
    cache_dir = rf".cache\{video_id}"
    makedirs(name=f"{cache_dir}", mode=755, exist_ok=True)

    # 下载文件
    input_video = download_video(video_id, cache_dir)

    # 提取音频文字
    whisper_result = load_or_execute(f"{cache_dir}/whisper_result.json", get_whisper_segments, input_video, language_from)

    # 文本分段
    segments_analysis = load_or_execute(f"{cache_dir}/ollama_segments_analysis.json", lambda rs: [{**segment, **ollama_segments_analysis(segment["text"])} for segment in rs], whisper_result)
    for index, analysis in enumerate(segments_analysis):
        analysis['sentences'] = [{"text": text.strip(), "step1": "", "step2": ""} for text in analysis["split"]]

    # 翻译
    segments_analysis = load_or_execute(f"{cache_dir}/ollama_translate.json", lambda objs: [{**segment, "sentences": ollama_translate(segment["sentences"], language_from)} for segment in objs], segments_analysis)

    # 查找句子时间
    for index, analysis in enumerate(segments_analysis):
        analysis['sentences']=find_sentences_times(analysis['sentences'], analysis["words"])

    # 写入字幕文件
    subtitle_file = f"{cache_dir}/subtitle.ass"
    write_subtitle(segments_analysis, subtitle_file)

    input_video_base_name = os.path.splitext(input_video)[0]

    # 去除 ‘.cache\\ggWLvh484hs\\’, 'ggWLvh484hs'可能为任意内容
    input_video_base_name = re.sub(r'^\.cache\\.+?\\', '', input_video_base_name)
    input_video_base_name = re.sub(r'[\'#]', '', input_video_base_name)
    input_video_base_name = input_video_base_name.strip()
    logging.info(f"input_video_base_name: {input_video_base_name}")

    title_translate = ollama_translate([{"text": input_video_base_name.strip(), "step1": "", "step2": ""}])
    output_video_base_name = "[双语字幕]" + title_translate[0]['step1']

    # 嵌入字幕
    embed_subtitles(Path(input_video).absolute(), Path(subtitle_file).absolute().as_posix(), rf"{Path(cache_dir).absolute()}\{output_video_base_name}.mp4", start_time)

    return

if __name__ == "__main__":
    # video_info_list = flat_playlist("")
    # print([video_info['id'] for video_info in video_info_list])

    ids = [
        ('tILXLxMTmgA', 0),
    ]
    for id, start_time in ids:
        process_video(id, "english", start_time)
