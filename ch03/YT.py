# 下載YT影片

import os
from yt_dlp import YoutubeDL

# 輸入 YouTube 影片的 URL
url = input("請輸入 YouTube 影片的 URL: ")
a = input("1)mp4   2)mp3   請輸入: ")

if a == '1' :
    # 設定下載選項
    ydl_opts = {
        'format': 'bestvideo[height<=1080][vcodec^=avc1]+bestaudio/best[height<=1080][vcodec^=avc1]',
        'outtmpl': '%(title)s.%(ext)s',  # 輸出檔案名稱格式
        # 'outtmpl': 'down.mp4',  # 固定輸出檔名為 down.mp4
        'merge_output_format': 'mp4',  # 合併影音時使用 mp4 格式
        'quiet': True,  # 避免輸出過多資訊
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)  # 先獲取影片資訊
            filename = ydl.prepare_filename(info_dict)  # 取得預設檔名
            print(f"即將下載的檔案名稱: {filename}")

            ydl.download([url])  # 開始下載
            print(f"下載完成！檔案名稱: {filename}")
    except Exception as e:
        print(f"下載影片時發生錯誤: {e}")


    # https://youtu.be/XqJQlNGjdI8?si=5-he5Yzd-1cV2pVC

    # os.rename(filename,'down.mp4')
    # os.system('ffmpeg -i down.mp4 -c:v copy -c:a aac -b:a 192k output.mp4')
    # os.rename('output.mp4', filename)
    # os.remove('down.mp4')

else :
    ############################################################################################################

    # 下載 mp3 格式  只有聲音，音樂
    # 設定下載選項
    ydl_opts = {
        'format': 'bestaudio/best',  # 選擇最佳音訊品質
        'outtmpl': '%(title)s.%(ext)s',  # 輸出檔案名稱格式
        'postprocessors': [{  # 使用後處理器將音訊轉換為 MP3
            'key': 'FFmpegExtractAudio',  # 提取音訊
            'preferredcodec': 'mp3',       # 轉換為 MP3 格式
            'preferredquality': '192',     # 音訊品質 (可選，預設為 192kbps)
        }],
    }

    # 下載影片
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])