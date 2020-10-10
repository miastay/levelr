from pytube import YouTube
import sys, time

videoid = sys.argv[1] + ''
vid = YouTube('https://www.youtube.com/watch?v=' + videoid)

stream = vid.streams.filter(only_audio=True)[0]

print(vid.thumbnail_url + "~~~" + videoid)
print(stream.url  + "~~~" + videoid)