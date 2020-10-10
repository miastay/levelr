import sys
from sclib import SoundcloudAPI, Track, Playlist

api = SoundcloudAPI()  # never pass a Soundcloud client ID that did not come from this library

videourl = sys.argv[1] + ''
track = api.resolve(videourl)

assert type(track) is Track

#filename = f'./{track.artist} - {track.title}.mp3'
attrs = dir(track)
print(track.artwork_url)

print(track.get_stream_url())
print(track.title)
print(track.artist)
print(track.tag_list)
print(track.id)
'''
print(track.album_artwork)
print(track.artwork_url)
print(track.tag_list)

with open(filename, 'wb+') as fp:
    track.write_mp3_to(fp)
'''