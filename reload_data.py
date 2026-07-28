import urllib.request
import sys

sys.stdout.reconfigure(encoding='utf-8')

file_path = r'c:\Users\Administrator\Desktop\ai agent\uploads\classified_diffusion_data.csv'
url = 'http://localhost:8002/api/upload'
boundary = '----FormBoundary7MA4YWxkTrZu0gW'

with open(file_path, 'rb') as f:
    file_data = f.read()

part1 = '--' + boundary + '\r\n'
part1 += 'Content-Disposition: form-data; name="file"; filename="classified_diffusion_data.csv"\r\n'
part1 += 'Content-Type: text/csv\r\n\r\n'
part2 = '\r\n--' + boundary + '--\r\n'

body = part1.encode('utf-8') + file_data + part2.encode('utf-8')

req = urllib.request.Request(
    url,
    data=body,
    headers={'Content-Type': 'multipart/form-data; boundary=' + boundary}
)

try:
    resp = urllib.request.urlopen(req, timeout=120)
    print('Success:', resp.read().decode('utf-8'))
except urllib.error.HTTPError as e:
    print('HTTP Error:', e.code, e.read().decode('utf-8'))
except Exception as e:
    print('Error:', e)
