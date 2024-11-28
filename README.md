
### Server Side

should be using a container or a virtual environment
```
python3 server.py
```


### Client Side

"-video", "Path to the input video file"

"-server", "Server's ip address"

"-resize", "Enable resizing: on/off"

"-crop", "Enable cropping: on/off"

"-adversarial", "Adversarial attack: none/noise/blur/crbr"

```
python3 client.py -video video_1.mp4 -server localhost -resize off -crop off -adversarial noise
```
