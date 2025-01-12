newp="/home/santhi/Documents/DACAT/src/Cholec80/data/frames_1fps/"
videop="/home/santhi/Documents/data_cholec80/videos/video"

seq 1 80 | parallel -j $(nproc) ' \
    mod={}; \
    curvideop='"${newp}"'$(printf "%02d" $mod); \
    mkdir -p $curvideop; \
    ffmpeg -hide_banner -hwaccel cuda -i '"${videop}"'$(printf "%02d" $mod).mp4 -r 1 -preset ultrafast -start_number 0 $curvideop/%08d.jpg \
'
