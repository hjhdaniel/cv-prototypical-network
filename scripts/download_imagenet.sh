function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  
rm -rf /tmp/cookies.txt
}

gdrive_download 0B3Irx3uQNoBMQ1FlNXJsZUdYWEE ../dataset/imagenet/images.zip

unzip -uq ../dataset/imagenet/images.zip
rm ../dataset/imagenet/images.zip
