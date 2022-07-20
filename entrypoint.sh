#! /bin/bash
sed -i "s/80/$NGINX_PORT/g" /etc/nginx/sites-enabled/default
service nginx start

# (nohup /root/.nvm/versions/node/v14.0.0/bin/node /root/.nvm/versions/node/v14.0.0/bin/json-server --watch parameters.json --host 0.0.0.0 > logs/json-server.log 2>&1 &)
cd myjson-server &&\
(nohup /root/.nvm/versions/node/v14.0.0/bin/node server.js > /phoenix/logs/json-server.log 2>&1 &) &&\
cd ..


FILE1=/opt/nvidia/deepstream/deepstream-5.0/sources/apps/sample_apps/smart_ppe/json-detection/json_to_stream.json
FILE2=/opt/nvidia/deepstream/deepstream-5.0/sources/apps/sample_apps/smart_ppe/json-detection/detection_results.json

if [ -f "$FILE1" ]; then
    > $FILE1 #If file exists remove
fi
if [ -f "$FILE2" ]; then
    > $FILE2 #Remove if exists
fi

#(nohup json-server --watch $FILE1 --port 2000 --host 0.0.0.0 > json-server.log &) &&\
(nohup /bin/bash json_validator.sh > json_validator.log &) 


#(nohup ./deepstream-rtsp-app $RTSP_STREAM > /phoenix/logs/main_code.log 2>&1 &)
(nohup ./deepstream-rtsp-app $RTSP_STREAM > /phoenix/logs/debug.log 2>&1 &)

export OPENBLAS_CORETYPE=ARMV8
cd json/snapshot && \
(nohup /root/.nvm/versions/node/v14.0.0/bin/node server.js > snapshot.log &) #SNAPSHOT

sleep infinity