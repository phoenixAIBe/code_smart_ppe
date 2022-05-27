#! /bin/bash
sed -i "s/80/$NGINX_PORT/g" /etc/nginx/sites-enabled/default
service nginx start

# (nohup /root/.nvm/versions/node/v14.0.0/bin/node /root/.nvm/versions/node/v14.0.0/bin/json-server --watch parameters.json --host 0.0.0.0 > logs/json-server.log 2>&1 &)
cd myjson-server &&\
(nohup /root/.nvm/versions/node/v14.0.0/bin/node server.js > /phoenix/logs/json-server.log 2>&1 &) &&\
cd ..

#(nohup ./deepstream-rtsp-app $RTSP_STREAM > /phoenix/logs/main_code.log 2>&1 &)
(nohup ./deepstream-rtsp-app $RTSP_STREAM > /phoenix/logs/debug.log 2>&1 &)

export OPENBLAS_CORETYPE=ARMV8
cd json/snapshot && \
(nohup /root/.nvm/versions/node/v14.0.0/bin/node server.js > snapshot.log &) #SNAPSHOT

sleep infinity